//! `hall.zig`'s jobs and what the scheduler asks of its decoder: a request that is ready made a job
//! (`makeJob`, `releaseJob`, for `Scheduler.decode`), and the decoder's table (`admitJob`, `roundJobs` and the
//! round's calls), which `Hall` hands to `server/scheduler.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const request_file = @import("../../../server/request.zig");
const sched = @import("../../../server/scheduler.zig");
const hall_file = @import("hall.zig");
const chat = @import("../session/chat.zig");
const Hall = hall_file.Hall;

const Job = struct {
    /// The request as the decoder takes it. Its prompt, its messages' starts and the constraint's scratch are
    /// this job's, the constraint itself the caller's, and its gate is the checkpoint's `</think>` where the
    /// reply thinks.
    ask: hall_file.Ask,
};

fn jobOf(ptr: *sched.Job) *Job {
    return @ptrCast(@alignCast(ptr));
}

fn toJob(job: *Job) *sched.Job {
    return @ptrCast(job);
}

pub const decoder_table: sched.Decoder.Table = .{
    .admit = admitJob,
    .round = roundJobs,
    .retire = retireJob,
    .preempt = preemptJob,
};

/// A job of a request that is ready, for `Scheduler.decode`; `releaseJob` frees it. It copies the prompt and
/// `starts` and borrows the constraint and the image, which outlive the job and are the caller's to free.
pub fn makeJob(self: *Hall, ready: chat.Prepared, depth: usize, opening: usize, starts: []const usize, kept: ?usize, eos: ?[]const u32, image: ?hall_file.Picture) Allocator.Error!*sched.Job {
    const prompt = try self.gpa.dupe(i32, ready.prompt);
    errdefer self.gpa.free(prompt);
    const words: usize = if (ready.constraint) |held| held.words else 0;
    const mask = try self.gpa.alloc(u32, words);
    errdefer self.gpa.free(mask);
    const starts_kept = try self.gpa.dupe(usize, starts);
    errdefer self.gpa.free(starts_kept);
    const job = try self.gpa.create(Job);
    job.* = .{
        .ask = .{
            .prompt = prompt,
            .settings = ready.settings,
            .max_tokens = ready.max_tokens,
            .depth = depth,
            .opening = opening,
            .constraint = ready.constraint,
            .mask = mask,
            .gate = if (ready.thinking) self.think_close else null,
            .starts = starts_kept,
            .kept = kept,
            .eos = eos,
            .logprobs = ready.logprobs,
            .image = image,
        },
    };
    return toJob(job);
}

/// `job` is a warm pass (`hall_warm.zig`): run only while nothing else is live, its one token sent nobody.
pub fn warmJob(job: *sched.Job) void {
    jobOf(job).ask.warm = true;
}

pub fn releaseJob(context: *anyopaque, job: *sched.Job) void {
    const self: *Hall = @ptrCast(@alignCast(context));
    const held = jobOf(job);
    self.gpa.free(held.ask.mask);
    self.gpa.free(held.ask.starts);
    self.gpa.free(held.ask.prompt);
    self.gpa.destroy(held);
}

fn admitJob(context: *anyopaque, job: *sched.Job, sent: []const i32, said: *request_file.Diagnostic) sched.Admission {
    const self: *Hall = @ptrCast(@alignCast(context));
    const held = jobOf(job);
    if (held.ask.max_tokens <= sent.len) {
        said.set("the reply already reached max_tokens", .{});
        return .{ .failed = error.BadRequest };
    }
    const taken = self.core.admit(held.ask, sent) catch |err| switch (err) {
        error.Never => {
            said.set("This server's maximum context length is {d} tokens; the prompt is {d}", .{ self.options.session.capacity, held.ask.prompt.len + sent.len });
            return .never;
        },
        // a wait on another prompt's fill keeps its record in the job (`Core.wait_offer`); a short pool
        // leaves it null, which holds no fill and waits on the queue as it did
        error.Wait => {
            if (self.core.wait_offer) |waiting| held.ask.wait = waiting else held.ask.wait = .{};
            return .wait;
        },
        // dropped, not queued: the next turn runs whole without it all the same
        error.Declined => {
            said.set("a warm pass is run only while nothing else is live", .{});
            return .{ .failed = error.Failed };
        },
        error.OutOfMemory => {
            said.set("out of memory", .{});
            return .{ .failed = error.OutOfMemory };
        },
        else => {
            said.set("admitting the request failed: {t}", .{err});
            return .{ .failed = error.Failed };
        },
    };
    return .{ .admitted = .{ .slot = taken.slot, .cached_tokens = taken.cached } };
}

fn roundJobs(context: *anyopaque, round: *sched.Round) sched.Outcome {
    const self: *Hall = @ptrCast(@alignCast(context));
    return self.core.round(.{
        .ctx = round,
        .emit = emitRound,
        .end = endRound,
        .fail = failRound,
        .held = heldRound,
        .tail = tailRound,
        .cancelled = cancelledRound,
        .waiting = waitingRound,
        .rows = rowsRound,
    });
}

fn emitRound(ctx: *anyopaque, slot: usize, tokens: []const i32) void {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    round.emit(slot, tokens);
}

fn rowsRound(ctx: *anyopaque, slot: usize, rows: []const request_file.LogprobRow) void {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    round.rows(slot, rows);
}

fn endRound(ctx: *anyopaque, slot: usize, finish: request_file.Finish) void {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    round.end(slot, finish);
}

fn failRound(ctx: *anyopaque, slot: usize, message: []const u8) void {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    round.fail(slot, error.Failed, message);
}

fn heldRound(ctx: *anyopaque, slot: usize, cached: usize) void {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    round.held(slot, cached);
}

fn tailRound(ctx: *anyopaque, slot: usize, said: sched.Tail) void {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    round.tail(slot, said);
}

fn cancelledRound(ctx: *anyopaque, slot: usize) bool {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    return round.cancelled(slot);
}

fn waitingRound(ctx: *anyopaque) bool {
    const round: *sched.Round = @ptrCast(@alignCast(ctx));
    return round.waiting();
}

fn retireJob(context: *anyopaque, slot: usize, why: sched.Leave, kept: usize) sched.Tail {
    const self: *Hall = @ptrCast(@alignCast(context));
    return self.core.retire(slot, why, kept);
}

fn preemptJob(context: *anyopaque, slot: usize) void {
    const self: *Hall = @ptrCast(@alignCast(context));
    self.core.preempt(slot);
}

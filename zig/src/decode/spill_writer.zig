//! The thread that writes the states that leave (`serve --spill-gib`, spec 074 WS-06), so that the decoding
//! thread pays a state's copy to host memory and not its file: 12 ms and not 0.28 s for 590 MiB on GB10. Host
//! only: a job is a state's image whole (`learned.zig`'s header, tokens, table and parts), and what the parts
//! are is the model's. The thread writes the image's files (`learned.writeFiles`): each block not on disk,
//! then the state's own file; its memory is the image whole, so a read of a listed job needs no file.
//!
//! Two threads meet here, under `mutex`:
//! - The offering thread, which is the decoding thread: `begin`, the bytes into the job's blocks, `commit`;
//!   every lookup (`longest`, `find`, `has`, `discard`); `drain` and `close`. It alone allocates with `gpa`,
//!   changes the store's index and takes a block for a job.
//! - The writer's own, which makes no device call and touches nothing of the store but its directory: a
//!   job's files, each a `.part` synced and renamed, and the job marked. The offering thread then takes the
//!   job off the list at its next call (`reap`): the file joins the store's index and the blocks are free.
//!
//! A job stays listed from `commit` until it is reaped, so a state is always in the list or in the index and
//! a prompt that asks for one not yet on disk reads the job's bytes from memory (`Job.read`): never a `.part`.
//!
//! What vLLM and SGLang do, and what is followed (vLLM `3c062d39`, `v1/kv_offload/`; SGLang `e1f43c9a`,
//! `HiCacheController`): both copy a block to host memory from the thread that steps the model, keep their
//! writer threads off the device, never make that thread wait for the disk, and under pressure leave a block
//! unwritten (it is prefilled when asked for again); a key in flight or on disk is not written twice, and a
//! failed write is logged and costs that block. Their queues have no bound of their own; the host memory
//! they stage in is the bound. Here it is `spill.queue_bytes`, or the store's cap where that is less (what is
//! listed is kept room for on disk, and a list above the cap would clear every file for states of which
//! some then go in turn): past it a state that leaves is not written, and only a stop waits for room. One
//! job is always taken, whatever its size.
//!
//! Blocks of `spill.block` bytes are kept once used, `queue_bytes` of them at most, and `spill.warm_bytes`
//! are touched by the writer's thread as it starts: a page touched for the first time costs the copy ten
//! times its own time, and would on the decoding thread.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const learned = @import("learned.zig");
const units = @import("../cuda/units.zig");
const learned_blocks = @import("learned_blocks.zig");
const spill = @import("spill.zig");
const Store = learned.Store;

const pages = std.heap.page_allocator;
const page: std.mem.Alignment = .fromByteUnits(std.heap.page_size_min);
/// The bytes one write of the writer's thread gives the file at most.
const stage_bytes = 8 << 20;

/// A state's file in host memory, from `commit` until the offering thread reaps it.
pub const Job = struct {
    writer: *Writer,
    key: u64,
    /// The tokens the state holds.
    len: u32,
    /// The image's bytes, which fill `blocks` in order; where they lie in its files, its prefix's bytes, and
    /// what it adds on disk (`learned.Plan`).
    bytes: u64,
    layout: learned.Layout,
    prefix: u64,
    disk: u64,
    blocks: [][]u8,
    state: State = .taking,
    /// `discard` met it: its file is not kept.
    dropped: bool = false,
    /// Its bytes reserved above the disk's floor (`Writer.floor`), given back when it is reaped.
    reserved: u64 = 0,
    /// Images open on it: it is not reaped under them.
    readers: u32 = 0,
    since: Io.Timestamp,

    const State = enum { taking, queued, writing, done, failed };

    fn live(self: *const Job) bool {
        return !self.dropped and self.state != .failed and self.state != .taking;
    }

    /// The image's bytes from `offset`, in its own memory: `dest.len` at most, none at its end.
    fn view(context: *const anyopaque, offset: u64, dest: []u8) anyerror![]const u8 {
        const self: *const Job = @ptrCast(@alignCast(context));
        if (offset >= self.bytes) return &.{};
        const size = self.writer.block;
        const within: usize = @intCast(offset % size);
        const n: usize = @intCast(@min(dest.len, size - within, self.bytes - offset));
        return self.blocks[@intCast(offset / size)][within..][0..n];
    }

    /// The image's bytes from `offset` into `dest`; fewer than `dest.len` at its end.
    pub fn read(self: *const Job, dest: []u8, offset: u64) usize {
        const size = self.writer.block;
        var done: usize = 0;
        while (done < dest.len and offset + done < self.bytes) {
            const at = offset + done;
            const within: usize = @intCast(at % size);
            const n = @min(dest.len - done, size - within, self.bytes - at);
            @memcpy(dest[done..][0..n], self.blocks[@intCast(at / size)][within..][0..n]);
            done += n;
        }
        return done;
    }

    /// An image over the job is closed.
    pub fn release(self: *Job) void {
        const writer = self.writer;
        writer.mutex.lockUncancelable(writer.io);
        defer writer.mutex.unlock(writer.io);
        self.readers -= 1;
    }
};

/// A job being filled, by the offering thread: `span` or `copy` for every byte of the file, then `commit`;
/// `cancel` where a copy failed.
pub const Taking = struct {
    job: *Job,
    at: u64 = 0,

    /// The next bytes of the file to write into, `most` at most and one at least: a device part is copied
    /// span by span.
    pub fn span(self: *Taking, most: usize) []u8 {
        const size = self.job.writer.block;
        const within: usize = @intCast(self.at % size);
        const n = @min(most, size - within);
        std.debug.assert(self.at + n <= self.job.bytes);
        defer self.at += n;
        return self.job.blocks[@intCast(self.at / size)][within..][0..n];
    }

    pub fn copy(self: *Taking, bytes: []const u8) void {
        var done: usize = 0;
        while (done < bytes.len) {
            const dest = self.span(bytes.len - done);
            @memcpy(dest, bytes[done..][0..dest.len]);
            done += dest.len;
        }
    }

    /// The job is the writer's, and counts as written (`Store.written`).
    pub fn commit(self: *Taking) void {
        const job = self.job;
        const writer = job.writer;
        std.debug.assert(self.at == job.bytes);
        writer.mutex.lockUncancelable(writer.io);
        job.state = .queued;
        writer.mutex.unlock(writer.io);
        writer.store.written += 1;
        writer.work.signal(writer.io);
    }

    pub fn cancel(self: *Taking) void {
        const writer = self.job.writer;
        // nothing was written: the bytes only, not a failure that holds the next back
        if (writer.floor) |floor| floor.release(self.job.reserved);
        self.job.reserved = 0;
        writer.mutex.lockUncancelable(writer.io);
        self.job.state = .failed;
        writer.mutex.unlock(writer.io);
        writer.reap();
    }
};

pub const Writer = struct {
    gpa: Allocator,
    io: Io,
    store: *Store,
    /// Bytes of a block, and of the blocks listed jobs may hold.
    block: usize,
    bound: u64,
    /// Blocks kept once used.
    keep: usize = 0,
    mutex: Io.Mutex = .init,
    /// For the writer's thread: a job queued, the hold lifted, the store closing.
    work: Io.Condition = .init,
    /// For the offering thread: a job ended.
    ended: Io.Condition = .init,
    thread: Io.Future(void) = undefined,
    /// In the order committed. The offering thread's to change, under the mutex.
    jobs: std.ArrayList(*Job) = .empty,
    /// Blocks no job holds: touched ones, then the ones the thread has still to touch.
    free: std.ArrayList([]u8) = .empty,
    cold: std.ArrayList([]u8) = .empty,
    /// Blocks there are: in jobs, free, cold, and the one being touched.
    blocks: usize = 0,
    /// The writer's thread's bound on one write's bytes.
    stage: []u8 = &.{},
    closing: bool = false,
    /// A test's: no job is begun while it is set (`hold`).
    held: bool = false,
    /// What the disk keeps free (`spill.Floor`), the caller's, which outlives the writer; null: no floor.
    floor: ?*spill.Floor = null,
    /// States not taken for want of room, files that failed, files on disk.
    dropped: u64 = 0,
    failed: u64 = 0,
    done: u64 = 0,

    /// Starts the store's writer where it has none. Without a thread to give it the states are written by
    /// the thread that offers them, as before the writer.
    pub fn start(store: *Store) void {
        if (store.writer != null) return;
        store.writer = open(store) catch |err| {
            std.log.warn("--spill-gib: no thread writes the states ({t}), so the decoding thread does", .{err});
            return;
        };
    }

    fn open(store: *Store) !*Writer {
        const gpa = store.gpa;
        const self = try gpa.create(Writer);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = store.io, .store = store, .block = spill.block, .bound = @max(@min(spill.queue_bytes, store.cap), spill.block) };
        errdefer self.release();
        self.keep = @intCast(self.bound / self.block);
        // a job's bytes are written from its own memory: the stage bounds a write and is never touched
        const stage = pages.rawAlloc(stage_bytes, page, @returnAddress()) orelse return error.OutOfMemory;
        self.stage = stage[0..stage_bytes];
        const warm: usize = @min(self.keep, spill.warm_bytes / self.block);
        // the writer's thread moves a block from `cold` to `free` without allocating
        try self.free.ensureTotalCapacity(gpa, self.keep + 1);
        try self.cold.ensureTotalCapacity(gpa, warm);
        for (0..warm) |_| self.cold.appendAssumeCapacity(self.map() orelse break);
        self.thread = try self.io.concurrent(run, .{self});
        store.blocks.writers[@backingInt(store.kind)] = self;
        return self;
    }

    fn map(self: *Writer) ?[]u8 {
        // not `alloc`: a debug build would fill the block, on the decoding thread
        const ptr = pages.rawAlloc(self.block, page, @returnAddress()) orelse return null;
        self.blocks += 1;
        return ptr[0..self.block];
    }

    fn unmap(self: *Writer, block: []u8) void {
        pages.rawFree(block, page, @returnAddress());
        self.blocks -= 1;
    }

    /// Everything the writer holds but itself; no thread runs.
    fn release(self: *Writer) void {
        for (self.jobs.items) |job| self.retire(job);
        for (self.free.items) |block| self.unmap(block);
        for (self.cold.items) |block| self.unmap(block);
        std.debug.assert(self.blocks == 0);
        if (self.stage.len > 0) pages.rawFree(self.stage, page, @returnAddress());
        self.jobs.deinit(self.gpa);
        self.free.deinit(self.gpa);
        self.cold.deinit(self.gpa);
    }

    /// Stops the thread at its next block, and frees everything: what is still queued is not written
    /// (`drain` first).
    pub fn close(self: *Writer) void {
        const io = self.io;
        self.mutex.lockUncancelable(io);
        self.closing = true;
        var left: usize = 0;
        for (self.jobs.items) |job| left += @intFromBool(job.state == .queued or job.state == .writing);
        self.mutex.unlock(io);
        self.work.signal(io);
        self.thread.await(io);
        self.store.blocks.writers[@backingInt(self.store.kind)] = null;
        if (left > 0) std.log.warn("{d} states that left were still to write and are not kept", .{left});
        for (self.jobs.items) |job| {
            std.debug.assert(job.readers == 0);
            const name = learned.fileOf(self.store.kind, job.key, false);
            if (job.state == .done and job.dropped) self.store.dir.deleteFile(io, name.get()) catch {};
        }
        self.release();
        self.gpa.destroy(self);
    }

    /// A test's: while `on`, the thread begins no job.
    pub fn hold(self: *Writer, on: bool) void {
        self.mutex.lockUncancelable(self.io);
        self.held = on;
        self.mutex.unlock(self.io);
        self.work.signal(self.io);
    }

    // -- the offering thread ---------------------------------------------------------------------------------

    fn retire(self: *Writer, job: *Job) void {
        for (job.blocks) |block| self.unmap(block);
        self.gpa.free(job.blocks);
        job.layout.deinit(self.gpa);
        self.gpa.destroy(job);
    }

    /// Takes the jobs that ended off the list: a file written joins the store's index, and the cap is applied.
    pub fn reap(self: *Writer) void {
        const io = self.io;
        const store = self.store;
        var wrote = false;
        self.mutex.lockUncancelable(io);
        var kept: usize = 0;
        for (self.jobs.items) |job| {
            if ((job.state != .done and job.state != .failed) or job.readers > 0) {
                self.jobs.items[kept] = job;
                kept += 1;
                continue;
            }
            const file = learned.fileOf(store.kind, job.key, false);
            const name = file.get();
            if (job.reserved > 0) if (self.floor) |floor| floor.finish(store.blocks.dir.handle, job.reserved, job.state == .done);
            if (job.state == .done) {
                if (job.dropped) {
                    store.dir.deleteFile(io, name) catch {};
                } else if (store.dir.statFile(io, name, .{})) |_| {
                    const own = job.prefix + job.layout.own_bytes;
                    const stems = learned.stemsOf(self.gpa, &job.layout) catch null;
                    // a state not indexed is not kept: its blocks would be collected under it
                    if (stems == null) store.dir.deleteFile(io, name) catch {} else store.index(job.key, job.len, own, stems.?) catch store.dir.deleteFile(io, name) catch {};
                } else |_| {}
                // its blocks, or a dropped one's, are held to the cap and collected
                wrote = true;
            }
            // a block is kept while there are no more than the bound's
            for (job.blocks) |block| {
                if (self.blocks <= self.keep) self.free.appendAssumeCapacity(block) else self.unmap(block);
            }
            self.gpa.free(job.blocks);
            job.layout.deinit(self.gpa);
            self.gpa.destroy(job);
        }
        self.jobs.items.len = kept;
        self.mutex.unlock(io);
        if (wrote) store.clear(self.pending()) catch |err| std.log.warn("the states that left are not held to their cap: {t}", .{err});
    }

    /// Bytes on disk of the files still to write.
    fn pending(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var total: u64 = 0;
        for (self.jobs.items) |job| {
            if (job.state == .queued or job.state == .writing) total += job.disk;
        }
        return total;
    }

    /// Whether a listed job's state names the block: it is not collected under it.
    pub fn wants(self: *Writer, block: learned_blocks.Stem) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.jobs.items) |job| for (job.layout.names) |name| {
            if (std.mem.eql(u8, name[0..16], &block)) return true;
        };
        return false;
    }

    fn listed(self: *Writer, key: u64) ?*Job {
        for (self.jobs.items) |job| {
            if (job.key == key and job.live()) return job;
        }
        return null;
    }

    /// Starts the job for `tokens`, of `parts`. Null: the state is listed or on disk already, the store's
    /// cap is under what it adds on disk, or the listed jobs leave its image no room; with `until`, a
    /// stop's, room is waited for till then.
    pub fn begin(self: *Writer, tokens: []const i32, parts: []const learned.Part, until: ?Io.Clock.Timestamp) !?Taking {
        const io = self.io;
        const gpa = self.gpa;
        const store = self.store;
        self.reap();
        const key = learned.keyOf(tokens);
        if (self.has(key) or store.onDisk(key)) return null;
        var planned = try store.plan(gpa, tokens, parts);
        var planned_kept = false;
        defer if (!planned_kept) planned.deinit(gpa);
        if (!store.fits(planned.disk)) return null;
        const total = planned.image;
        const count: usize = @intCast((total + self.block - 1) / self.block);
        while (true) {
            self.mutex.lockUncancelable(io);
            var listed_blocks: usize = 0;
            for (self.jobs.items) |job| listed_blocks += job.blocks.len;
            if (self.jobs.items.len == 0 or (listed_blocks + count) * self.block <= self.bound) {
                self.mutex.unlock(io);
                break;
            }
            const deadline = until orelse {
                self.dropped += 1;
                self.mutex.unlock(io);
                std.log.warn("a state that leaves at {d} is not written: {d} MiB are still to write, of {d} at most", .{ tokens.len - 1, (listed_blocks * self.block) >> 20, self.bound >> 20 });
                return null;
            };
            const timed_out = if (self.ended.waitTimeout(io, &self.mutex, .{ .deadline = deadline })) |_| deadline.durationFromNow(io).raw.nanoseconds <= 0 else |_| true;
            if (timed_out) {
                self.mutex.unlock(io);
                std.log.warn("a state that leaves at {d} is not written: the stop's {d} s ran out with {d} MiB still to write, of {d} at most", .{ tokens.len - 1, spill.flush_seconds, (listed_blocks * self.block) >> 20, self.bound >> 20 });
                return null;
            }
            self.mutex.unlock(io);
            self.reap();
        }
        if (!spill.admitted(self.floor, store.blocks.dir.handle, planned.disk, tokens.len)) return null;
        errdefer if (self.floor) |floor| floor.release(planned.disk);
        const job = try gpa.create(Job);
        errdefer gpa.destroy(job);
        job.* = .{ .writer = self, .key = key, .len = @intCast(tokens.len), .bytes = total, .layout = planned.layout, .prefix = planned.prefix, .disk = planned.disk, .blocks = try gpa.alloc([]u8, count), .since = .now(io, .awake), .reserved = if (self.floor != null) planned.disk else 0 };
        errdefer gpa.free(job.blocks);
        self.mutex.lockUncancelable(io);
        // under the mutex: the writer's thread reads the list
        self.jobs.ensureUnusedCapacity(gpa, 1) catch |err| {
            self.mutex.unlock(io);
            return err;
        };
        var got: usize = 0;
        var touched: usize = 0;
        while (got < count) : (got += 1) {
            job.blocks[got] = self.free.pop() orelse {
                touched = got;
                break;
            };
        } else touched = count;
        while (got < count) : (got += 1) {
            job.blocks[got] = self.cold.pop() orelse self.map() orelse break;
        }
        if (got < count) {
            for (job.blocks[0..got]) |block| self.unmap(block);
            self.mutex.unlock(io);
            return error.OutOfMemory;
        }
        self.jobs.appendAssumeCapacity(job);
        planned_kept = true;
        self.mutex.unlock(io);
        // a block no thread has touched is touched here: a device copy into its pages faults each of them
        // at ten times this cost (0.53 s against 0.05 s for the six blocks a second state at once took)
        for (job.blocks[touched..]) |block| @memset(block, 0);
        // the files that go for this one go now: when it is on disk the store is within its cap
        store.clear(self.pending() + job.disk) catch |err| std.log.warn("the states that left are not held to their cap: {t}", .{err});
        var taking: Taking = .{ .job = job };
        const header: learned.Header = .{ .magic = learned.magic, .identity = store.identity, .tokens = @intCast(tokens.len), .parts = @intCast(parts.len) };
        taking.copy(std.mem.asBytes(&header));
        taking.copy(std.mem.sliceAsBytes(tokens));
        taking.copy(std.mem.sliceAsBytes(parts));
        return taking;
    }

    pub fn has(self: *Writer, key: u64) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.listed(key) != null;
    }

    /// `Store.longest` over the listed jobs: the longest above `best`.
    pub fn longest(self: *Writer, prompt: []const i32, under: usize, best: ?usize) ?usize {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var most = best;
        for (self.jobs.items) |job| {
            if (!job.live() or job.len >= under or job.len > prompt.len or (most != null and job.len <= most.?)) continue;
            if (learned.keyOf(prompt[0..job.len]) == job.key) most = job.len;
        }
        return most;
    }

    /// The listed job for `key`, held for an image until `Job.release`.
    pub fn find(self: *Writer, key: u64) ?*Job {
        self.reap();
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const job = self.listed(key) orelse return null;
        job.readers += 1;
        return job;
    }

    /// The listed job for `key` is not kept: not written where the thread has not begun it, its file deleted
    /// where it has.
    pub fn discard(self: *Writer, key: u64) void {
        self.mutex.lockUncancelable(self.io);
        if (self.listed(key)) |job| job.dropped = true;
        self.mutex.unlock(self.io);
        self.reap();
    }

    /// Waits until every listed job has ended, `until` at most, and reaps them. Whether all had.
    pub fn drain(self: *Writer, until: Io.Clock.Timestamp) bool {
        const io = self.io;
        self.mutex.lockUncancelable(io);
        const all = while (true) {
            var left = false;
            for (self.jobs.items) |job| left = left or job.state == .queued or job.state == .writing;
            if (!left) break true;
            const timed_out = if (self.ended.waitTimeout(io, &self.mutex, .{ .deadline = until })) |_| until.durationFromNow(io).raw.nanoseconds <= 0 else |_| true;
            if (timed_out) break false;
        };
        self.mutex.unlock(io);
        self.reap();
        return all;
    }

    // -- the writer's thread ---------------------------------------------------------------------------------

    fn run(self: *Writer) void {
        const io = self.io;
        self.mutex.lockUncancelable(io);
        while (!self.closing) {
            const next: ?*Job = if (self.held) null else for (self.jobs.items) |job| {
                if (job.state == .queued) break job;
            } else null;
            const job = next orelse {
                if (self.cold.pop()) |block| {
                    self.mutex.unlock(io);
                    @memset(block, 0);
                    self.mutex.lockUncancelable(io);
                    self.free.appendAssumeCapacity(block);
                    continue;
                }
                self.work.waitUncancelable(io, &self.mutex);
                continue;
            };
            if (job.dropped) {
                job.state = .failed;
                self.ended.broadcast(io);
                continue;
            }
            job.state = .writing;
            self.mutex.unlock(io);
            const began: Io.Timestamp = .now(io, .awake);
            const result = self.write(job);
            if (result) |bytes| {
                std.log.info("a state that left at {d} is on disk: {d} MiB written of {d} in {d:.3} s, {d:.3} s after it left", .{ job.len - 1, bytes >> 20, job.bytes >> 20, units.secondsSince(io, began), units.secondsSince(io, job.since) });
            } else |err| {
                if (err != error.Closing) std.log.warn("a state that left at {d} is not written: {t}", .{ job.len - 1, err });
            }
            self.mutex.lockUncancelable(io);
            if (result) |_| {
                job.state = .done;
                self.done += 1;
            } else |err| {
                job.state = .failed;
                self.failed += @intFromBool(err != error.Closing);
            }
            self.ended.broadcast(io);
        }
        self.mutex.unlock(io);
    }

    fn stopping(self: *Writer) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.closing;
    }

    /// The job's files, each synced and renamed; no part of one is left where that failed. Its bytes.
    fn write(self: *Writer, job: *const Job) !u64 {
        const source: learned.Source = .{ .context = job, .view = Job.view };
        return learned.writeFiles(self.io, self.store.blocks.dir, self.store.dir, self.store.kind, self.store.identity, job.key, &job.layout, job.prefix, source, self.stage, .{ .context = self, .call = stopped });
    }

    fn stopped(context: *anyopaque) bool {
        const self: *Writer = @ptrCast(@alignCast(context));
        return self.stopping();
    }
};

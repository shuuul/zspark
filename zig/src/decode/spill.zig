//! Which kept states go to disk when they leave a pool or a session (`serve --spill-gib`, spec 074), and
//! which file a prompt starts from. Host only and no model's: the files are `learned.zig`'s, a state's bytes
//! the model's (`models/qwen4/session/learnt.zig`, `pool/hall_spill.zig`).
//!
//! The rule, and where it comes from. vLLM (`v1/kv_offload/`, its `fs` tier) and SGLang (`HiCacheController`,
//! `HiCacheFile`) both write a block or a page as it is made, in the background, check a byte count and no
//! checksum on a read, prefill what a read did not give, drop the least recently used, and read a previous
//! process's files. All of that is followed but the time of the write: their unit is 16 tokens or a page,
//! written once, and a file here is a whole state (a recurrent state of 110 MiB on Flash Next and every
//! cache row below it, 30.8 kB a token), so writing as a reply ends would write a conversation's rows again
//! at each turn. A state is therefore written when it leaves (the recipe's rule, SGLang's `write_back`), and
//! what is still held when the server stops is written then, within `flush_seconds`.
//!
//! Of a sequence's states one is written: the highest of its prompt's (`pick`), which is where the
//! conversation's next turn starts; a reply's own states stand in tokens the next prompt does not repeat.

const std = @import("std");
const learned = @import("learned.zig");

/// A state below this position is not written: its prompt is run again as soon as a file is read. A test
/// lowers it.
pub var floor: usize = 2048;
/// A file is read where it stands this many tokens above where the prompt starts without it.
pub var worth: usize = 512;
/// Seconds a stop may write for; what is left then is not written.
pub const flush_seconds = 60;
/// What `spill_writer.zig` holds in host memory: the bytes of a block, of the states still to write at most
/// (past them a state that leaves is not written), and of the blocks its thread touches as it starts.
pub var block: usize = 32 << 20;
pub var queue_bytes: u64 = 4 << 30;
pub var warm_bytes: usize = 1 << 30;

/// A state of a sequence that leaves.
pub const Candidate = struct {
    pos: usize,
    /// It stands in the sequence's prompt, not in its reply.
    prompt: bool,
};

/// Which of `held`, the states that leave with a sequence or after which none of its prompt's is left, is
/// written: the highest of its prompt's at `floor` or above. Null: none.
pub fn pick(held: []const Candidate) ?usize {
    var best: ?usize = null;
    for (held, 0..) |one, i| {
        if (!one.prompt or one.pos < floor) continue;
        if (best == null or one.pos > held[best.?].pos) best = i;
    }
    return best;
}

/// Where `prompt` starts from a file of `store`: the position of the longest state it holds that the prompt
/// goes on from (a file's tokens are the sequence through the token after its state, and one token more is
/// the prompt's own to run), shorter than `under` tokens and `worth` above `start`, where the prompt starts
/// without it. Null: none.
pub fn serving(store: *const learned.Store, prompt: []const i32, start: usize, under: usize) ?usize {
    const len = store.longest(prompt, @min(under, prompt.len)) orelse return null;
    if (len < 2 or len - 1 < start + worth) return null;
    return len - 1;
}

/// Whether a stop that began writing at `began` may write another file.
pub fn flushing(io: std.Io, began: std.Io.Timestamp) bool {
    return began.untilNow(io, .awake).toNanoseconds() < flush_seconds * std.time.ns_per_s;
}

/// What the disk keeps free under the states' files (`--learn-min-free-gib`, 4 GiB unless given), TensorFold
/// 1.0.12's `lanes/learned_disk.zig` `Admission` as it is: a write reserves its bytes above the floor and the
/// bytes reserved before it, or is not made; a write refused or failed holds back the next ones 1, 2, 4 ...
/// 60 s, unless the disk has freed space since. One floor serves `--learn`'s files and `--spill-gib`'s, as
/// TensorFold has one store. vLLM's file tier and SGLang's `HiCacheFile` keep no floor: a write that fails is
/// logged and costs that block. Asked on the thread that offers the states (`spill_writer.zig`'s header).
pub const Floor = struct {
    floor: u64 = 4 << 30,
    reserved: u64 = 0,
    failures: u8 = 0,
    retry_at: u64 = 0,
    failed_free: ?u64 = null,
    /// Bytes free to an unprivileged writer on the file system of a directory, null: not known.
    free: *const fn (std.posix.fd_t) ?u64 = available,
    /// Seconds of a monotonic clock, null: not known.
    clock: *const fn () ?u64 = now,

    pub const Result = enum { ready, refused, quiet };

    /// Reserves `bytes` on `dir`'s disk: `.ready`, or `.refused` (below the floor, the free space or the
    /// clock unknown), or `.quiet` while a refusal's wait lasts.
    pub fn reserve(self: *Floor, dir: std.posix.fd_t, bytes: u64) Result {
        const space = self.free(dir);
        const changed = if (space) |n| if (self.failed_free) |old| n > old else false else false;
        const tick = self.clock() orelse return self.block(space);
        if (tick < self.retry_at and !changed) return .quiet;
        const need = std.math.add(u64, self.floor, self.reserved) catch return self.block(space);
        const full = std.math.add(u64, need, bytes) catch return self.block(space);
        if (space == null or space.? < full) return self.block(space);
        self.reserved += bytes;
        return .ready;
    }

    /// A reserved write ended: written, or failed, which holds the next ones back.
    pub fn finish(self: *Floor, dir: std.posix.fd_t, bytes: u64, success: bool) void {
        self.release(bytes);
        if (success) {
            self.failures = 0;
            self.retry_at = 0;
            self.failed_free = null;
        } else _ = self.block(self.free(dir));
    }

    /// A reserved write that was not begun: its bytes only.
    pub fn release(self: *Floor, bytes: u64) void {
        std.debug.assert(self.reserved >= bytes);
        self.reserved -= bytes;
    }

    fn block(self: *Floor, space: ?u64) Result {
        self.failed_free = space;
        self.failures = @min(self.failures +| 1, 7);
        const wait = @min(@as(u64, 1) << @intCast(self.failures - 1), 60);
        self.retry_at = (self.clock() orelse 0) +| wait;
        return .refused;
    }
};

/// Whether a file of `bytes` for a state at `len - 1` may be written above `kept` (null: no floor), its bytes
/// then reserved: a refusal is said once a wait, and not while it lasts.
pub fn admitted(kept: ?*Floor, dir: std.posix.fd_t, bytes: u64, len: usize) bool {
    const held = kept orelse return true;
    return switch (held.reserve(dir, bytes)) {
        .ready => true,
        .quiet => false,
        .refused => {
            const free = if (held.failed_free) |n| n >> 20 else 0;
            std.log.warn("a state at {d} is not written: {d} MiB free on its disk, {d} MiB to be kept free (--learn-min-free-gib) and {d} reserved", .{ len -| 1, free, held.floor >> 20, held.reserved >> 20 });
            return false;
        },
    };
}

const builtin = @import("builtin");
const Count = if (builtin.os.tag == .linux) u64 else u32;
/// glibc's `struct statvfs` (TensorFold's declaration of it).
const Stat = extern struct {
    bsize: u64,
    frsize: u64,
    blocks: Count,
    bfree: Count,
    bavail: Count,
    files: Count,
    ffree: Count,
    favail: Count,
    fsid: u64,
    flag: u64,
    namemax: u64,
    tail: [if (builtin.os.tag == .linux) 24 else 0]u8,
};
extern "c" fn fstatvfs(fd: c_int, result: *Stat) c_int;

pub fn available(dir: std.posix.fd_t) ?u64 {
    var stat: Stat = undefined;
    if (fstatvfs(dir, &stat) != 0) return null;
    return std.math.mul(u64, stat.frsize, stat.bavail) catch null;
}

fn now() ?u64 {
    var t: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t) != 0 or t.sec < 0) return null;
    return @intCast(t.sec);
}

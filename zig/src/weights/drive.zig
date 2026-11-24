//! The drive under a checkpoint's mapping, kept awake while rows are read from it. The Spark's NVMe drive
//! (Samsung MZALC4T0HBL1, both machines) enters a deep idle state after 100 to 120 ms without I/O, and the
//! next read then waits 10 ms (spec 089: 9.9 to 10.0 ms after any gap from 120 ms to 5 s, 0.2 to 1 ms
//! before it). A reply's rounds read a few table rows each, and three rounds running whose rows the page
//! cache held leave the drive that long alone: one round in 6 to 18 on new text then stalled the device 7 to
//! 10 ms. A thread here reads one block past the page cache every `period` while rows are being asked for,
//! and nothing once they are not: a server with no request issues no I/O. What it reads goes nowhere, so no
//! row's bits can depend on it. Host only; the machine's own setting (the kernel's APST latency) is the
//! owner's and is not touched. A departure: TensorFold has no such reader.

const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;

pub const Drive = struct {
    /// A file on the drive opened with O_DIRECT, its owner's: it must stay open until `close`.
    fd: linux.fd_t,
    /// Whole blocks of the file: a read is one of them.
    blocks: u64,
    period_ns: u64,
    linger_ns: u64,
    /// Counts every `touch`. Any thread adds to it; the reader only compares it.
    touched: std.atomic.Value(u32) = .init(0),
    /// The reader has seen no touch for `linger_ns` and is about to sleep, or sleeps, until one rings.
    asleep: std.atomic.Value(bool) = .init(false),
    /// What the reader sleeps on: rung by a touch that finds it asleep, and by `close`.
    bell: std.atomic.Value(u32) = .init(0),
    closing: std.atomic.Value(bool) = .init(false),
    /// Blocks read so far.
    reads: std.atomic.Value(u64) = .init(0),
    thread: std.Thread = undefined,

    /// What a read asks for: the logical block of the drive's filesystem as O_DIRECT wants it aligned, and
    /// the least that is I/O.
    pub const block = 4096;

    pub const Options = struct {
        /// Between two reads. The drive's step is between 100 and 120 ms of idle: at 50 ms a read that comes
        /// a whole period late (a thread not scheduled under a load, a read behind a queue) still lands
        /// before 100 ms. 20 reads of 4 KiB a second are nothing to the drive: a reader every 5 ms was
        /// measured, and 50 keeps it out of the deep state the same (spec 089, WS-04).
        period_ns: u64 = 50 * std.time.ns_per_ms,
        /// How long after the last touch the reads go on: longer than a prompt's window runs (1.3 s for
        /// 2,048 rows of NVFP4, one gather a window), so the reader does not stop and start inside a
        /// prompt, and short enough that a request's end is the reads' end.
        linger_ns: u64 = 2 * std.time.ns_per_s,
    };

    /// A reader over `fd`, a file of `len` bytes opened with O_DIRECT; it sleeps until the first `touch`.
    /// Null, and nothing is kept awake, where there is no such descriptor (a filesystem without direct
    /// reads), the file holds no whole block, or no thread or memory is to be had: a round then waits as
    /// it did, which is no error.
    pub fn open(gpa: Allocator, fd: linux.fd_t, len: u64, options: Options) ?*Drive {
        if (fd < 0 or len < block) return null;
        const self = gpa.create(Drive) catch return null;
        self.* = .{ .fd = fd, .blocks = len / block, .period_ns = options.period_ns, .linger_ns = options.linger_ns };
        self.thread = std.Thread.spawn(.{}, work, .{self}) catch {
            gpa.destroy(self);
            return null;
        };
        return self;
    }

    /// Ends the reader. Nothing may `touch` after.
    pub fn close(self: *Drive, gpa: Allocator) void {
        self.closing.store(true, .seq_cst);
        self.ring();
        self.thread.join();
        gpa.destroy(self);
    }

    /// Rows are being read from the drive's files: it is kept awake from now until `linger_ns` after the
    /// last such call. From any thread; an add and a load, and a system call only where the reader slept.
    pub fn touch(self: *Drive) void {
        _ = self.touched.fetchAdd(1, .seq_cst);
        if (self.asleep.load(.seq_cst)) self.ring();
    }

    fn ring(self: *Drive) void {
        _ = self.bell.fetchAdd(1, .seq_cst);
        _ = linux.futex_3arg(&self.bell.raw, .{ .cmd = .WAKE, .private = true }, 1);
    }

    fn now() u64 {
        var time: linux.timespec = undefined;
        _ = linux.clock_gettime(.MONOTONIC, &time);
        return @as(u64, @intCast(time.sec)) * std.time.ns_per_s + @as(u64, @intCast(time.nsec));
    }

    fn work(self: *Drive) void {
        var scratch: [block]u8 align(block) = undefined;
        const period: linux.timespec = .{ .sec = @intCast(self.period_ns / std.time.ns_per_s), .nsec = @intCast(self.period_ns % std.time.ns_per_s) };
        var seen: u32 = 0;
        var last: u64 = 0;
        // no touch yet: asleep from the start
        var awake = false;
        var step: u64 = 0;
        while (true) {
            const rung = self.bell.load(.seq_cst);
            if (self.closing.load(.seq_cst)) return;
            const count = self.touched.load(.seq_cst);
            const time = now();
            if (count != seen) {
                seen = count;
                last = time;
                awake = true;
            }
            if (!awake or time - last >= self.linger_ns) {
                awake = false;
                // A touch from here on finds `asleep` and rings; one before it is seen by the count, read
                // after the flag is up: none is slept through.
                self.asleep.store(true, .seq_cst);
                if (self.touched.load(.seq_cst) == seen) _ = linux.futex_4arg(&self.bell.raw, .{ .cmd = .WAIT, .private = true }, rung, null);
                self.asleep.store(false, .seq_cst);
                continue;
            }
            // Another block each time, far from the last (a step of a prime number of blocks): none is
            // answered from what the drive or the kernel still holds of the read before.
            step +%= 1;
            const at = (step *% 7919) % self.blocks * block;
            const got: isize = @bitCast(linux.pread(self.fd, &scratch, block, @intCast(at)));
            // a descriptor that does not read (an error, not a short file) will not read next time either
            if (got >= 0) {
                _ = self.reads.fetchAdd(1, .monotonic);
            } else if (linux.errno(@as(usize, @bitCast(got))) != .INTR) return;
            _ = linux.futex_4arg(&self.bell.raw, .{ .cmd = .WAIT, .private = true }, rung, &period);
        }
    }
};

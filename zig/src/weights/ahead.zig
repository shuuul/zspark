//! A window's table rows gathered before its turn, by a thread of its own: asked/done generation words, a
//! bell futex, id-prefix handover (`take` / `give`), and the worker's lifetime. The source is a comptime
//! type with `gather(ids, out) Error!void` and `stride()`; Flash Next's PLE table is the first
//! (`models/qwen4/blocks/ple.zig`). N-gram ids, history and when to ask stay with that caller. Host only.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Gather-ahead over `Source`. `self` must not move once it has been asked: the thread holds its address.
pub fn Ahead(comptime Source: type) type {
    return struct {
        const Self = @This();

        /// [ids]: the table rows asked for; the thread reads those `asked` counts, which are not written
        /// again until it has gathered them.
        ids: []u64,
        /// The gathered rows, of length `ids.len * source.stride()`: `give` exchanges the two.
        host: []u16,
        /// An asking's number (high half) and how many of `ids` it asks for: one word, so the thread never
        /// reads a count of one asking with another's ids. Written by the asking thread alone.
        asked: std.atomic.Value(u64) = .init(0),
        /// The value of `asked` the thread has gathered up to. Written by the thread alone.
        done: std.atomic.Value(u64) = .init(0),
        /// A gather failed, a row out of range: the thread's to write, read once it is idle.
        failed: std.atomic.Value(bool) = .init(false),
        /// Counts every asking and every gather done; what either thread sleeps on.
        bell: std.atomic.Value(u32) = .init(0),
        closing: std.atomic.Value(bool) = .init(false),
        thread: ?std.Thread = null,
        source: Source = undefined,

        pub fn init(gpa: Allocator, ids: usize, stride: usize) Allocator.Error!Self {
            const id_buf = try gpa.alloc(u64, ids);
            errdefer gpa.free(id_buf);
            return .{ .ids = id_buf, .host = try gpa.alloc(u16, ids * stride) };
        }

        pub fn deinit(self: *Self, gpa: Allocator) void {
            if (self.thread) |thread| {
                self.closing.store(true, .release);
                self.ring();
                thread.join();
            }
            gpa.free(self.ids);
            gpa.free(self.host);
            self.* = undefined;
        }

        /// How many of `ids` are asked for.
        pub fn held(self: *const Self) usize {
            return @as(u32, @truncate(self.asked.raw));
        }

        /// Nothing is held: a new asking, of no rows, once the thread has ended the one before.
        pub fn drop(self: *Self) void {
            self.wait();
            self.failed.store(false, .release);
            self.asked.store(((self.asked.raw >> 32) +% 1) << 32, .release);
        }

        /// `source` is the thread's from here; what was held is gone.
        pub fn adopt(self: *Self, source: Source) void {
            self.drop();
            self.source = source;
        }

        /// Copy `ids` onto `self.ids[have..]` and publish that many as held. Spawns the worker if this is
        /// the first asking; a thread that cannot be started leaves the count as it was.
        pub fn ask(self: *Self, have: usize, ids: []const u64) void {
            const want = have + ids.len;
            if (self.thread == null) self.thread = std.Thread.spawn(.{}, work, .{self}) catch return;
            @memcpy(self.ids[have..want], ids);
            self.asked.store((self.asked.raw & ~@as(u64, std.math.maxInt(u32))) | want, .release);
            self.ring();
        }

        fn work(self: *Self) void {
            var gathered: u64 = 0;
            while (true) {
                const rung = self.bell.load(.acquire);
                if (self.closing.load(.acquire)) return;
                const asked = self.asked.load(.acquire);
                if (asked >> 32 != gathered >> 32) gathered = asked & ~@as(u64, std.math.maxInt(u32));
                if (gathered == asked) {
                    self.done.store(asked, .release);
                    _ = std.os.linux.futex_4arg(&self.bell.raw, .{ .cmd = .WAIT, .private = true }, rung, null);
                    continue;
                }
                const from: usize = @as(u32, @truncate(gathered));
                const to: usize = @as(u32, @truncate(asked));
                const stride = self.source.stride();
                self.source.gather(self.ids[from..to], self.host[from * stride .. to * stride]) catch self.failed.store(true, .release);
                gathered = asked;
                self.done.store(asked, .release);
                self.ring();
            }
        }

        fn ring(self: *Self) void {
            _ = self.bell.fetchAdd(1, .release);
            _ = std.os.linux.futex_3arg(&self.bell.raw, .{ .cmd = .WAKE, .private = true }, 2);
        }

        /// Until the thread has gathered all it was asked for.
        fn wait(self: *Self) void {
            if (self.thread == null or self.held() == 0) return;
            while (true) {
                const rung = self.bell.load(.acquire);
                if (self.done.load(.acquire) == self.asked.raw) return;
                _ = std.os.linux.futex_4arg(&self.bell.raw, .{ .cmd = .WAIT, .private = true }, rung, null);
            }
        }

        /// The rows held for the first of `ids`, `Source.gather`'s bytes for them: all of `ids` or, where
        /// fewer rows are held, as many, if the ids are the same in order; else none. They stay until the
        /// next `ask` that drops them.
        pub fn take(self: *Self, ids: []const u64) []const u16 {
            self.wait();
            const count = @min(self.held(), ids.len);
            if (count == 0 or self.failed.load(.acquire) or !std.mem.eql(u64, self.ids[0..count], ids[0..count])) return &.{};
            return self.host[0 .. count * self.source.stride()];
        }

        /// `take` without waiting: the rows the thread has already gathered for the first of `ids`, as many
        /// as are, and none where the ids differ or a gather failed. A row gathered is not written again
        /// before the next asking that drops it, so the rows may be read while the thread gathers the rest.
        pub fn ready(self: *Self, ids: []const u64) []const u16 {
            if (self.thread == null) return &.{};
            const done = self.done.load(.acquire);
            const asked = self.asked.raw;
            // `done` may still be an asking's before this one, whose rows are gone
            if (done >> 32 != asked >> 32) return &.{};
            const count = @min(@as(u32, @truncate(done)), ids.len);
            if (count == 0 or self.failed.load(.acquire) or !std.mem.eql(u64, self.ids[0..count], ids[0..count])) return &.{};
            return self.host[0 .. count * self.source.stride()];
        }

        /// `take` with no copy: `host` becomes the memory that holds the rows, and what it was is gathered
        /// into next. Returns how many of `ids` its rows are; 0, and nothing changed, where the rows held
        /// are not theirs. Rows handed over are held no more.
        pub fn give(self: *Self, ids: []const u64, host: *[]u16) usize {
            std.debug.assert(host.len == self.host.len);
            const rows = self.take(ids);
            if (rows.len == 0) return 0;
            std.mem.swap([]u16, host, &self.host);
            self.drop();
            return rows.len / self.source.stride();
        }
    };
}

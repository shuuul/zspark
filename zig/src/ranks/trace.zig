//! What a run of `zspark ranks-check` left, to hold another run to: the bytes of every buffer read, in the order
//! read, each with what it is, and the first place two runs part, in words. A model's check names what it reads
//! (`What`: an enum with `text`, and `gathered` for a buffer of every rank's rows, `[world, rows, ..]`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");

/// What a run left over `What`, in the order it was read.
pub fn Trace(comptime What: type) type {
    return struct {
        const Self = @This();

        gpa: Allocator,
        bytes: std.ArrayList(u8) = .empty,
        marks: std.ArrayList(Mark) = .empty,
        /// Where the run stands: what the next marks are of.
        step: usize = 0,
        rows: usize = 0,

        pub const Mark = struct {
            what: What,
            /// The window, or a drafter's step.
            step: usize,
            rows: usize,
            layer: ?usize,
            from: usize,
            len: usize,
            /// Bytes a value, values a row.
            unit: usize,
            wide: usize,
        };

        pub fn deinit(self: *Self) void {
            self.bytes.deinit(self.gpa);
            self.marks.deinit(self.gpa);
        }

        /// `view`'s values as they are on the device once all that was launched has run.
        pub fn take(self: *Self, what: What, layer: ?usize, view: cuda.View) !void {
            try cuda.synchronize();
            const len = view.count() * view.dtype.size();
            const from = self.bytes.items.len;
            try self.bytes.resize(self.gpa, from + len);
            try cuda.download(view, self.bytes.items[from..]);
            try self.marks.append(self.gpa, .{ .what = what, .step = self.step, .rows = self.rows, .layer = layer, .from = from, .len = len, .unit = view.dtype.size(), .wide = @intCast(view.size[@intCast(view.ndim - 1)]) });
        }

        /// Host values the run chose or merged: `bytes` of `unit` bytes a value, `wide` values a row.
        pub fn put(self: *Self, what: What, bytes: []const u8, unit: usize, wide: usize) !void {
            const from = self.bytes.items.len;
            try self.bytes.appendSlice(self.gpa, bytes);
            try self.marks.append(self.gpa, .{ .what = what, .step = self.step, .rows = self.rows, .layer = null, .from = from, .len = bytes.len, .unit = unit, .wide = wide });
        }

        pub fn of(self: *const Self, mark: Mark) []const u8 {
            return self.bytes.items[mark.from..][0..mark.len];
        }

        /// The marks `keep` keeps alone: what a run through graphs can read, of a run that read more.
        pub fn only(self: *const Self, gpa: Allocator, comptime keep: fn (Mark) bool) !Self {
            var out: Self = .{ .gpa = gpa };
            errdefer out.deinit();
            for (self.marks.items) |mark| {
                if (!keep(mark)) continue;
                var copy = mark;
                copy.from = out.bytes.items.len;
                try out.bytes.appendSlice(gpa, self.of(mark));
                try out.marks.append(gpa, copy);
            }
            return out;
        }

        /// A window's end: what a captured graph leaves to read (a layer's marks are the launched run's alone).
        pub fn atEnd(mark: Mark) bool {
            return mark.layer == null;
        }

        /// The first place `got` is not `want`, in words; null when every bit is the same.
        pub fn differs(arena: Allocator, want: *const Self, got: *const Self, want_name: []const u8, got_name: []const u8) !?[]const u8 {
            if (want.marks.items.len != got.marks.items.len)
                return try std.fmt.allocPrint(arena, "{s} read {d} buffers, {s} {d}", .{ want_name, want.marks.items.len, got_name, got.marks.items.len });
            for (want.marks.items, got.marks.items) |a, b| {
                const where = if (a.layer) |layer|
                    try std.fmt.allocPrint(arena, "step {d} ({d} rows), layer {d}, {s}", .{ a.step, a.rows, layer, a.what.text() })
                else
                    try std.fmt.allocPrint(arena, "step {d} ({d} rows), {s}", .{ a.step, a.rows, a.what.text() });
                if (a.what != b.what or a.step != b.step or a.layer != b.layer or a.len != b.len)
                    return try std.fmt.allocPrint(arena, "{s}: {s} read {d} bytes here, {s} {s} of step {d} in {d}", .{ where, want_name, a.len, got_name, b.what.text(), b.step, b.len });
                const x = want.of(a);
                const y = got.of(b);
                const at = std.mem.indexOfDiff(u8, x, y) orelse continue;
                const value = at / a.unit;
                var row = value / a.wide;
                const whose = if (a.what.gathered()) try std.fmt.allocPrint(arena, "rank {d}'s ", .{row / a.rows}) else "";
                if (a.what.gathered()) row %= a.rows;
                var different: usize = 0;
                for (0..x.len / a.unit) |v| different += @intFromBool(!std.mem.eql(u8, x[v * a.unit ..][0..a.unit], y[v * a.unit ..][0..a.unit]));
                return try std.fmt.allocPrint(arena, "{s}: {s}row {d}, column {d} is 0x{x} {s} and 0x{x} {s}; {d} of {d} values differ", .{
                    where,                                                             whose,          row,                                                               value % a.wide,
                    std.mem.readVarInt(u64, x[value * a.unit ..][0..a.unit], .little), want_name,      std.mem.readVarInt(u64, y[value * a.unit ..][0..a.unit], .little), got_name,
                    different,                                                         x.len / a.unit,
                });
            }
            return null;
        }

        /// One comparison's line, over NCCL against the loopback; the first difference found is kept.
        pub fn report(arena: Allocator, stdout: *std.Io.Writer, failure: *?[]const u8, comptime how: []const u8, want: *const Self, got: *const Self) !void {
            if (try differs(arena, want, got, "in loopback", "over NCCL")) |text| {
                try stdout.print("over NCCL, " ++ how ++ ": DIFFERS: {s}\n", .{text});
                if (failure.* == null) failure.* = try std.fmt.allocPrint(arena, how ++ ": {s}", .{text});
            } else try stdout.print("over NCCL, " ++ how ++ ": {d} buffers, {d} bytes, the loopback's bits\n", .{ got.marks.items.len, got.bytes.items.len });
            try stdout.flush();
        }
    };
}

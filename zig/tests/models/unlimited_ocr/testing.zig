//! What Unlimited-OCR-2's forward tests share (`forward_dash_test.zig`).

const std = @import("std");
const numeric = @import("../../../src/weights/numeric.zig");

/// Cosine (the worst row's and the whole's) and the largest absolute difference of bf16 rows against the
/// reference's, bf16 or f32.
pub const Match = struct {
    worst: f64 = 1,
    whole: f64 = 1,
    max_abs: f64 = 0,

    pub fn of(mine: []const u16, comptime T: type, want: []const T, width: usize) Match {
        var out: Match = .{};
        var dot_all: f64 = 0;
        var aa_all: f64 = 0;
        var bb_all: f64 = 0;
        for (0..mine.len / width) |row| {
            var dot: f64 = 0;
            var aa: f64 = 0;
            var bb: f64 = 0;
            for (mine[row * width ..][0..width], want[row * width ..][0..width]) |a_bits, b_raw| {
                const a: f64 = numeric.f32FromBf16(a_bits);
                const b: f64 = if (T == u16) numeric.f32FromBf16(b_raw) else b_raw;
                dot += a * b;
                aa += a * a;
                bb += b * b;
                out.max_abs = @max(out.max_abs, @abs(a - b));
            }
            out.worst = @min(out.worst, dot / @sqrt(aa * bb + 1e-300));
            dot_all += dot;
            aa_all += aa;
            bb_all += bb;
        }
        out.whole = dot_all / @sqrt(aa_all * bb_all + 1e-300);
        return out;
    }

    pub fn merge(self: *Match, other: Match) void {
        self.worst = @min(self.worst, other.worst);
        self.whole = @min(self.whole, other.whole);
        self.max_abs = @max(self.max_abs, other.max_abs);
    }
};

/// Levenshtein's distance between two token sequences.
pub fn editDistance(gpa: std.mem.Allocator, a: []const u32, b: []const u32) !usize {
    var row = try gpa.alloc(usize, b.len + 1);
    defer gpa.free(row);
    for (row, 0..) |*r, j| r.* = j;
    for (a, 1..) |x, i| {
        var diag = row[0];
        row[0] = i;
        for (b, 1..) |y, j| {
            const up = row[j];
            row[j] = @min(@min(row[j] + 1, row[j - 1] + 1), diag + @intFromBool(x != y));
            diag = up;
        }
    }
    return row[b.len];
}

//! A prompt's placeholder rows and what an encoder made for them (spec 019's image, spec 138's audio): the forward
//! writes the features over the rows' embeddings, in every stream (TensorFold 0.6.5's `image_rows.embed` and
//! `compute`'s `features`). Moved from `models/qwen4/forward_state.zig`'s `Image` when Qwen3-ASR's audio rows became
//! its second user (spec 138 WS-04). The copy is `kernels/common/embed.cu`'s `zs_embed_rows`: no arithmetic, so a
//! row's bits are its feature's in any window.

const std = @import("std");
const cuda = @import("../cuda/cuda.zig");

/// Lent: the caller's until the state that holds it lets it go.
pub const Placeholders = struct {
    /// bf16 [placeholders, hidden]: a placeholder's features a row, in the prompt's order.
    features: cuda.View,
    /// The prompt position of each feature row, ascending.
    rows: []const u32,

    /// Whether a window of `count` rows from position `pos` holds one of the rows.
    pub fn touches(self: Placeholders, pos: usize, count: usize) bool {
        for (self.rows) |row| if (row >= pos and row < pos + count) return true;
        return false;
    }

    /// The pairs (row of the window, feature row) of a window of `count` rows from position `pos`, into `out`.
    pub fn within(self: Placeholders, pos: usize, count: usize, out: [][2]i32) usize {
        var n: usize = 0;
        // the rows are ascending: the first at or past pos, then while inside the window
        var i = std.sort.lowerBound(u32, self.rows, @as(u32, @intCast(pos)), struct {
            fn order(want: u32, row: u32) std.math.Order {
                return std.math.order(want, row);
            }
        }.order);
        while (i < self.rows.len and self.rows[i] < pos + count) : (i += 1) {
            out[n] = .{ @intCast(self.rows[i] - pos), @intCast(i) };
            n += 1;
        }
        return n;
    }
};

/// Rows `pairs` int32 [K, 2] name of h bf16 [rows, streams * hidden] take features bf16 [N, hidden]'s rows, in
/// every stream.
pub fn embed(features: cuda.View, pairs: cuda.View, h: cuda.View, streams: usize) cuda.Error!void {
    return cuda.launched(cuda.kernels.zs_embed_rows(&features, &pairs, &h, @intCast(streams)));
}

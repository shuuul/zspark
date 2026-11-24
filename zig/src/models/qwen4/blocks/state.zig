//! The sliding windows a committed sequence keeps on the device.

const cuda = @import("../../../cuda/cuda.zig");

/// Where a layer's new rows sit in `fresh`: layer `l`, row `r` starts at `l * layer + r * row` elements, and its
/// first values are the window's channels.
pub const Fresh = struct {
    view: cuda.View,
    layer: usize,
    row: usize,
    rows: usize,
};

/// old bf16 [layers, taps, channels] in place: its rows become rows `keep` onward of [old; the new rows], so the
/// window ends at the last kept row.
pub fn shiftWindows(old: cuda.View, fresh: Fresh, keep: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_shift_windows(&old, &fresh.view, @intCast(keep), @intCast(fresh.layer), @intCast(fresh.row), @intCast(fresh.rows)));
}

/// Whether a commit shifts its layers' windows in one launch (`shiftEach`) and not a launch a layer: the same
/// copies. `ZSPARK_SHIFT_ONCE=0` keeps a launch a layer (`main` reads it).
pub var once: bool = true;
/// The most windows `shiftEach` takes.
pub const most = 64;

/// `shiftWindows` for each of `olds` (bf16 [1, taps, channels], one shape) from its own `freshes`' rows, `row`
/// elements apart, in one launch.
pub fn shiftEach(olds: []const cuda.View, freshes: []const cuda.View, row: usize, rows: usize, keep: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_shift_each(olds.ptr, freshes.ptr, @intCast(olds.len), @intCast(keep), @intCast(row), @intCast(rows)));
}

extern fn zs_qwen4_shift_each(olds: [*]const cuda.View, freshes: [*]const cuda.View, count: i64, keep: i64, fresh_row: i64, fresh_rows: i64) c_int;
extern fn zs_qwen4_shift_windows(old: *const cuda.View, fresh: *const cuda.View, keep: i64, fresh_layer: i64, fresh_row: i64, fresh_rows: i64) c_int;

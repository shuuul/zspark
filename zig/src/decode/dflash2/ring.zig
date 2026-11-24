//! DFlash2's sliding-window context ring: capacity from the window, a block and a tile, and the wrap a
//! restore copies in two runs. Parameterized by `Config.window`; GLM's exclusive window-1, Qwen Flash
//! Next's stored 2048 and Nemotron's none (a flat context, the whole of it seen) all go through here. The
//! wrap is the same bits given the same cap, ring flag, end and row count. Host only.
//!
//! WS-03's Qwen drafter imports `Shape`, `shape`, `Wrap` and `wrap` from here, WS-07's Nemotron the same.
//! Qwen's capture of taps is not this file.

const config_file = @import("config.zig");
const Config = config_file.Config;

/// Rows the context holds, and whether position p is row p mod cap.
pub const Shape = struct {
    cap: usize,
    ring: bool,
};

/// For a context of up to `capacity` positions; `ring`: kept in the ring of the window and a block where
/// that is smaller. No window: flat always, `capacity + block` rows.
pub fn shape(c: *const Config, capacity: usize, ring: bool) Shape {
    const rows = c.ringRows();
    const ringed = ring and rows > 0 and rows < capacity + c.block;
    return .{ .cap = if (ringed) rows else capacity + c.block, .ring = ringed };
}

/// The start slot of `rows` ending at `end`, and how many of them run without wrapping.
pub const Wrap = struct {
    slot: usize,
    run: usize,
};

pub fn wrap(cap: usize, ring: bool, end: usize, rows: usize) Wrap {
    const from = end - rows;
    const slot = if (ring) from % cap else from;
    return .{ .slot = slot, .run = @min(rows, cap - slot) };
}

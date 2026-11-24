//! Shard bounds of a host table: the first row of each shard, then the table's row count, a stored row's
//! width in bytes, and a binary search that places a row id in its shard. Flash Next's PLE (`models/qwen4/blocks/ple.zig`)
//! is the first user.

const std = @import("std");

/// The shard holding a row and the row's index in it.
pub const At = struct {
    shard: usize,
    row: usize,
};

/// Starts of `count` shards, then the table's row count at `starts[count]`. `width` is the stored row's bytes.
pub const Shards = struct {
    starts: []const u64,
    width: usize,
    count: usize,

    pub fn rows(self: Shards) u64 {
        return self.starts[self.count];
    }

    /// The shard holding row `id` and the row's index in it.
    pub fn locate(self: Shards, id: u64) error{RowOutOfRange}!At {
        if (id >= self.rows()) return error.RowOutOfRange;
        // the last shard that starts at or before the row
        var low: usize = 0;
        var high: usize = self.count;
        while (high - low > 1) {
            const mid = (low + high) / 2;
            if (self.starts[mid] <= id) low = mid else high = mid;
        }
        return .{ .shard = low, .row = @intCast(id - self.starts[low]) };
    }
};

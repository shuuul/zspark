//! `Graphs`: `rounds.zig`'s round captured for `{rows, context}`, which that file names as its own.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cache_file = @import("../../../decode/graphs.zig");
const forward = @import("../forward.zig");
const gdn = @import("../blocks/gdn.zig");
const model = @import("../model.zig");
const rounds = @import("rounds.zig");
const Gem = rounds.Gem;
const Round = rounds.Round;
const Shared = rounds.Shared;
const Slot = rounds.Slot;
const computeOver = rounds.computeOver;
const Parts = rounds.Parts;

/// The shared round as captured graphs, keyed by the round's total rows and the context bucket. The DeltaNet
/// side, the position and which slot a row belongs to are in the tables `stage` uploads, so one set serves
/// every slot. A capture records its launches and runs none.
pub const Graphs = struct {
    cache: cache_file.Cache(Key),
    gem: ?Gem = null,

    pub const Key = struct { rows: usize, context: usize };

    pub fn init(gpa: Allocator, m: model.Model, slots: usize, rows: usize, capacity: usize) !Graphs {
        var gem: ?Gem = null;
        if (m.attention) |attn| if (slots > 1) {
            gem = try Gem.init(slots, rows, attn.shape, capacity);
        };
        errdefer if (gem) |*held| held.deinit();
        return .{ .cache = try .init(gpa), .gem = gem };
    }

    /// The device bytes of buffers `init` asks, with nothing made: the GEMMs' regions. A captured graph's
    /// memory is the runtime's and in no buffer.
    pub fn bytesFor(m: model.Model, slots: usize, rows: usize, capacity: usize) usize {
        const attn = m.attention orelse return 0;
        return if (slots > 1) Gem.bytesFor(slots, rows, attn.shape, capacity) else 0;
    }

    pub fn deinit(self: *Graphs) void {
        self.cache.deinit();
        if (self.gem) |*held| held.deinit();
        self.* = undefined;
    }

    /// Captures `{rows, context}` unless it is held. `saves` is where each DeltaNet layer leaves a replay, when
    /// the keep is decided after the graph runs.
    pub fn prepare(self: *Graphs, m: model.Model, slots: []Slot, shared: *Shared, b: *forward.Buffers, round: *Round, rows: usize, context: usize, saves: ?[]gdn.Scratch) !void {
        return self.prepareOver(m, slots, shared, b, round, rows, context, saves, null);
    }

    /// `prepare` where the rank holds a share (`rounds.computeOver`): the graph holds the all-gathers of its
    /// partials among the kernels, so both ranks launch it in step.
    pub fn prepareOver(self: *Graphs, m: model.Model, slots: []Slot, shared: *Shared, b: *forward.Buffers, round: *Round, rows: usize, context: usize, saves: ?[]gdn.Scratch, parts: ?*const Parts) !void {
        std.debug.assert(rows >= 1 and rows < gdn.wide_rows and rows <= round.rows and rows <= b.rows);
        const key: Key = .{ .rows = rows, .context = context };
        if (self.cache.contains(key)) return;
        const used = round.used;
        round.used = rows;
        defer round.used = used;
        _ = try self.cache.capture(key, computeOver, .{ m, slots, shared, b, round, &.{}, context, .all, if (self.gem) |*held| held else null, saves, parts });
    }

    pub fn launch(self: *Graphs, rows: usize, context: usize) !void {
        try self.cache.launch(.{ .rows = rows, .context = context });
    }
};

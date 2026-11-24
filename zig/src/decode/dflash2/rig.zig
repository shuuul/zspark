//! The DFlash2 drafter in the MTP layer's place for a model's session, over the ranks the process holds:
//! `generate DIR drafter=DRAFTER`, `chat` and `serve --drafter DRAFTER`. Its directory is opened and held to
//! the main model before anything loads (`fit`), its share is loaded as the tests load it (nothing written),
//! and its passes are captured graphs unless the command is `eager`. Drafts only propose: the main model
//! chooses every token from its own rows.
//!
//! What a model keeps is its `dflash2/rig.zig`: how a directory is opened (`Opened`), the heads its drafts are
//! scored with (`draftHeads`), and `Target.initRig` (`pass.zig`'s header), which taps the main model's buffers
//! and calls `begin` and `finish`. `opened` is a model's `Opened` (`cli.Opened` or its own): `path`, `config`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const sampling = @import("../sampling.zig");
const safetensors = @import("../../weights/safetensors.zig");
const config_file = @import("config.zig");
const drafter_file = @import("drafter.zig");
const pass = @import("pass.zig");
const weights = @import("weights.zig");

const world = config_file.world;

/// Whether the opened drafter is one for the main model `c` run to its first `layers` layers; said where not.
/// A tap after a layer that is not loaded becomes the last loaded layer's: a trial of the first layers drafts
/// from rows the drafter was not trained on, and both ranks from the same ones.
/// `c` is the main model's config: its `hidden`, `vocab` and `layers`.
pub fn fit(arena: Allocator, opened: anytype, c: anytype, layers: usize) Allocator.Error!bool {
    const d = &opened.config;
    // The model's own layer count, not the loaded ones': a trial of the first layers keeps its drafter.
    if (try d.mismatch(arena, c.hidden, c.vocab, c.layers)) |why| {
        std.log.err("{s}: {s}", .{ opened.path, why });
        return false;
    }
    const taps = try arena.dupe(u32, d.taps);
    var moved = false;
    for (taps) |*tap| {
        if (tap.* < layers) continue;
        tap.* = @intCast(layers - 1);
        moved = true;
    }
    if (moved) std.log.warn("{s}: taps after layers {any}, of which {d} are loaded: read after layers {any}", .{ opened.path, d.taps, layers, taps });
    d.taps = taps;
    return true;
}

/// `path` opened into `opened` and held to the model (`fit`); false with the reason said.
pub fn enter(init: std.process.Init, opened: anytype, path: []const u8, c: anytype, layers: usize) !bool {
    return try opened.open(init, path, .say) and try fit(init.arena.allocator(), opened, c, layers);
}

/// `enter` from another machine's drafter (`serve --weights-from-peer` on rank 1): its `config.json` text and the
/// checkpoint of its files' headers (`safetensors.Checkpoint.fromHeads`), which `opened` takes. Nothing is opened
/// under `path`; the tensors' bytes come over the link at the load (`Source.cross`).
pub fn given(init: std.process.Init, opened: anytype, path: []const u8, config: []const u8, checkpoint: safetensors.Checkpoint, c: anytype, layers: usize) !bool {
    return try opened.given(init, path, config, checkpoint) and try fit(init.arena.allocator(), opened, c, layers);
}

/// The drafter's tensors as its load reads them (`weights.Weights.load`'s and `Codebooks.load`'s `source`): at
/// their staged bytes where `lay` staged them, through the mapping otherwise.
pub const Source = struct {
    checkpoint: *const safetensors.Checkpoint,
    stage: safetensors.Stage = .{},
    /// The stage's bytes as `lay` laid them out.
    laid: []u8 = &.{},

    pub fn deinit(self: *Source, gpa: Allocator) void {
        self.stage.deinit(gpa);
        self.* = undefined;
    }

    pub fn get(self: *const Source, tensor_name: []const u8) ?safetensors.Tensor {
        var tensor = self.checkpoint.get(tensor_name) orelse return null;
        if (self.stage.moved(tensor.data)) |staged| tensor.data = staged;
        return tensor;
    }

    /// Every tensor's bytes laid out in the stage, alike on both ranks (`safetensors.Stage.lay`), and read there
    /// from the file with direct reads where `read`: under `serve --weights-from-peer` the bytes rank 0 sends (rank
    /// 1 receives them, `cross`), otherwise each rank's own. `pin` makes the buffer known to the device (and keeps
    /// its pages in memory); null where nothing crosses, or in a test. A read that fails leaves nothing staged, and
    /// the load reads through the mapping; under `pin` it is refused.
    pub fn lay(self: *Source, gpa: Allocator, read: bool, pin: ?safetensors.Stage.Pin) ![]u8 {
        var ranges: std.ArrayList([]const u8) = .empty;
        defer ranges.deinit(gpa);
        for (self.checkpoint.tensors.values()) |tensor| try ranges.append(gpa, tensor.data);
        self.stage.pin = pin;
        const bytes = try self.stage.lay(gpa, self.checkpoint, ranges.items, true) orelse return error.OutOfMemory;
        if (pin != null and !self.stage.pinned) {
            std.log.err("the drafter's {d} bytes cannot be page-locked: they would cross unpinned, and could be swapped out", .{bytes.len});
            return error.Unsupported;
        }
        if (read and !self.stage.fetch() and pin != null) {
            std.log.err("rank 0: the drafter's file is not read with direct reads", .{});
            return error.Unsupported;
        }
        self.laid = bytes;
        return bytes;
    }

    /// `serve --weights-from-peer`: rank 0 sends its stage (`lay`, read) whole over `comm`, rank 1 receives it into
    /// its own, the same layout; each loads its share from there. Nothing of it is written anywhere: rank 1's
    /// bytes are page-locked memory alone, freed with the source. `comm` is a `*const nccl.Comm`.
    pub fn cross(self: *Source, comm: anytype) !void {
        const bytes = self.laid;
        const move: @TypeOf(comm.*).Move = if (comm.rank == 0)
            .{ .peer = 1, .send = bytes.ptr, .send_count = bytes.len, .recv = null, .recv_count = 0 }
        else
            .{ .peer = 0, .send = null, .send_count = 0, .recv = bytes.ptr, .recv_count = bytes.len };
        try comm.exchange(.u8, &.{move});
        // the host reads them next (`share.take`)
        try cuda.synchronize();
        cuda.phases.mark("the drafter crossed");
        if (comm.rank != 0) std.log.info("rank 1: the drafter's {d:.2} GiB received from rank 0 over the link, none read from a disk", .{@as(f64, @floatFromInt(bytes.len)) / (1 << 30)});
    }
};

/// A `Source` laid out and read on a thread of its own from a session's start, before the shares' load: a stage
/// allocated, faulted in and page-locked once the device holds the shares cost 5.5 s for the drafter's 2.18 GiB
/// against 0.3 s before them, and its tensors read through the mapping then 1.7 to 2.3 s against 0.6 from a
/// stage (`ZSPARK_LOAD_PHASES`, 2026-10-21).
pub const Early = struct {
    source: Source,
    thread: ?std.Thread = null,
    failed: ?anyerror = null,

    /// Begins `source.lay` (`read`, `pin` as there); `wait` before the source is read.
    pub fn begin(self: *Early, gpa: Allocator, read: bool, pin: ?safetensors.Stage.Pin) void {
        self.thread = std.Thread.spawn(.{}, run, .{ self, gpa, read, pin }) catch null;
        if (self.thread == null) run(self, gpa, read, pin);
    }

    fn run(self: *Early, gpa: Allocator, read: bool, pin: ?safetensors.Stage.Pin) void {
        _ = self.source.lay(gpa, read, pin) catch |err| {
            self.failed = err;
            return;
        };
        cuda.phases.mark("the drafter's stage is read");
    }

    /// The source, laid out and read.
    pub fn wait(self: *Early) !*Source {
        if (self.thread) |thread| thread.join();
        self.thread = null;
        if (self.failed) |err| return err;
        return &self.source;
    }

    pub fn deinit(self: *Early, gpa: Allocator) void {
        if (self.thread) |thread| thread.join();
        self.source.deinit(gpa);
        self.* = undefined;
    }
};

/// One word for the opened drafter where both ranks must hold the same: its tensors' headers (`digest`, the
/// checkpoint's as `join.headers` makes it) and the taps it reads.
pub fn name(arena: Allocator, opened: anytype, digest: [32]u8) Allocator.Error![]const u8 {
    return std.fmt.allocPrint(arena, "dflash2:{x}", .{std.hash.Wyhash.hash(std.mem.readInt(u64, digest[0..8], .little), std.mem.sliceAsBytes(opened.config.taps))});
}

/// The drafter's share, context and passes over `held` ranks.
pub fn Rig(comptime held: usize, comptime Target: type) type {
    return struct {
        const Self = @This();

        w: [held]weights.Weights,
        books: weights.Codebooks,
        pair: pass.PairOf(held, Target),
        passes: ?pass.Graphs,
        d: drafter_file.DrafterOf(held, Target),
        gpa: Allocator,

        /// In place, after `load`: the drafter keeps the pair's address and the pair its graphs'. The model's
        /// (`Target.initRig`, and `Target.initRigHeld` where it has one): the main model's buffers tapped,
        /// then `begin` and `finish`. On a failure `load`'s share is the caller's to `unload`.
        pub const init = Target.initRig;
        pub const initHeld = Target.initRigHeld;

        /// The drafter's share on each held rank and its codebooks, from `source` (`Early`'s, laid out and read),
        /// which the caller frees once the device has read them (after a `cuda.synchronize`): right after the
        /// model's shares, so that its stage is gone before the caches and the heads are made (`Early`). `ranks`
        /// are the held ranks' numbers, `stored` loads the linears BF16 as the file has them, not NVFP4 made at
        /// load (`.mixed`). `crossing`: the ranks' communicator under `serve --weights-from-peer`, over which rank
        /// 1 is sent the drafter (`Source.cross`), or null. The linears are loaded for the rows of one update
        /// (`pass.batch`, `pass.wide` where `Target.wide`). `init` makes the rest; `unload` undoes this one where
        /// it does not run.
        pub fn load(self: *Self, arena: Allocator, gpa: Allocator, opened: anytype, source: *Source, ranks: [held]usize, pool: *cuda.Scratch, stored: bool, crossing: anytype) !void {
            const c = &opened.config;
            if (@TypeOf(crossing) != @TypeOf(null)) if (crossing) |comm| try source.cross(comm);
            const source_view: *const Source = source;
            const rows: cuda.Scratch.Rows = .{ .pool = pool, .count = if (Target.wide) pass.wide else pass.batch };
            self.gpa = gpa;
            var done: usize = 0;
            errdefer for (self.w[0..done]) |*one| one.deinit();
            for (&self.w, ranks) |*one, rank| {
                one.* = try .load(arena, gpa, source_view, c, rank, if (stored) .stored else .mixed, rows);
                done += 1;
            }
            cuda.phases.mark("the drafter's share is loaded");
            self.books = try .load(gpa, source_view, c);
            cuda.phases.mark("the drafter's codebooks are widened");
        }

        /// What `load` made, where `init` does not follow.
        pub fn unload(self: *Self) void {
            self.books.deinit(self.gpa);
            for (&self.w) |*one| one.deinit();
        }

        /// `init`'s first half: the pair for `capacity` positions of a sequence over ranks `first` onward, the
        /// others' through `gather`, told by `lead` where this rank leads, and its graphs unless `eager`.
        /// `abandon` undoes it where `finish` fails.
        pub fn begin(self: *Self, gpa: Allocator, c: *const config_file.Config, capacity: usize, first: usize, gather: Target.ranks.Gather, lead: ?*Target.lead.Lead, eager: bool) !void {
            self.pair = try .initFrom(gpa, self.w, c, capacity, true, c.vocab / world, first, gather);
            errdefer self.pair.deinit();
            cuda.phases.mark("the drafter's context is made");
            // a rank that leads tells the pair's passes as the main model's
            self.pair.lead = lead;
            self.passes = if (eager) null else try .init(gpa);
            if (self.passes) |*graphs| self.pair.graphs = graphs;
        }

        /// What `begin` made.
        pub fn abandon(self: *Self) void {
            if (self.passes) |*graphs| graphs.deinit();
            self.pair.deinit();
        }

        /// `init`'s second half: the drafter over `target`, scoring with `heads` (`draftHeads`) for up to an
        /// update's rows, and `widened` the bf16 rows they were made of where the head is not BF16. `heads` are
        /// the drafter's, freed with it, where `Target.owns_heads` (on an error here too); the caller's otherwise.
        /// Null heads borrow the target's stored proposer and never enter owned cleanup.
        pub fn finish(self: *Self, gpa: Allocator, c: *const config_file.Config, target: Target.Of(held), heads: ?[held]weights.Linear, widened: ?[held]cuda.View, settings: sampling.Sampling) !void {
            var owned = heads;
            errdefer if (Target.owns_heads) {
                if (owned) |*all| for (all) |*one| one.deinit();
            };
            self.d = try .init(gpa, c, &self.pair, target, self.books, settings);
            self.d.heads = owned;
            self.d.widened = widened;
            cuda.phases.mark("the drafter is made");
        }

        /// The pair's passes run eagerly from here on, and their graphs are freed: captured over the taps of
        /// buffers the drafter no longer reads (the pool's seats, `models/qwen4/pool/hall_engine.zig`).
        pub fn dropPasses(self: *Self) cuda.Error!void {
            // they launch on a stream of their own
            try cuda.synchronize();
            self.pair.graphs = null;
            if (self.passes) |*graphs| graphs.deinit();
            self.passes = null;
        }

        pub fn deinit(self: *Self) void {
            var heads = self.d.heads;
            self.d.heads = null;
            self.d.deinit();
            if (Target.owns_heads) if (heads) |*held_heads| for (held_heads) |*one| one.deinit();
            if (self.passes) |*graphs| graphs.deinit();
            self.pair.deinit();
            self.books.deinit(self.gpa);
            for (&self.w) |*one| one.deinit();
            self.* = undefined;
        }
    };
}

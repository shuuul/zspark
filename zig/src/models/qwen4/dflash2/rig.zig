//! Qwen3.8 Flash Next's DFlash2 drafter in the MTP layer's place for a session (`../session/session.zig`):
//! `decode/dflash2/rig.zig` with what is Qwen's. Its directory is opened with Qwen's stored sliding_window
//! (`Opened`), the linears loaded as `ZSPARK_DRAFTER_STORAGE` says (`drafter_stored`). Capture of taps is
//! `Buffers.setTaps` with the drafter's layers, default f32-acc mean. Drafts only propose: the target chooses
//! every token from its own rows through `decode/draft.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const sampling = @import("../../../decode/sampling.zig");
const shared = @import("../../../decode/dflash2/rig.zig");
const config_file = @import("../../../decode/dflash2/config.zig");
const layout_file = @import("../../../decode/dflash2/layout.zig");
const weights = @import("../../../decode/dflash2/weights.zig");
const qwen4_config = @import("../config.zig");
const forward_file = @import("../forward.zig");
const model_file = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const drafter_file = @import("drafter.zig");
const pass = @import("pass.zig");
const safetensors = @import("../../../weights/safetensors.zig");

const world = config_file.world;

pub const Source = shared.Source;
pub const Early = shared.Early;
pub const name = shared.name;

/// The drafter's share, context and passes over `held` ranks.
pub fn Rig(comptime held: usize) type {
    return shared.Rig(held, pass.Target);
}

/// The pinned Flash Next DFlash2 drafter's sizes (`~/models/Qwen3.8-Flash-Next-DFlash2`).
pub const published = struct {
    pub const hidden: usize = 2560;
    pub const layers: usize = 5;
    pub const heads: usize = 32;
    pub const kv_heads: usize = 8;
    pub const head_dim: usize = 128;
    pub const intermediate: usize = 7680;
    pub const window: usize = 2048;
    pub const block: usize = 8;
    pub const group: usize = 16;
    pub const selector_rank: usize = 256;
    pub const top_k: usize = 16;
    pub const vocab: usize = 248320;
    pub const mask: u32 = 248077;
    pub const taps = [_]u32{ 5, 15, 25, 35, 45 };
    pub const target_layers: usize = 48;
};

/// A checkpoint directory opened with Qwen's stored sliding_window. Not `cli.Opened`, whose `configure` and
/// `given` read `Config.read`/`parse`: GLM's exclusive window, and a refusal where heads x head_dim is not
/// hidden, which Qwen's GQA drafter (4096 over 2560) is not. Its other steps are `cli.Opened`'s.
pub const Opened = struct {
    path: []const u8 = "",
    dir: ?std.Io.Dir = null,
    config: config_file.Config = undefined,
    checkpoint: ?safetensors.Checkpoint = null,
    layout: layout_file.Layout = undefined,
    diagnostic: config_file.Diagnostic = .{},

    pub const Missing = enum { raise, say };

    pub fn open(self: *Opened, init: std.process.Init, path: []const u8, missing: Missing) !bool {
        return try self.enter(init.io, path, missing) and try self.configure(init) and try self.resolve(init);
    }

    pub fn enter(self: *Opened, io: std.Io, path: []const u8, missing: Missing) !bool {
        self.path = path;
        self.dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| switch (missing) {
            .raise => return err,
            .say => {
                std.log.err("{s}: {t}", .{ path, err });
                return false;
            },
        };
        return true;
    }

    pub fn configure(self: *Opened, init: std.process.Init) !bool {
        self.config = config_file.Config.readStored(init.arena.allocator(), init.io, self.dir.?, &self.diagnostic) catch |err| return self.refused(err);
        return true;
    }

    pub fn resolve(self: *Opened, init: std.process.Init) !bool {
        self.checkpoint = try .open(init.gpa, init.io, .cwd(), self.path);
        self.layout = layout_file.Layout.init(init.arena.allocator(), &self.config, &self.checkpoint.?, &self.diagnostic) catch |err| return self.refused(err);
        return true;
    }

    pub fn given(self: *Opened, init: std.process.Init, path: []const u8, config: []const u8, checkpoint: safetensors.Checkpoint) !bool {
        self.path = path;
        self.checkpoint = checkpoint;
        self.config = config_file.Config.parseStored(init.arena.allocator(), config, &self.diagnostic) catch |err| return self.refused(err);
        self.layout = layout_file.Layout.init(init.arena.allocator(), &self.config, &self.checkpoint.?, &self.diagnostic) catch |err| return self.refused(err);
        return true;
    }

    fn refused(self: *const Opened, err: anyerror) bool {
        std.log.err("{s}: {s}", .{ self.path, if (self.diagnostic.message.len > 0) self.diagnostic.message else @errorName(err) });
        return false;
    }

    pub fn close(self: *Opened, io: std.Io) void {
        if (self.checkpoint) |*checkpoint| checkpoint.close();
        if (self.dir) |dir| dir.close(io);
    }
};

/// Whether the opened drafter is one for the main model `c` run to its first `layers` layers; said where not.
/// A tap after a layer that is not loaded becomes the last loaded layer's.
pub fn fit(arena: Allocator, opened: *Opened, c: *const qwen4_config.Config, layers: usize) Allocator.Error!bool {
    return shared.fit(arena, opened, c, layers);
}

/// `path` opened into `opened` and held to the model (`fit`); false with the reason said.
pub fn enter(init: std.process.Init, opened: *Opened, path: []const u8, c: *const qwen4_config.Config, layers: usize) !bool {
    return shared.enter(init, opened, path, c, layers);
}

pub fn given(init: std.process.Init, opened: *Opened, path: []const u8, config: []const u8, checkpoint: safetensors.Checkpoint, c: *const qwen4_config.Config, layers: usize) !bool {
    return shared.given(init, opened, path, config, checkpoint, c, layers);
}

/// Each held rank's half of the target's vocabulary the drafter proposes by, from the face on the device:
/// null for stored BF16 (the target lends it), or the four-bit copy TensorFold's drafters propose by, its
/// picks scored again against the stored rows (`pass.rescore`, spec 135 WS-04). One-rank target: split the
/// whole head. Held ranks: each share already holds its half (`shareHead`).
pub fn draftHeads(comptime held: usize, arena: Allocator, gpa: Allocator, m: model_file.Model, stored: bool, rows: cuda.Scratch.Rows) !?[held]weights.Linear {
    if (stored) return null;
    const all = try drafter_file.splitHeads(arena, gpa, m, rows);
    if (held == world) return all;
    var heads: [held]weights.Linear = undefined;
    var made: usize = 0;
    errdefer for (heads[0..made]) |*one| one.deinit();
    for (&heads, all[0..held]) |*one, src| {
        one.* = src;
        made += 1;
    }
    if (held < world) for (all[held..]) |*extra| extra.deinit();
    return heads;
}

/// Four-bit copies of held ranks' heads; null in stored mode, which borrows through the target.
pub fn draftHeadsHeld(comptime held: usize, arena: Allocator, gpa: Allocator, m: held_file.Held(held).Model, stored: bool, rows: cuda.Scratch.Rows) !?[held]weights.Linear {
    if (stored) return null;
    var ranks: [held]model_file.Model = undefined;
    for (&ranks, m.ranks) |*one, rank| one.* = rank;
    return try drafter_file.splitHeadsFromShares(held, arena, gpa, ranks, rows);
}

/// How `serve --drafter` and `ranks-check ... drafter=` hold Flash Next's DFlash2 drafter: every linear and the
/// proposing head BF16 as stored (true), or `.mixed` with the four-bit head as TensorFold's drafters (false, the
/// default since 2026-10-25: +7.2% decode at two ranks, spec 135 WS-03). `main.zig` sets it from `ZSPARK_DRAFTER_STORAGE`; the reply's tokens are the same.
pub var drafter_stored: bool = false;

/// `Rig.init`: the one-rank target `m` and its buffers `b`, not yet tapped and no graph captured over them.
/// `heads` (`draftHeads`) are the rig's once the pair is made: freed with it, or where the drafter fails.
pub fn initRig(self: anytype, gpa: Allocator, opened: *const Opened, heads: ?[world]weights.Linear, widened: ?[world]cuda.View, m: model_file.Model, b: *forward_file.Buffers, capacity: usize, settings: sampling.Sampling, eager: bool) !void {
    var owned = heads;
    var handed = false;
    errdefer if (!handed) {
        if (owned) |*all| for (all) |*one| one.deinit();
    };
    const c = &opened.config;
    try b.setTaps(c.taps, .f32_acc);
    try self.begin(gpa, c, capacity, 0, .loopback, null, eager);
    errdefer self.abandon();
    const target = try drafter_file.Of(world).one(gpa, m, b);
    handed = true;
    try self.finish(gpa, c, target, heads, widened, settings);
}

/// `Rig.initHeld`: over the ranks a process holds, each rank's own taps and already-halved head. `m.lead` is
/// the pair's, so rank 1 can replay the flash ops the leader tells. `heads` as `initRig`'s.
pub fn initRigHeld(self: anytype, gpa: Allocator, opened: *const Opened, heads: anytype, widened: anytype, m: anytype, b: anytype, capacity: usize, settings: sampling.Sampling, eager: bool) !void {
    var owned = heads;
    var handed = false;
    errdefer if (!handed) {
        if (owned) |*all| for (all) |*one| one.deinit();
    };
    const c = &opened.config;
    try b.setTaps(c.taps, .f32_acc);
    try self.begin(gpa, c, capacity, m.first, m.gather, m.lead, eager);
    errdefer self.abandon();
    const target = try @TypeOf(self.d.target).heldBy(gpa, m, b);
    handed = true;
    try self.finish(gpa, c, target, heads, widened, settings);
}

/// A process's split drafter and per-stream contexts, allocated before the target pool's fit. The
/// target's buffers are bound only after its engine has reached its permanent address, before captures.
pub const Pool = struct {
    flash: Rig(1),
    seats: pass.SeatsOf(1),
    buffers: held_file.Held(1).Buffers = undefined,
    heads: ?[1]weights.Linear = null,
    bound: bool = false,

    pub fn create(arena: Allocator, gpa: Allocator, opened: *const Opened, m: held_file.Held(1).Model, capacity: usize, slots: usize, stored: bool) !*Pool {
        const self = try gpa.create(Pool);
        errdefer gpa.destroy(self);
        self.* = .{ .flash = undefined, .seats = undefined };
        var source: Source = .{ .checkpoint = &opened.checkpoint.? };
        try self.flash.load(arena, gpa, opened, &source, .{m.first}, m.ranks[0].scratch, stored, null);
        errdefer self.flash.unload();
        try self.flash.begin(gpa, &opened.config, capacity, m.first, m.gather, m.lead, true);
        errdefer self.flash.abandon();
        self.seats = try .init(gpa, &self.flash.pair, slots);
        errdefer self.seats.deinit();
        self.heads = try draftHeadsHeld(1, arena, gpa, m, stored, .{ .pool = m.ranks[0].scratch, .count = @max(@import("../../../decode/dflash2/pass.zig").batch, slots * (opened.config.block - 1)) });
        return self;
    }

    /// `self.buffers` is where the target's held buffers stand for the rig's life. Ownership stays with the
    /// caller on failure: a `heldBy` that fails is its own, and `finish` frees `heads` itself where it fails
    /// (`Target.owns_heads`), so `self.heads` is nulled as they are handed over.
    pub fn bind(self: *Pool, m: held_file.Held(1).Model, buffers: held_file.Held(1).Buffers) !void {
        std.debug.assert(!self.bound);
        self.buffers = buffers;
        const target = try drafter_file.Of(1).heldBy(self.flash.gpa, m, &self.buffers);
        const heads = self.heads;
        self.heads = null;
        try self.flash.finish(self.flash.gpa, self.flash.pair.ranks[0].c, target, heads, null, .{ .seed = 0 });
        self.bound = true;
    }

    pub fn destroy(self: *Pool) void {
        const gpa = self.flash.gpa;
        self.seats.deinit();
        if (self.bound) self.flash.deinit() else {
            if (self.heads) |*heads| for (heads) |*head| head.deinit();
            self.flash.abandon();
            self.flash.unload();
        }
        gpa.destroy(self);
    }
};

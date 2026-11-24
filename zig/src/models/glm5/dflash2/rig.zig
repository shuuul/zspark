//! GLM-5.3-Flash's DFlash2 drafter in the MTP layer's place for a session (`../session/session.zig`; spec 045
//! WS-09): `decode/dflash2/rig.zig` with what is GLM's. The directory is opened by `cli.Opened` (GLM's exclusive
//! window), its share loaded as the tests load it (the large linears four bits in memory). It scores its drafts
//! with a four-bit copy of the main model's rows of the vocabulary, made in memory at load (`draftHeads`,
//! TensorFold's `draft_head`; spec 067's recommendation, switched on by spec 068) and the session's, which the
//! MTP layer scores with too: drafts only propose, the main model chooses every token from its own rows, and
//! the head is 5.7 ms of a block read as stored and 1.5 so.
//!
//! The drafter's weights are CC BY-NC-ND 4.0, the user's own copy on disk: nothing here writes them or anything
//! made of them, and they leave the process only over the link into rank 1's memory (`Source.cross`, root
//! `AGENTS.md`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const sampling = @import("../../../decode/sampling.zig");
const shared = @import("../../../decode/dflash2/rig.zig");
const config_file = @import("../../../decode/dflash2/config.zig");
const layout_file = @import("../../../decode/dflash2/layout.zig");
const weights = @import("../../../decode/dflash2/weights.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const glm5_config = @import("../layers/config.zig");
const held_file = @import("../layers/ranks_forward.zig");
const pass = @import("pass.zig");

pub const Opened = cli.Opened(config_file.Config, layout_file.Layout, config_file.Diagnostic);
pub const Source = shared.Source;
pub const Early = shared.Early;
pub const name = shared.name;

/// The drafter's share, context and passes over `held` ranks.
pub fn Rig(comptime held: usize) type {
    return shared.Rig(held, pass.Target);
}

/// Whether the opened drafter is one for the main model `c` run to its first `layers` layers; said where not.
/// A tap after a layer that is not loaded becomes the last loaded layer's: a trial of the first layers drafts
/// from rows the drafter was not trained on, and both ranks from the same ones. The model's own layer count,
/// not the loaded ones', is held: a trial of the first layers keeps its drafter.
pub fn fit(arena: Allocator, opened: *Opened, c: *const glm5_config.Config, layers: usize) Allocator.Error!bool {
    return shared.fit(arena, opened, c, layers);
}

/// `path` opened into `opened` and held to the model (`fit`); false with the reason said.
pub fn enter(init: std.process.Init, opened: *Opened, path: []const u8, c: *const glm5_config.Config, layers: usize) !bool {
    return shared.enter(init, opened, path, c, layers);
}

/// `enter` from another machine's drafter (`serve --weights-from-peer` on rank 1): its `config.json` text and the
/// checkpoint of its files' headers (`safetensors.Checkpoint.fromHeads`), which `opened` takes. Nothing is opened
/// under `path`; the tensors' bytes come over the link at the load (`Source.cross`).
pub fn given(init: std.process.Init, opened: *Opened, path: []const u8, config: []const u8, checkpoint: safetensors.Checkpoint, c: *const glm5_config.Config, layers: usize) !bool {
    return shared.given(init, opened, path, config, checkpoint, c, layers);
}

/// Each held rank's four-bit copy of its rows of the vocabulary (`weights.draftHead`): what a session's
/// drafters score with, the MTP layer and DFlash2 alike, any number of rows a call (a four-bit linear lends no
/// scratch). The rows are read back from the model's head on the device, which holds the checkpoint's bytes:
/// the file is not read again, and its pages stay out of the cache (`src/AGENTS.md`). In memory only. Over a
/// head not BF16 (FP4-Local's MXFP8), `widened` is given the bf16 rows the copy was made of, on the device,
/// where it asks for them: what DFlash2 scores its picks against again (`pass.rescore`); otherwise null.
pub fn draftHeads(comptime held: usize, arena: Allocator, gpa: Allocator, m: held_file.Held(held).Model, rows: cuda.Scratch.Rows, widened: ?*?[held]cuda.Buffer) ![held]weights.Linear {
    var heads: [held]weights.Linear = undefined;
    var made: usize = 0;
    errdefer for (heads[0..made]) |*one| one.deinit();
    var kept: [held]cuda.Buffer = undefined;
    var uploaded: usize = 0;
    errdefer for (kept[0..uploaded]) |*one| one.deinit();
    try cuda.synchronize();
    cuda.phases.mark("the draft heads begin");
    for (&heads, m.ranks) |*one, rank| {
        // the copy is made of bf16 rows: a BF16 head's bytes, or an MXFP8 head's (FP4-Local) widened, exact
        const stored, const k = switch (rank.head) {
            .bf16 => |head| read: {
                const bytes = try gpa.alloc(u8, head.weight.bytes());
                errdefer gpa.free(bytes);
                // touched first, as `bf16Rows`' copies
                @memset(bytes, 0);
                try head.weight.download(bytes);
                break :read .{ bytes, head.k };
            },
            .mxfp8 => |head| .{ try head.bf16Rows(gpa), head.k },
            else => {
                std.log.err("a four-bit copy of the head is made of a BF16 or MXFP8 head; this checkpoint stores it {t}", .{rank.head});
                return error.Unsupported;
            },
        };
        defer gpa.free(stored);
        if (widened != null and rank.head != .bf16) {
            kept[uploaded] = try .init(.bf16, &.{ stored.len / (2 * k), k });
            uploaded += 1;
            try kept[uploaded - 1].upload(stored);
        }
        one.* = try weights.draftHead(arena, gpa, stored, k, rows);
        made += 1;
    }
    if (widened) |out| out.* = if (uploaded == held) kept else null;
    return heads;
}

/// `Rig.init`: `m` and `b` are the main model's held pair, `b` not yet tapped and no graph captured over it;
/// `capacity` the positions of a sequence. `heads` are the caller's (`draftHeads`): what the drafter scores
/// with, for up to `pass.batch` rows; `widened` the bf16 rows they were made of where the head is not BF16
/// (`draftHeads`' `widened`). A rank that leads tells the pair's passes as the main model's.
pub fn initRig(self: anytype, gpa: Allocator, opened: *const Opened, heads: anytype, widened: anytype, m: anytype, b: anytype, capacity: usize, settings: sampling.Sampling, eager: bool) !void {
    const c = &opened.config;
    try b.setTaps(c.taps);
    try self.begin(gpa, c, capacity, m.ranks[0].rank, m.gather, m.lead, eager);
    errdefer self.abandon();
    try self.finish(gpa, c, .{ .m = m, .main = b }, heads, widened, settings);
}

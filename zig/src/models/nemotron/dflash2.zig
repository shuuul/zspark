//! Nemotron-3.5-Transcribe-30B-A4B's DFlash2 drafter (`NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`,
//! OpenMDW-1.1, bf16 as stored) in the MTP layer's place for a session (spec 151 WS-07):
//! `decode/dflash2/` whole — the shared schema names every one of its tensors (`layout.kinds`), its
//! `dflash_config` (block 16, mask 990, selector 256 over the top 16, grouped convolution of two taps over
//! groups of 16 channels) is what the kernels are written for, and its six taps of the target's 52 layers
//! are the config's `target_layer_ids`. What is Nemotron's alone is here: the config's `sliding_window` is
//! null and every layer `full_attention`, so the drafter sees the whole context (`Config.parseFull`, vLLM
//! `qwen3_dflash.py:_resolve_layer_attention`), and its GQA's q_inner (4,096) is not hidden (2,688), which
//! Qwen's `parseStored` already allows.
//!
//! What the target owes it (ForwardSlice's side, the `Target` of `decode/dflash2/pass.zig`'s header):
//! `setTaps(c.taps)` leaving every kept row's hidden after layers 1, 5, 19, 29, 41 and 51 bf16 [rows, 6 x
//! 2688] (one residual stream: the row itself, no mean), each rank's embedding and its rows of the
//! vocabulary [65536, 2688] bf16 for `Main`, and `commit(keep)` rolling the Mamba state of a rejected
//! draft back — `mamba.zig`'s parity replay does that: a window's state commits through the kept rows
//! alone (`AGENTS.md`, "DFlash2"). `initRig`/`initRigHeld` and `draftHeads` join when the target's forward
//! does (WS-02); the drafter's linears are loaded `.mixed` (the large ones NVFP4 in memory) unless a flag
//! asks for stored, and its drafts are scored with a four-bit copy of the head as GLM's are, the picks
//! scored again against the stored rows (`pass.rescore`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cli = @import("../../cli.zig");
const shared = @import("../../decode/dflash2/rig.zig");
const config_file = @import("../../decode/dflash2/config.zig");
const layout_file = @import("../../decode/dflash2/layout.zig");
const nemotron_config = @import("config.zig");
const safetensors = @import("../../weights/safetensors.zig");

pub const Source = shared.Source;
pub const Early = shared.Early;
pub const name = shared.name;

/// The published drafter's sizes (`hf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`): six full-attention
/// layers beside the backbone's 52, no window, a GQA whose q_inner is not hidden.
pub const published = struct {
    pub const hidden: usize = 2688;
    pub const layers: usize = 6;
    pub const heads: usize = 32;
    pub const kv_heads: usize = 2;
    pub const head_dim: usize = 128;
    pub const intermediate: usize = 6144;
    pub const window: ?usize = null;
    pub const block: usize = 16;
    pub const group: usize = 16;
    pub const selector_rank: usize = 256;
    pub const top_k: usize = 16;
    pub const vocab: usize = 131072;
    pub const mask: u32 = 990;
    pub const taps = [_]u32{ 1, 5, 19, 29, 41, 51 };
    pub const target_layers: usize = 52;
};

/// A checkpoint directory opened with Nemotron's null window (`parseFull`): the drafter's layers are all
/// `full_attention` so it sees the whole context, and its stored `sliding_window` null is required. The
/// rest is `cli.Opened`'s shape.
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
        self.config = config_file.Config.readFull(init.arena.allocator(), init.io, self.dir.?, &self.diagnostic) catch |err| return self.refused(err);
        return true;
    }

    pub fn resolve(self: *Opened, init: std.process.Init) !bool {
        self.checkpoint = try .open(init.gpa, init.io, .cwd(), self.path);
        self.layout = layout_file.Layout.init(init.arena.allocator(), &self.config, &self.checkpoint.?, &self.diagnostic) catch |err| return self.refused(err);
        return true;
    }

    /// `open`'s steps from another machine's (`serve --weights-from-peer` on rank 1): the drafter's
    /// `config.json` text and the checkpoint of its files' headers, which `opened` takes.
    pub fn given(self: *Opened, init: std.process.Init, path: []const u8, config: []const u8, checkpoint: safetensors.Checkpoint) !bool {
        self.path = path;
        self.checkpoint = checkpoint;
        self.config = config_file.Config.parseFull(init.arena.allocator(), config, &self.diagnostic) catch |err| return self.refused(err);
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

/// Whether the opened drafter is one for the main model `c` run to its first `layers` layers; said where
/// not. `c.backbone`: hidden 2,688, vocabulary 131,072 and 52 layers. A tap after a layer that is not
/// loaded becomes the last loaded layer's.
pub fn fit(arena: Allocator, opened: *Opened, c: *const nemotron_config.Config, layers: usize) Allocator.Error!bool {
    return shared.fit(arena, opened, c.backbone, layers);
}

/// `path` opened into `opened` and held to the model (`fit`); false with the reason said.
pub fn enter(init: std.process.Init, opened: *Opened, path: []const u8, c: *const nemotron_config.Config, layers: usize) !bool {
    return shared.enter(init, opened, path, c.backbone, layers);
}

/// The published drafter's directory name under `~/models`, the default `drafter=DIR`/`--drafter` gives
/// when none is named (spec 151 criterion 74: DFlash2 is the default drafter, MTP and serial explicit).
pub const dir_name = "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2";

/// The default drafter's path — `~/models/<dir_name>` — or null where HOME is not in the environment
/// (a minimal session): `enter`'s refusal then names what it tried. `allocPrint` into the caller's arena.
pub fn defaultDir(arena: Allocator, init: std.process.Init) Allocator.Error![]const u8 {
    const home = init.minimal.environ.getPosix("HOME") orelse return dir_name;
    return std.fmt.allocPrint(arena, "{s}/models/{s}", .{ home, dir_name });
}

/// `enter` from another machine's drafter: its `config.json` text and the checkpoint of its files'
/// headers (`safetensors.Checkpoint.fromHeads`), which `opened` takes.
pub fn given(init: std.process.Init, opened: *Opened, path: []const u8, config: []const u8, checkpoint: safetensors.Checkpoint, c: *const nemotron_config.Config, layers: usize) !bool {
    return shared.given(init, opened, path, config, checkpoint, c.backbone, layers);
}

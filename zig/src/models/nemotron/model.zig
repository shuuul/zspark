//! Nemotron-3.5-Transcribe-30B-A4B's backbone on the device (`model.py`, `cuda/weights.py`): the embeddings,
//! the layers `layers_block_type` names (Mamba-2, full attention, MoE), the final norm and the untied head,
//! plus the MTP block's own layers (`llm.mtp.layers.0` attention, `llm.mtp.layers.1` MoE), its fused input
//! projection and norms. The head drafts over `lm_head` itself: TensorFold's `draft_ids.txt` subset was built
//! for Nemotron-3.5-Lightning's tokenizer, whose ids 100 to 141 are Transcribe's `<spk:>`/`<ev:>`/`<sty:>`
//! tokens, so the list scores rows of the wrong tokens here (`AGENTS.md`).
//!
//! Every weight is read as the checkpoint stores it: BF16 for the norms, embeddings, router and head; NVFP4
//! or per-tensor FP8 for the projections and experts as `hf_quant_config.json`'s map says (`layout.zig`
//! already placed every tensor and read its shape; a tensor outside the manifest never reaches here).
//!
//! A layer's weights are a tagged union (`Block`): the residual RMSNorm's weight beside the mixer's own
//! (`mamba`'s projections are `linear.Linear`s of the checkpoint's FP8, its conv and SSM tensors widened as
//! `mamba.Mixer` asks; `attention` NoPE GQA's stacked qkv and o, `moe` the router and the routed plus shared
//! experts). The MTP block holds an attention and a MoE of the same types under `llm.mtp.` prefixes, read as
//! the quant map stores them (`model-mtp-fp8.safetensors`), plus `enorm`, `hnorm`, `eh_proj`, the block's two
//! layer norms and its final norm.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const config_file = @import("config.zig");
const layout_file = @import("layout.zig");
const mamba_file = @import("mamba.zig");
const attention_file = @import("attention.zig");
const moe_file = @import("moe.zig");
const linear = @import("../../weights/linear.zig");
const bf16 = @import("../../weights/bf16.zig");
const modelopt = @import("../../weights/modelopt.zig");
const experts_file = @import("../../weights/experts.zig");
const ranks_file = @import("ranks.zig");
const safetensors = @import("../../weights/safetensors.zig");

pub const Config = config_file.Config;
pub const Diagnostic = config_file.Diagnostic;
pub const Error = error{ MissingTensor, UnexpectedTensor, UnsupportedShape, Indivisible, InvalidCsf, NoMtpHead } || Allocator.Error || cuda.Error;

/// One backbone layer's residual norm and mixer weights, by `layers_block_type`.
pub const Block = union(config_file.LayerType) {
    /// `norm.weight`, the Mamba-2 mixer and its in/out projections.
    mamba: struct {
        norm: cuda.Buffer,
        in_proj: linear.Linear,
        mixer: mamba_file.Mixer,
        out_proj: linear.Linear,
    },
    /// `norm.weight` and the NoPE attention's weights (`attention.zig` holds qkv and o itself).
    attention: struct {
        norm: cuda.Buffer,
        mixer: attention_file.Attention,
    },
    /// `norm.weight` and the MoE block (`moe.zig` holds the router, the routed and the shared experts).
    moe: struct {
        norm: cuda.Buffer,
        mixer: moe_file.Moe,
    },

    /// The block's residual-norm weight, bf16 [hidden] on the device.
    pub fn norm(b: *const Block) cuda.View {
        return switch (b.*) {
            inline else => |*one| one.norm.view,
        };
    }

    pub fn deinit(b: *Block) void {
        switch (b.*) {
            .mamba => |*one| {
                one.in_proj.deinit();
                one.mixer.deinit();
                one.out_proj.deinit();
            },
            inline else => |*one| one.mixer.deinit(),
        }
        switch (b.*) {
            inline else => |*one| one.norm.deinit(),
        }
    }
};

/// The MTP block's weights (`llm.mtp.layers.*`), as TensorFold's `Mtp`.
pub const Mtp = struct {
    /// `layers.0.enorm`, `layers.0.hnorm`: the concat norms' weights over the embedding and the hidden row.
    enorm: cuda.Buffer,
    hnorm: cuda.Buffer,
    /// `layers.0.eh_proj`: [hidden, 2 * hidden] as stored (per-tensor FP8 on this build).
    eh_proj: linear.Linear,
    /// `layers.0.norm`, `layers.1.norm`.
    attn_norm: cuda.Buffer,
    moe_norm: cuda.Buffer,
    /// `layers.0.mixer.*`: the block's NoPE attention.
    attn: attention_file.Attention,
    /// `layers.1.mixer.*`: its MoE.
    moe: moe_file.Moe,
    /// `layers.1.final_layernorm`.
    final_norm: cuda.Buffer,

    pub fn deinit(m: *Mtp) void {
        m.enorm.deinit();
        m.hnorm.deinit();
        m.eh_proj.deinit();
        m.attn_norm.deinit();
        m.moe_norm.deinit();
        m.attn.deinit();
        m.moe.deinit();
        m.final_norm.deinit();
        m.* = undefined;
    }
};

/// The drafting head: `lm_head` itself, the whole vocabulary scored (TensorFold's subset list is
/// Lightning's; see the file comment). `count` the rows a draw's logits hold.
pub const DraftHead = struct {
    /// [count, hidden] bf16, `lm_head.weight` as stored.
    linear: bf16.Linear,
    count: usize,

    pub fn deinit(d: *DraftHead) void {
        d.linear.deinit();
        d.* = undefined;
    }
};

/// The whole backbone on the device, passed by value.
pub const Model = struct {
    gpa: Allocator,
    config: Config,
    /// `backbone.embeddings.weight` bf16 [vocab, hidden].
    embed: cuda.Buffer,
    blocks: []Block,
    /// `backbone.norm_f.weight` bf16 [hidden].
    norm_f: cuda.Buffer,
    /// `lm_head.weight` bf16 [vocab, hidden], untied.
    head: bf16.Linear,
    /// The MTP block; null where the config asks for none.
    mtp: ?Mtp = null,
    /// The drafting head over the draft ids; null with `mtp`.
    draft: ?DraftHead = null,
    /// The most rows one window runs (`draft.forward_max_rows`).
    max_rows: usize,
    /// The most rows a prompt chunk runs (what the scratch pools were sized for).
    prompt_rows: usize,
    /// The linears' shared staging pool; a model keeps it so its linears' pointers stay.
    scratch: *cuda.Scratch,

    /// `checkpoint`'s tensors uploaded (its config and layout already read), for windows of `max_rows` and
    /// prompt chunks of `prompt_rows`; `scratch` the pool the linears stage in, `staging` recycled over the
    /// layers' experts. Uploads on the caller's stream; synchronizes before returning.
    pub fn load(arena: Allocator, gpa: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic, staging: *experts_file.Staging, scratch: *cuda.Scratch, max_rows: usize, prompt_rows: usize) Error!Model {
        return loadShared(arena, gpa, c, checkpoint, diagnostic, staging, scratch, max_rows, prompt_rows, null);
    }

    /// `load` as `share`'s rank of two holds it (`ranks.zig`'s cut): this rank's half of every wide
    /// projection and of `lm_head`, the replicated tensors whole, the mixers' shapes halved. The rank's
    /// layers then run their share and the partials meet in `ranks_forward.zig`.
    pub fn loadShared(arena: Allocator, gpa: Allocator, c: *const Config, checkpoint: *const safetensors.Checkpoint, diagnostic: *Diagnostic, staging: *experts_file.Staging, scratch: *cuda.Scratch, max_rows: usize, prompt_rows: usize, share: ?ranks_file.Share) Error!Model {
        const b = c.backbone;
        if (share != null) try ranks_file.divisible(arena, b, diagnostic);
        const shapes = ranks_file.Shapes.of(b, share);
        const rows: cuda.Scratch.Rows = .{ .pool = scratch, .count = @max(max_rows, prompt_rows) };
        var m: Model = .{
            .gpa = gpa,
            .config = c.*,
            .embed = try tensor(checkpoint, "backbone.embeddings.weight"),
            .blocks = try gpa.alloc(Block, b.types.len),
            .norm_f = try tensor(checkpoint, "backbone.norm_f.weight"),
            .head = try headLinear(arena, gpa, checkpoint, c, diagnostic, rows, share),
            .max_rows = max_rows,
            .prompt_rows = prompt_rows,
            .scratch = scratch,
        };
        var made: usize = 0;
        errdefer {
            for (m.blocks[0..made]) |*blk| blk.deinit();
            gpa.free(m.blocks);
            m.head.deinit();
            m.norm_f.deinit();
            m.embed.deinit();
        }
        for (m.blocks, 0..) |*blk, i| {
            blk.* = try block(arena, gpa, checkpoint, c, diagnostic, staging, rows, shapes, share, i);
            made += 1;
        }
        if (b.mtp_layers != 0) {
            m.mtp = try loadMtp(arena, gpa, checkpoint, c, diagnostic, staging, rows, shapes, share);
            errdefer m.mtp.?.deinit();
            m.draft = try draftHead(arena, gpa, checkpoint, c, diagnostic, rows, share);
        }
        try cuda.synchronize();
        return m;
    }

    pub fn deinit(m: *Model) void {
        for (m.blocks) |*blk| blk.deinit();
        m.gpa.free(m.blocks);
        m.head.deinit();
        m.norm_f.deinit();
        m.embed.deinit();
        if (m.mtp) |*one| one.deinit();
        if (m.draft) |*one| one.deinit();
        m.* = undefined;
    }

    /// A backbone layer at index `i`: its norm and `types[i]`'s mixer under `backbone.layers.N.`.
    fn block(arena: Allocator, gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, diagnostic: *Diagnostic, staging: *experts_file.Staging, rows: cuda.Scratch.Rows, shapes: ranks_file.Shapes, share: ?ranks_file.Share, i: usize) Error!Block {
        const b = c.backbone;
        var name: [160]u8 = undefined;
        var norm = try tensor(checkpoint, std.fmt.bufPrint(&name, layout_file.backbone_prefix ++ "{d}.norm.weight", .{i}) catch unreachable);
        errdefer norm.deinit();
        switch (b.types[i]) {
            .mamba => {
                var in_proj = try mambaLinear(arena, gpa, checkpoint, c, diagnostic, rows, shapes.mamba, share, i, "in_proj", b.inWidth(), b.hidden);
                errdefer in_proj.deinit();
                var out_proj = try mambaLinear(arena, gpa, checkpoint, c, diagnostic, rows, shapes.mamba, share, i, "out_proj", b.hidden, b.mambaInner());
                errdefer out_proj.deinit();
                const mixer = try mambaMixer(arena, gpa, checkpoint, c, shapes.mamba, share, i);
                return .{ .mamba = .{ .norm = norm, .in_proj = in_proj, .mixer = mixer, .out_proj = out_proj } };
            },
            .attention => {
                const mixer = try attention_file.Attention.init(arena, gpa, b, layout_file.backbone_prefix, i, checkpoint, c.quant, rows, diagnostic, shapes.attention, share);
                return .{ .attention = .{ .norm = norm, .mixer = mixer } };
            },
            .moe => {
                var buf: [160]u8 = undefined;
                const prefix = std.fmt.bufPrint(&buf, layout_file.backbone_prefix ++ "{d}.mixer", .{i}) catch unreachable;
                const mixer = try moe_file.Moe.load(arena, gpa, staging, checkpoint, shapes.moe, prefix, c.quant, diagnostic, share);
                return .{ .moe = .{ .norm = norm, .mixer = mixer } };
            },
        }
    }

    /// A Mamba layer's projection, `backbone.layers.i.mixer.leaf`, [n, k] as stored.
    /// A Mamba layer's projection, `backbone.layers.i.mixer.leaf`, [n, k] as stored; `share`'s rows of
    /// `in_proj` (the segmented cut `ranks.mambaInRows` lays out) or columns of `out_proj`.
    fn mambaLinear(arena: Allocator, gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, diagnostic: *Diagnostic, rows: cuda.Scratch.Rows, s: mamba_file.Shape, share: ?ranks_file.Share, i: usize, leaf: []const u8, n: usize, k: usize) Error!linear.Linear {
        var name: [160]u8 = undefined;
        const module = std.fmt.bufPrint(&name, layout_file.backbone_prefix ++ "{d}.mixer.{s}", .{ i, leaf }) catch unreachable;
        var stored = try modelopt.read(arena, checkpoint, module, c.quant.find(module), n, k, diagnostic);
        if (share) |sh| {
            if (std.mem.eql(u8, leaf, "in_proj")) {
                const idx = try arena.alloc(usize, s.proj());
                stored = try ranks_file.linearRows(arena, stored, ranks_file.mambaInRows(c.backbone, s, sh, idx), diagnostic);
            } else stored = try modelopt.cut(arena, stored, .col, sh.rank, ranks_file.world, diagnostic);
        }
        if (stored.scheme == .fp8) return linear.Linear.storedStaticFp8(gpa, stored, rows);
        return linear.Linear.stored(arena, gpa, &.{stored}, rows, diagnostic);
    }

    /// The Mamba mixer's own tensors widened (`mamba.Mixer.Host`), under `backbone.layers.i.mixer.`; a rank
    /// keeps its heads' and groups' slices (`ranks.mambaConvCols`, the head vectors' halves).
    fn mambaMixer(arena: Allocator, gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, s: mamba_file.Shape, share: ?ranks_file.Share, i: usize) Error!mamba_file.Mixer {
        const b = c.backbone;
        const full: mamba_file.Shape = .of(b);
        var head_rows: [128]usize = undefined;
        var norm_rows: [4096]usize = undefined;
        var conv_cols: [8704]usize = undefined;
        const heads = if (share) |sh| ranks_file.mambaHeadRows(s, sh, &head_rows) else null;
        const conv = if (share) |sh| ranks_file.mambaConvCols(b, s, sh, conv_cols[0..s.cd()]) else null;
        const norm = if (share) |sh| ranks_file.mambaNormRows(s, sh, &norm_rows) else null;
        var names: [6][160]u8 = undefined;
        const host: mamba_file.Mixer.Host = .{
            .a_log = try bf16Part(arena, checkpoint, std.fmt.bufPrint(&names[0], layout_file.backbone_prefix ++ "{d}.mixer.A_log", .{i}) catch unreachable, full.heads, heads),
            .d = try bf16Part(arena, checkpoint, std.fmt.bufPrint(&names[1], layout_file.backbone_prefix ++ "{d}.mixer.D", .{i}) catch unreachable, full.heads, heads),
            .dt_bias = try bf16Part(arena, checkpoint, std.fmt.bufPrint(&names[2], layout_file.backbone_prefix ++ "{d}.mixer.dt_bias", .{i}) catch unreachable, full.heads, heads),
            .conv_w = try bf16Conv(arena, checkpoint, std.fmt.bufPrint(&names[3], layout_file.backbone_prefix ++ "{d}.mixer.conv1d.weight", .{i}) catch unreachable, full.cd(), conv),
            .conv_b = try bf16Part(arena, checkpoint, std.fmt.bufPrint(&names[4], layout_file.backbone_prefix ++ "{d}.mixer.conv1d.bias", .{i}) catch unreachable, full.cd(), conv),
            .norm = try bf16Part(arena, checkpoint, std.fmt.bufPrint(&names[5], layout_file.backbone_prefix ++ "{d}.mixer.norm.weight", .{i}) catch unreachable, full.xd, norm),
        };
        return mamba_file.Mixer.init(gpa, s, host, @floatCast(b.dt_min), @floatCast(b.dt_max), b.eps) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
            error.UnexpectedTensor => error.UnexpectedTensor,
            else => error.CudaFailure,
        };
    }

    /// The MTP block (`llm.mtp.layers.*`): `mtp_layers_block_type` attention then MoE. `eh_proj` and the
    /// norms stay whole on both ranks; the attention and the MoE split as the backbone's do.
    fn loadMtp(arena: Allocator, gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, diagnostic: *Diagnostic, staging: *experts_file.Staging, rows: cuda.Scratch.Rows, shapes: ranks_file.Shapes, share: ?ranks_file.Share) Error!Mtp {
        const b = c.backbone;
        var m: Mtp = .{
            .enorm = try tensor(checkpoint, "llm.mtp.layers.0.enorm.weight"),
            .hnorm = try tensor(checkpoint, "llm.mtp.layers.0.hnorm.weight"),
            .eh_proj = undefined,
            .attn_norm = try tensor(checkpoint, "llm.mtp.layers.0.norm.weight"),
            .moe_norm = try tensor(checkpoint, "llm.mtp.layers.1.norm.weight"),
            .attn = undefined,
            .moe = undefined,
            .final_norm = try tensor(checkpoint, "llm.mtp.layers.1.final_layernorm.weight"),
        };
        errdefer {
            m.enorm.deinit();
            m.hnorm.deinit();
            m.attn_norm.deinit();
            m.moe_norm.deinit();
            m.final_norm.deinit();
        }
        {
            const stored = try modelopt.read(arena, checkpoint, "llm.mtp.layers.0.eh_proj", c.quant.find("llm.mtp.layers.0.eh_proj"), b.hidden, 2 * b.hidden, diagnostic);
            m.eh_proj = try linear.Linear.stored(arena, gpa, &.{stored}, rows, diagnostic);
        }
        errdefer m.eh_proj.deinit();
        m.attn = try attention_file.Attention.init(arena, gpa, b, layout_file.mtp_prefix, 0, checkpoint, c.quant, rows, diagnostic, shapes.attention, share);
        errdefer m.attn.deinit();
        m.moe = try moe_file.Moe.load(arena, gpa, staging, checkpoint, shapes.moe, "llm.mtp.layers.1.mixer", c.quant, diagnostic, share);
        return m;
    }

    /// `lm_head.weight` as `bf16.Linear`: the head is BF16 on this checkpoint (the quant map names it not).
    /// A rank keeps its half of the vocabulary's rows.
    fn headLinear(arena: Allocator, gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, diagnostic: *Diagnostic, rows: cuda.Scratch.Rows, share: ?ranks_file.Share) Error!bf16.Linear {
        var stored = try modelopt.read(arena, checkpoint, "lm_head", c.quant.find("lm_head"), c.backbone.vocab, c.backbone.hidden, diagnostic);
        if (share) |sh| stored = try modelopt.cut(arena, stored, .vocab, sh.rank, ranks_file.world, diagnostic);
        var l: linear.Linear = try .stored(arena, gpa, &.{stored}, rows, diagnostic);
        return switch (l) {
            .bf16 => |h| h,
            else => {
                diagnostic.set(arena, "lm_head.weight is stored quantized; zspark's Nemotron reads a BF16 head", .{});
                l.deinit();
                return error.UnexpectedTensor;
            },
        };
    }

    /// The draft head is `lm_head` itself: scoring it keeps the draw over the checkpoint's own ids, which
    /// the shipped subset list (Lightning's vocabulary) does not — its ids 100 to 141 are Transcribe's
    /// `<spk:>`/`<ev:>`/`<sty:>` tokens.
    /// The draft head is `lm_head` itself (a rank's the same vocabulary half). Scoring it keeps the draw
    /// over the checkpoint's own ids, which the shipped subset list (Lightning's vocabulary) does not — its
    /// ids 100 to 141 are Transcribe's `<spk:>`/`<ev:>`/`<sty:>` tokens.
    fn draftHead(arena: Allocator, gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, diagnostic: *Diagnostic, rows: cuda.Scratch.Rows, share: ?ranks_file.Share) Error!DraftHead {
        const l = try headLinear(arena, gpa, checkpoint, c, diagnostic, rows, share);
        return .{ .linear = l, .count = l.n };
    }

    /// A BF16 vector's u16s as stored, or `rows`'s of them where a rank slices it (null: whole).
    fn bf16Part(arena: Allocator, checkpoint: *const safetensors.Checkpoint, name: []const u8, total: usize, rows: ?[]const usize) Error![]align(1) const u16 {
        const t = checkpoint.get(name) orelse return error.MissingTensor;
        if (t.dtype != .BF16 or t.count() != total) return error.UnexpectedTensor;
        if (rows) |idx| {
            const cut = try ranks_file.bf16Rows(arena, t, idx);
            return std.mem.bytesAsSlice(u16, cut.data);
        }
        return std.mem.bytesAsSlice(u16, t.data);
    }

    /// `conv1d.weight` [cd, 1, 4] as the mixer's [cd, 4] u16s, `cols` of them where a rank slices it.
    fn bf16Conv(arena: Allocator, checkpoint: *const safetensors.Checkpoint, name: []const u8, total: usize, cols: ?[]const usize) Error![]align(1) const u16 {
        const t = checkpoint.get(name) orelse return error.MissingTensor;
        if (t.dtype != .BF16 or t.count() != 4 * total) return error.UnexpectedTensor;
        const data = if (cols) |idx| blk: {
            const inner = t.data.len / total;
            const flat = try arena.alignedAlloc(u8, .@"64", idx.len * inner);
            for (idx, 0..) |r, at| @memcpy(flat[at * inner ..][0..inner], t.data[r * inner ..][0..inner]);
            break :blk flat;
        } else t.data;
        return std.mem.bytesAsSlice(u16, data);
    }

    /// A BF16 tensor uploaded whole, by name.
    fn tensor(checkpoint: *const safetensors.Checkpoint, name: []const u8) Error!cuda.Buffer {
        const t = checkpoint.get(name) orelse return error.MissingTensor;
        if (t.dtype != .BF16) return error.UnexpectedTensor;
        var dims: [3]usize = undefined;
        for (t.shape, 0..) |dim, i| dims[i] = @intCast(dim);
        var b: cuda.Buffer = try .init(.bf16, dims[0..t.shape.len]);
        errdefer b.deinit();
        try b.upload(t.data);
        return b;
    }
};

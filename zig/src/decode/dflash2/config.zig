//! DFlash2's config.json (`architectures` `DFlash2DraftModel`, `model_type` `qwen3`) as
//! `families/glm5_next/cuda/dflash2.py:Drafter.__init__` reads it, and what this port refuses. Host only.
//! GLM-5.3-Flash and Qwen3.8 Flash Next both use this schema; `Window` is how `sliding_window` becomes
//! `Config.window` (GLM exclusive, Qwen stored, Nemotron full: a null window, the drafter's layers all
//! `full_attention`, sees the whole context as vLLM's `_resolve_layer_attention` resolves it). The drafter's
//! weights are the user's own copy on disk: nothing here writes them.
//!
//! Capture of taps (the streams' mean at completed layers) stays in each target's forward; this file names no
//! capture site. Qwen's published sizes are `models/qwen4/dflash2/rig.zig`'s `published`, Nemotron's
//! `models/nemotron/dflash2.zig`'s.

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const share = @import("../../weights/share.zig");

/// Why a config was refused, in the drafter's own words.
pub const Diagnostic = struct {
    message: []const u8 = "",

    pub fn set(self: *Diagnostic, arena: Allocator, comptime format: []const u8, args: anytype) void {
        self.message = std.fmt.allocPrint(arena, format, args) catch "out of memory";
    }
};

pub const Error = error{ InvalidConfig, UnsupportedModel } || Allocator.Error;

/// How `sliding_window` becomes `Config.window`.
pub const Window = enum {
    /// GLM: a context key at k is seen from a query at q when q - k is at most sliding_window - 1.
    exclusive,
    /// Qwen Flash Next DFlash2: the stored sliding_window (2048), not less one.
    stored,
    /// Nemotron Transcribe DFlash2: no window (`sliding_window` null, every layer `full_attention`): the
    /// drafter's attention sees the whole context (vLLM's `_resolve_layer_attention`).
    full,
};

/// The ranks its heads and its MLP are split over: the main model's.
pub const world = share.ranks;

/// The name config.json's `model_type` carries, which is not the drafter's alone: `architectures` says it.
pub const model_types = [_][]const u8{"qwen3"};
pub const architecture = "DFlash2DraftModel";

/// A tile of the attention's keys; the ring is whole tiles (`geometry.py:draft_ring_rows`).
pub const tile = 64;

/// The drafter. Slices live in the arena `parse` was given.
pub const Config = struct {
    hidden: usize,
    vocab: usize,
    layers: usize,
    /// Query heads, key and value heads (a KV head serves `heads / kv_heads` query heads), a head's width.
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    eps: f64,
    theta: f64,
    /// How far back a query sees a context key: GLM stores `sliding_window` less one (`Window.exclusive`);
    /// Qwen Flash Next DFlash2 stores the window as written (`Window.stored`); Nemotron's null is no window
    /// at all (`Window.full`), the whole context seen.
    window: ?usize,
    /// False: a block's rows see each other both ways.
    causal: bool,

    /// The block: the pending token and `block - 1` masks, each a draft.
    block: usize,
    mask: u32,
    /// The grouped convolution: channels a group (its taps are two, refused otherwise).
    group: usize,
    /// The selector: a projected row's width, and the candidates a rank's half head gives a draft.
    selector_rank: usize,
    top_k: usize,
    /// The main model's layers whose output is a tap, in the order `fc` reads them, and how many layers the
    /// drafter was trained beside.
    taps: []const u32,
    target_layers: usize,

    /// A rank's query heads, KV heads and MLP width.
    pub fn rankHeads(self: Config) usize {
        return self.heads / world;
    }

    pub fn rankKvHeads(self: Config) usize {
        return self.kv_heads / world;
    }

    pub fn rankWidth(self: Config) usize {
        return self.intermediate / world;
    }

    /// `fc`'s input: the taps side by side.
    pub fn tapWidth(self: Config) usize {
        return self.taps.len * self.hidden;
    }

    /// A kernel projection's outputs: two branches of two taps over the groups.
    pub fn kernelWidth(self: Config) usize {
        return 2 * 2 * (self.hidden / self.group);
    }

    /// The context's rows as a ring: the window, a block and a tile, in whole tiles
    /// (`geometry.py:draft_ring_rows`). No window, no ring.
    pub fn ringRows(self: Config) usize {
        const window = self.window orelse return 0;
        return (window + self.block + tile + tile - 1) / tile * tile;
    }

    /// A rank's K and V of `rows` context rows over the layers, bf16.
    pub fn contextBytes(self: Config, rows: usize) u64 {
        return @as(u64, 2) * self.layers * self.rankKvHeads() * rows * self.head_dim * 2;
    }

    /// The window the kernels launch with: the config's, or `cap` rows where there is none (a query sees
    /// every key behind its block anyway).
    pub fn seen(self: Config, cap: usize) usize {
        return self.window orelse cap;
    }

    /// Reads `dir`'s config.json with GLM's exclusive window.
    pub fn read(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        return parse(arena, text, diagnostic);
    }

    /// Reads `dir`'s config.json with the stored sliding_window (Qwen Flash Next DFlash2).
    pub fn readStored(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        return parseStored(arena, text, diagnostic);
    }

    /// Reads `dir`'s config.json with no window (Nemotron Transcribe DFlash2, `sliding_window` null).
    pub fn readFull(arena: Allocator, io: Io, dir: Io.Dir, diagnostic: *Diagnostic) (Error || Io.Dir.ReadFileAllocError)!Config {
        const text = try dir.readFileAlloc(io, "config.json", arena, .limited(1 << 26));
        return parseFull(arena, text, diagnostic);
    }

    /// `text` is config.json, GLM's exclusive window. The result lives in `arena`.
    pub fn parse(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Config {
        const c = try parseWindow(arena, text, diagnostic, .exclusive);
        // GLM's q projection is written [hidden, heads * head_dim]: its inner width IS hidden.
        // Qwen's GQA has q_inner = heads * head_dim = 4096 over hidden 2560, so parseStored does
        // not decide this; the layouts' `Dim.q_inner` carries either.
        if (c.heads * c.head_dim != c.hidden)
            return (Parser{ .arena = arena, .diagnostic = diagnostic }).fail(error.InvalidConfig, "config.json: {d} heads of {d} make an inner width of {d}, not hidden {d}", .{ c.heads, c.head_dim, c.heads * c.head_dim, c.hidden });
        return c;
    }

    /// `text` is config.json, Qwen's stored sliding_window. The result lives in `arena`.
    pub fn parseStored(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Config {
        return parseWindow(arena, text, diagnostic, .stored);
    }

    /// `text` is config.json, Nemotron's null sliding_window. The result lives in `arena`.
    pub fn parseFull(arena: Allocator, text: []const u8, diagnostic: *Diagnostic) Error!Config {
        return parseWindow(arena, text, diagnostic, .full);
    }

    /// Why this drafter does not go with a main model of `hidden`, `vocab` and `layers`; null when it does.
    pub fn mismatch(self: Config, arena: Allocator, hidden: usize, vocab: usize, layers: usize) Allocator.Error!?[]const u8 {
        if (self.hidden != hidden or self.vocab != vocab or self.target_layers != layers)
            return try std.fmt.allocPrint(arena, "the drafter is for a model of hidden {d}, vocabulary {d} and {d} layers; this one has {d}, {d} and {d}", .{ self.hidden, self.vocab, self.target_layers, hidden, vocab, layers });
        return null;
    }
};

fn parseWindow(arena: Allocator, text: []const u8, diagnostic: *Diagnostic, how: Window) Error!Config {
    const p: Parser = .{ .arena = arena, .diagnostic = diagnostic };
    const root = json.parseFromSliceLeaky(json.Value, arena, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return p.fail(error.InvalidConfig, "config.json is not valid JSON ({t})", .{err}),
    };
    if (root != .object) return p.fail(error.InvalidConfig, "config.json is not a JSON object", .{});
    const raw = root.object;
    if (!is(raw)) return p.fail(error.UnsupportedModel, "config.json's architectures does not name {s}: not a DFlash2 drafter", .{architecture});
    const d = try p.object(raw, "dflash_config");
    const rope = try p.object(raw, "rope_parameters");

    var c: Config = .{
        .hidden = try p.uint(raw, "hidden_size"),
        .vocab = try p.uint(raw, "vocab_size"),
        .layers = try p.uint(raw, "num_hidden_layers"),
        .heads = try p.uint(raw, "num_attention_heads"),
        .kv_heads = try p.uint(raw, "num_key_value_heads"),
        .head_dim = try p.uint(raw, "head_dim"),
        .intermediate = try p.uint(raw, "intermediate_size"),
        .eps = try p.number(raw, "rms_norm_eps"),
        .theta = try p.number(rope, "rope_theta"),
        .window = try windowOf(p, raw, how),
        // true when absent, as `Drafter.__init__` reads it
        .causal = if (raw.get("is_causal") == null) true else try p.flag(raw, "is_causal"),
        .block = try p.uint(d, "block_size"),
        .mask = @intCast(try p.uint(d, "mask_token_id")),
        .group = try p.uint(d, "conv_group_size"),
        .selector_rank = try p.uint(d, "selector_rank"),
        .top_k = try p.uint(d, "selector_top_k"),
        .taps = try p.ids(d, "target_layer_ids"),
        .target_layers = try p.uint(raw, "num_target_layers"),
    };
    if (c.window) |window| {
        if (window == 0) return p.fail(error.InvalidConfig, "config.json: sliding_window is 0", .{});
        if (how == .exclusive) c.window = window - 1;
    }

    // What the kernels are written for; each is a number of the config this port reads as a fact.
    if (try p.uint(d, "conv_kernel_size") != 2)
        return p.fail(error.UnsupportedModel, "config.json: conv_kernel_size is {d}; the drafter's convolution is ported for two taps", .{try p.uint(d, "conv_kernel_size")});
    if (c.head_dim != 128)
        return p.fail(error.UnsupportedModel, "config.json: head_dim is {d}; the drafter's attention is written for heads of 128", .{c.head_dim});
    if (c.causal)
        return p.fail(error.UnsupportedModel, "config.json: is_causal is true; the drafter's block pass is written for rows that see each other both ways", .{});
    if (c.heads == 0 or c.kv_heads == 0 or c.heads % world != 0 or c.kv_heads % world != 0 or c.intermediate % world != 0)
        return p.fail(error.InvalidConfig, "config.json: {d} heads, {d} KV heads and an MLP of {d}: two ranks each take half", .{ c.heads, c.kv_heads, c.intermediate });
    // Qwen Flash Next GQA: q_inner is heads*head_dim (4096) while hidden is 2560. GLM's happen to match.
    if (c.heads % c.kv_heads != 0)
        return p.fail(error.InvalidConfig, "config.json: {d} heads of {d} over {d} KV heads, hidden {d}", .{ c.heads, c.head_dim, c.kv_heads, c.hidden });
    if (c.group == 0 or c.hidden % c.group != 0)
        return p.fail(error.InvalidConfig, "config.json: conv_group_size {d} does not divide hidden_size {d}", .{ c.group, c.hidden });
    if (c.block < 2 or c.block > tile)
        return p.fail(error.InvalidConfig, "config.json: block_size is {d}; a block is the pending token and 1 to {d} drafts", .{ c.block, tile - 1 });
    if (c.top_k == 0 or c.top_k > c.vocab / world)
        return p.fail(error.InvalidConfig, "config.json: selector_top_k is {d}", .{c.top_k});
    if (c.layers == 0 or c.taps.len == 0)
        return p.fail(error.InvalidConfig, "config.json: {d} layers and {d} taps", .{ c.layers, c.taps.len });
    for (c.taps, 0..) |tap, at| {
        if (tap >= c.target_layers or (at > 0 and tap <= c.taps[at - 1]))
            return p.fail(error.InvalidConfig, "config.json: target_layer_ids are not rising layers of {d}", .{c.target_layers});
    }
    if (c.mask >= c.vocab)
        return p.fail(error.InvalidConfig, "config.json: mask_token_id {d} is not in the vocabulary of {d}", .{ c.mask, c.vocab });
    return c;
}

/// `sliding_window` as `how` reads it: a count for `exclusive` and `stored`, null for `full` (its presence a
/// count is refused, so a windowed drafter cannot slip through Nemotron's reader).
fn windowOf(p: Parser, raw: json.ObjectMap, how: Window) Error!?usize {
    const value = raw.get("sliding_window");
    if (how == .full) {
        if (value == null) return p.fail(error.InvalidConfig, "config.json has no sliding_window", .{});
        if (value.? == .null) return null;
        return p.fail(error.UnsupportedModel, "config.json: sliding_window is set; Nemotron's drafter keeps none (its layers are all full_attention)", .{});
    }
    if (value == null) return p.fail(error.InvalidConfig, "config.json has no sliding_window", .{});
    if (value.? == .integer and value.?.integer >= 0) return @intCast(value.?.integer);
    return p.fail(error.InvalidConfig, "config.json: sliding_window is not a count", .{});
}

/// Whether a config.json's root names the drafter's architecture.
pub fn is(raw: json.ObjectMap) bool {
    const list = raw.get("architectures") orelse return false;
    if (list != .array) return false;
    for (list.array.items) |item| if (item == .string and std.mem.eql(u8, item.string, architecture)) return true;
    return false;
}

/// config.json's fields by kind, each refusal a line in the diagnostic.
const Parser = struct {
    arena: Allocator,
    diagnostic: *Diagnostic,

    fn fail(p: Parser, err: Error, comptime format: []const u8, args: anytype) Error {
        p.diagnostic.set(p.arena, format, args);
        return err;
    }

    fn field(p: Parser, fields: json.ObjectMap, key: []const u8) Error!json.Value {
        return fields.get(key) orelse p.fail(error.InvalidConfig, "config.json has no {s}", .{key});
    }

    fn object(p: Parser, fields: json.ObjectMap, key: []const u8) Error!json.ObjectMap {
        const value = try p.field(fields, key);
        return if (value == .object) value.object else p.fail(error.InvalidConfig, "config.json: {s} is not an object", .{key});
    }

    fn uint(p: Parser, fields: json.ObjectMap, key: []const u8) Error!usize {
        const value = try p.field(fields, key);
        if (value == .integer and value.integer >= 0) return @intCast(value.integer);
        return p.fail(error.InvalidConfig, "config.json: {s} is not a count", .{key});
    }

    fn number(p: Parser, fields: json.ObjectMap, key: []const u8) Error!f64 {
        return switch (try p.field(fields, key)) {
            .integer => |v| @floatFromInt(v),
            .float => |v| v,
            else => p.fail(error.InvalidConfig, "config.json: {s} is not a number", .{key}),
        };
    }

    fn flag(p: Parser, fields: json.ObjectMap, key: []const u8) Error!bool {
        const value = try p.field(fields, key);
        return if (value == .bool) value.bool else p.fail(error.InvalidConfig, "config.json: {s} is not true or false", .{key});
    }

    /// One id or a list of them.
    fn ids(p: Parser, fields: json.ObjectMap, key: []const u8) Error![]const u32 {
        const value = try p.field(fields, key);
        const items: []const json.Value = if (value == .array) value.array.items else &.{value};
        const out = try p.arena.alloc(u32, items.len);
        for (out, items) |*id, item| {
            if (item != .integer or item.integer < 0) return p.fail(error.InvalidConfig, "config.json: {s} is not a token id or a list of them", .{key});
            id.* = @intCast(item.integer);
        }
        return out;
    }
};

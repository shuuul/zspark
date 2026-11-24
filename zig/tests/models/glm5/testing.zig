//! What the device tests of this directory share: buffers from host slices and back, random bf16 rows, the
//! random model in loopback (`Fixture`, `Loop`), a reply decoded on it (`reply`) and by serial steps
//! (`serialReply`), and the opener of the FP4-Local pack (`openPack`). What needs no device is `fixtures.zig`'s.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const numeric_fixture = @import("../../weights/numeric_fixture.zig");
const gpu = @import("../../cuda/testing.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const config_file = @import("../../../src/models/glm5/layers/config.zig");
const draft_file = @import("../../../src/models/glm5/draft.zig");
const loopback = @import("../../../src/models/glm5/layers/loopback.zig");
const random_file = @import("../../../src/models/glm5/checks/random.zig");
const forward = @import("../../../src/models/glm5/layers/forward.zig");
const kda = @import("../../../src/models/glm5/layers/kda.zig");

const fx = random_file.fx;
const Source = random_file.Source;

/// One form for every window of the random model, a step's: bf16 rows (`forward.bf16_rows` past the model's
/// widest window, spec 115 giving a longer one the quantized forms) and KDA's recurrent chain (`kda.chunked_windows`
/// off; the chunked one is its within a bound, `kda_test`). What a test holding a window to its serial steps takes;
/// `restore` after it.
pub const OneForm = struct {
    rows: usize,

    pub fn restore(self: OneForm) void {
        forward.bf16_rows = self.rows;
        kda.chunked_windows = true;
    }
};

pub fn oneForm() OneForm {
    const was: OneForm = .{ .rows = forward.bf16_rows };
    forward.bf16_rows = fx.max_rows;
    kda.chunked_windows = false;
    return was;
}

pub const randomBf16 = numeric_fixture.randomBf16;

pub fn randomF32(random: std.Random, out: []f32, spread: f32) void {
    for (out) |*one| one.* = random.floatNorm(f32) * spread;
}

pub const device = gpu.device;

pub const fetch = gpu.fetch;

/// bf16 values a rounding apart at most: `exp` differs between the host and the device.
pub fn expectClose(want_bits: u16, got_bits: u16) !void {
    const want = numeric.f32FromBf16(want_bits);
    try std.testing.expectApproxEqAbs(want, numeric.f32FromBf16(got_bits), @abs(want) / 64.0 + 1e-30);
}

/// The config and the files of the pack a test on `ZSPARK_GLM_FP4_LOCAL` reads; `close` the checkpoint after.
pub fn openPack(gpa: Allocator, arena: Allocator, path: []const u8) !struct { config_file.Config, safetensors.Checkpoint } {
    const io = std.testing.io;
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.read(arena, io, dir, &diagnostic);
    return .{ c, try .open(gpa, io, .cwd(), path) };
}

/// `count` tokens after `prompt` by one-row steps, each token keyed by its place in the sequence, written
/// without the decoder. `vocab` is the whole vocabulary.
pub fn serialReply(gpa: Allocator, m: loopback.Model, vocab: usize, prompt: []const i32, count: usize, settings: sampling.Sampling) ![]i32 {
    var state: loopback.State = try .init(m, prompt.len + count);
    defer state.deinit();
    var b: loopback.Buffers = try .init(m, prompt.len + count);
    defer b.deinit();
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    for (prompt) |token| {
        try loopback.forward(m, &state, &b, &.{token});
        try loopback.commit(m, &state, b, &.{token}, 1);
    }
    for (out, 0..) |*token, made| {
        try loopback.logitsRow(b, 0, logits);
        token.* = try sampler.choose(logits, prompt.len + made, settings);
        try loopback.forward(m, &state, &b, &.{token.*});
        try loopback.commit(m, &state, b, &.{token.*}, 1);
    }
    return out;
}

/// The random model's tensors and a sequence of ids; `loopback` loads both shares of it.
pub const Fixture = struct {
    arena_state: std.heap.ArenaAllocator,
    pool: cuda.Scratch,
    c: config_file.Config,
    source: Source,
    tokens: []i32,

    pub fn init(self: *Fixture, seed: u64, total: usize) !void {
        self.arena_state = .init(std.testing.allocator);
        const arena = self.arena_state.allocator();
        self.pool = .init(std.testing.allocator);
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        self.c = try fx.config(arena);
        self.source = .empty;
        try fx.putModel(arena, random, &self.source);
        self.tokens = try arena.alloc(i32, total);
        for (self.tokens) |*id| id.* = random.intRangeLessThan(i32, 0, fx.vocab);
    }

    pub fn deinit(self: *Fixture) void {
        self.pool.deinit();
        self.arena_state.deinit();
    }

    pub fn rows(self: *Fixture) cuda.Scratch.Rows {
        return .{ .pool = &self.pool, .count = fx.max_rows };
    }

    /// Both shares on the device, with a sequence of up to `capacity` positions.
    pub fn loop(self: *Fixture, capacity: usize) !Loop {
        var m: loopback.Model = try .load(self.arena_state.allocator(), std.testing.allocator, self.source, &self.c, self.rows());
        errdefer m.deinit();
        var state: loopback.State = try .init(m, capacity);
        errdefer state.deinit();
        return .{ .m = m, .state = state, .b = try .init(m, capacity) };
    }
};

pub const Loop = struct {
    m: loopback.Model,
    state: loopback.State,
    b: loopback.Buffers,

    pub fn deinit(self: *Loop) void {
        self.b.deinit();
        self.state.deinit();
        self.m.deinit();
    }
};

pub const Reply = struct { tokens: []i32, stats: draft_file.Stats };

/// `count` tokens after `prompt` from an empty sequence, by the decoder: the prompt as `Decoder.prompt` cuts it
/// (`drafter` absorbs every window), then rounds, drafted where a drafter is given.
pub fn reply(gpa: Allocator, loop: *Loop, prompt: []const i32, count: usize, settings: sampling.Sampling, eos: []const u32, drafter: ?draft_file.Drafter, graphs: ?*loopback.Graphs) !Reply {
    try loop.state.reset();
    const vocab = loop.m.ranks[0].head.outputs() * loop.m.ranks.len;
    var sampler: sampling.Sampler = try .init(gpa, vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, vocab);
    defer gpa.free(logits);
    var decoder: draft_file.Decoder = .{ .m = loop.m, .state = &loop.state, .b = &loop.b, .graphs = graphs, .sampler = &sampler, .settings = settings, .eos = eos, .logits = logits };

    const out = try gpa.alloc(i32, count);
    errdefer gpa.free(out);
    var chosen: [draft_file.forward_max_rows]i32 = undefined;
    var first: [draft_file.forward_max_rows]i32 = undefined;
    const rows = try decoder.prompt(drafter, prompt, &chosen);
    @memcpy(first[0..rows], chosen[0..rows]);
    out[0] = chosen[rows - 1];
    var made: usize = 1;
    var drafts: [draft_file.forward_max_rows]i32 = undefined;
    var kept: []const i32 = first[0..rows];
    while (made < count and std.mem.indexOfScalar(u32, eos, @intCast(out[made - 1])) == null) {
        const n = if (drafter) |d| try decoder.draft(d, kept, count - made - 1, &drafts) else 0;
        const keep = try decoder.round(out[made - 1], drafts[0..n], &chosen);
        @memcpy(out[made..][0..keep], chosen[0..keep]);
        made += keep;
        kept = out[made - keep .. made];
    }
    return .{ .tokens = try gpa.realloc(out, made), .stats = decoder.stats };
}

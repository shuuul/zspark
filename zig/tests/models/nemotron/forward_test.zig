//! `model.zig` + `forward.zig` over `fixture.zig`'s four-layer synthetic checkpoint (mamba, moe, attention,
//! moe, with the MTP block): the model loads; a prompt run whole, in chunks and as serial steps gives the
//! same logits and the same committed state bit for bit; a verify window rolled back leaves the kept rows'
//! state as a serial run's (the row rule, end to end).

const std = @import("std");
const cuda = @import("../../../src/cuda/cuda.zig");
const gpu = @import("../../cuda/testing.zig");
const config_file = @import("../../../src/models/nemotron/config.zig");
const model_file = @import("../../../src/models/nemotron/model.zig");
const forward_file = @import("../../../src/models/nemotron/forward.zig");
const state_file = @import("../../../src/models/nemotron/state.zig");
const fixture = @import("fixture.zig");
const experts_file = @import("../../../src/weights/experts.zig");
const draft_file = @import("../../../src/models/nemotron/draft.zig");
const mtp_file = @import("../../../src/models/nemotron/mtp.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");

const testing = std.testing;
const Allocator = std.mem.Allocator;

const fx = struct {
    var config: config_file.Config = undefined;
    var checkpoint: safetensors.Checkpoint = undefined;
    var diagnostic: config_file.Diagnostic = undefined;
    var model: model_file.Model = undefined;
    var staging: experts_file.Staging = .{};
    var scratch: *cuda.Scratch = undefined;
    const max_rows = draft_file.forward_max_rows;
    const prompt_rows = 64;
    const capacity = 512;
};

var arena_state: std.heap.ArenaAllocator = undefined;

fn load(gpa: Allocator) !void {
    return loadSeeded(gpa, null);
}

/// `load` with the checkpoint's bytes rewritten for the MTP test: `fixture.agreeable` seeds the weights
/// so the draft head's argmax is the backbone's.
fn loadSeeded(gpa: Allocator, seed: ?u64) !void {
    arena_state = .init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();
    fx.config = try fixture.parse(arena, &fx.diagnostic);
    fx.checkpoint = try fixture.synthetic(arena, fx.config, 0x5151);
    fixture.fill(&fx.checkpoint, 0x5151);
    if (seed) |s| {
        fixture.fill(&fx.checkpoint, s);
        fixture.agreeable(&fx.checkpoint);
    }
    fx.scratch = try cuda.Scratch.create(gpa);
    errdefer fx.scratch.destroy();
    fx.model = try model_file.Model.load(arena, gpa, &fx.config, &fx.checkpoint, &fx.diagnostic, &fx.staging, fx.scratch, fx.max_rows, fx.prompt_rows);
    try fx.staging.settle();
    try cuda.synchronize();
}

/// `fill` alone: finite logits that stay *varied* (no `agreeable` forcing one argmax) — the DFlash2
/// end-to-end needs a real target whose picks differ from the drafter's so rejections actually fire.
fn loadFilled(gpa: Allocator) !void {
    arena_state = .init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();
    fx.config = try fixture.parse(arena, &fx.diagnostic);
    fx.checkpoint = try fixture.synthetic(arena, fx.config, 0x5151);
    fixture.fill(&fx.checkpoint, 0xf11);
    fx.scratch = try cuda.Scratch.create(gpa);
    errdefer fx.scratch.destroy();
    fx.model = try model_file.Model.load(arena, gpa, &fx.config, &fx.checkpoint, &fx.diagnostic, &fx.staging, fx.scratch, fx.max_rows, fx.prompt_rows);
    try fx.staging.settle();
    try cuda.synchronize();
}

fn unload(gpa: Allocator) void {
    fx.model.deinit();
    fx.staging.deinit(gpa);
    fx.scratch.destroy();
    arena_state.deinit();
}

/// One fresh state + buffers, run `forward` of `tokens` then `commit` them all, and read the last row's
/// logits plus the committed position.
fn runPrompt(gpa: Allocator, tokens: []const i32, chunk: usize) !struct { logits: []u16, pos: usize } {
    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    const logits = try gpa.alloc(u16, fx.config.backbone.vocab);
    errdefer gpa.free(logits);
    var at: usize = 0;
    var last_rows: usize = 0;
    while (at < tokens.len) {
        const n = @min(@min(chunk, fx.max_rows), tokens.len - at);
        try forward_file.forward(fx.model, &state, &b, tokens[at..][0..n]);
        try forward_file.commit(fx.model, &state, b, tokens[at..][0..n], n);
        at += n;
        last_rows = n;
    }
    try forward_file.logitsRow(b, last_rows - 1, logits);
    return .{ .logits = logits, .pos = state.pos };
}

test "model loads the synthetic checkpoint, every backbone tensor" {
    const gpa = testing.allocator;
    try load(gpa);
    defer unload(gpa);
    try testing.expectEqual(@as(usize, 4), fx.model.blocks.len);
    try testing.expect(fx.model.mtp != null);
}

test "rank loaders retain only their Mamba x, B and C convolution channels" {
    const ranks = @import("../../../src/models/nemotron/ranks.zig");
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var diagnostic: config_file.Diagnostic = .{};
    var c = try fixture.parse(arena.allocator(), &diagnostic);
    c.backbone.groups = 2;
    c.backbone.moe_width = 64;
    c.backbone.shared_width = 128;
    var checkpoint = try fixture.synthetic(arena.allocator(), c, 0x5151);
    fixture.fill(&checkpoint, 0x5151);
    const source = std.mem.bytesAsSlice(u16, checkpoint.get("backbone.layers.0.mixer.conv1d.weight").?.data);
    for (0..ranks.world) |rank| {
        var staging: experts_file.Staging = .{};
        defer staging.deinit(gpa);
        const scratch = try cuda.Scratch.create(gpa);
        defer scratch.destroy();
        var model = try model_file.Model.loadShared(arena.allocator(), gpa, &c, &checkpoint, &diagnostic, &staging, scratch, fx.max_rows, fx.prompt_rows, .of(rank));
        defer model.deinit();
        try staging.settle();
        const mixer = model.blocks[0].mamba.mixer;
        const s = mixer.shape;
        const weights = try gpa.alloc(f32, 4 * s.cd());
        defer gpa.free(weights);
        try mixer.parts[3].download(std.mem.sliceAsBytes(weights));
        for (0..s.cd()) |channel| {
            const full_channel = if (channel < s.xd)
                rank * s.xd + channel
            else blk: {
                const within = channel - s.xd;
                const width = s.groups * s.state;
                break :blk c.backbone.mambaInner() + (within / width * c.backbone.groups + rank * s.groups) * s.state + within % width;
            };
            for (0..4) |tap| {
                const want: f32 = @bitCast(@as(u32, source[full_channel * 4 + tap]) << 16);
                try testing.expectEqual(want, weights[tap * s.cd() + channel]);
            }
        }
    }
}

test "a prompt whole, in chunks of 7 and of 1 is the same logits and committed state" {
    const gpa = testing.allocator;
    try load(gpa);
    defer unload(gpa);
    var random = std.Random.DefaultPrng.init(0xbeef);
    const r = random.random();
    var tokens: [40]i32 = undefined;
    for (&tokens) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    const whole = try runPrompt(gpa, &tokens, tokens.len);
    defer gpa.free(whole.logits);
    const chunked = try runPrompt(gpa, &tokens, 7);
    defer gpa.free(chunked.logits);
    const serial = try runPrompt(gpa, &tokens, 1);
    defer gpa.free(serial.logits);
    try testing.expectEqual(whole.pos, chunked.pos);
    try testing.expectEqual(whole.pos, serial.pos);
    try testing.expectEqualSlices(u16, whole.logits, chunked.logits);
    try testing.expectEqualSlices(u16, whole.logits, serial.logits);
}

test "a verify window with rejected drafts leaves the kept tokens' serial state" {
    const gpa = testing.allocator;
    try load(gpa);
    defer unload(gpa);
    var random = std.Random.DefaultPrng.init(0xdead);
    const r = random.random();
    // a prompt, then an 8-row window of which 3 are kept (5 rejected): equals the serial run of the
    // prompt + 3 kept tokens, in committed position and in the next row's logits.
    var prompt_toks: [12]i32 = undefined;
    for (&prompt_toks) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    var window: [8]i32 = undefined;
    for (&window) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    const keep = 3;

    // the serial reference: prompt + the 3 kept window tokens, one row a step
    var serial_toks: [prompt_toks.len + keep]i32 = undefined;
    @memcpy(serial_toks[0..prompt_toks.len], &prompt_toks);
    @memcpy(serial_toks[prompt_toks.len..], window[0..keep]);
    const serial = try runPrompt(gpa, &serial_toks, 1);
    defer gpa.free(serial.logits);

    // the drafted path: prompt kept whole, then the 8-row window committed at keep=3
    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    try forward_file.forward(fx.model, &state, &b, &prompt_toks);
    try forward_file.commit(fx.model, &state, b, &prompt_toks, prompt_toks.len);
    try forward_file.forward(fx.model, &state, &b, &window);
    try forward_file.commit(fx.model, &state, b, &window, keep);
    try testing.expectEqual(serial.pos, state.pos);
    // the kept rows' state feeds the same next row: run one more token and compare logits
    var next: [1]i32 = .{@intCast(r.intRangeAtMost(u32, 3, 95))};
    try forward_file.forward(fx.model, &state, &b, &next);
    const drafted_logits = try gpa.alloc(u16, fx.config.backbone.vocab);
    defer gpa.free(drafted_logits);
    try forward_file.logitsRow(b, 0, drafted_logits);
    const serial_next = try runPrompt(gpa, &(serial_toks ++ next), 1);
    defer gpa.free(serial_next.logits);
    try testing.expectEqualSlices(u16, serial_next.logits, drafted_logits);
}

test "the DFlash2 taps hold each named layer's completed residual, a row's its own" {
    const gpa = testing.allocator;
    try load(gpa);
    defer unload(gpa);
    // taps after layers 1 and 3 of the fixture's four (the published drafter's are 1, 5, 19, 29, 41, 51):
    // slot 0 is layer 1's output (the first MoE's), slot 1 layer 3's — vLLM `nemotron_h.py` captures
    // `hidden_states + residual` after decoder layer `target_layer_ids[i]`, the +1 conversion in
    // `gpu_model_runner.py` making the ids zero-based decoder layers. The fixture saturates to 0x7fff
    // early, so equality — not difference — is what is asserted: the last layer's tap is `b.hn` itself,
    // and a tap of one layer is the same bits whatever other layers are tapped.
    var random = std.Random.DefaultPrng.init(0x7a9);
    const r = random.random();
    var tokens: [6]i32 = undefined;
    for (&tokens) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    const hidden = fx.config.backbone.hidden;

    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    try b.setTaps(&.{ 1, 3 });
    try forward_file.forward(fx.model, &state, &b, &tokens);
    const taps = try gpa.alloc(u16, tokens.len * 2 * hidden);
    defer gpa.free(taps);
    try cuda.download(b.taps.?.rows(tokens.len), std.mem.sliceAsBytes(taps));
    // the forward ran every row through: taps are written, not the null buffer's bytes
    var any_nonzero = false;
    for (taps) |bits| if (bits != 0) {
        any_nonzero = true;
        break;
    };
    try testing.expect(any_nonzero);
    // slot 1 (after layer 3, the last) is the completed residual `b.hn` holds of the window, bit for bit
    const hn = try gpa.alloc(u16, tokens.len * hidden);
    defer gpa.free(hn);
    try cuda.download(b.hn.span(0, tokens.len), std.mem.sliceAsBytes(hn));
    for (0..tokens.len) |row| {
        try testing.expectEqualSlices(u16, hn[row * hidden ..][0..hidden], taps[row * 2 * hidden + hidden ..][0..hidden]);
    }

    // the same window with only layer 1 tapped: its slot is the {1, 3} run's slot 0, bit for bit —
    // a layer's tap is that layer's residual, not the taps beside it
    var state2: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state2.deinit();
    var b2: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b2.deinit();
    try b2.setTaps(&.{1});
    try forward_file.forward(fx.model, &state2, &b2, &tokens);
    const lone = try gpa.alloc(u16, tokens.len * hidden);
    defer gpa.free(lone);
    try cuda.download(b2.taps.?.span(0, tokens.len), std.mem.sliceAsBytes(lone));
    for (0..tokens.len) |row| {
        try testing.expectEqualSlices(u16, lone[row * hidden ..][0..hidden], taps[row * 2 * hidden ..][0..hidden]);
    }
}

test "the taps' rows are the serial run's bits (the row rule through the capture)" {
    const gpa = testing.allocator;
    try load(gpa);
    defer unload(gpa);
    var random = std.Random.DefaultPrng.init(0xb15);
    const r = random.random();
    var tokens: [10]i32 = undefined;
    for (&tokens) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    const hidden = fx.config.backbone.hidden;

    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    try b.setTaps(&.{ 1, 3 });
    try forward_file.forward(fx.model, &state, &b, &tokens);
    const window_taps = try gpa.alloc(u16, tokens.len * 2 * hidden);
    defer gpa.free(window_taps);
    try cuda.download(b.taps.?.rows(tokens.len), std.mem.sliceAsBytes(window_taps));

    // the same rows serially: each step's tap row is that row's bits
    var state2: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state2.deinit();
    var b2: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b2.deinit();
    try b2.setTaps(&.{ 1, 3 });
    const serial_taps = try gpa.alloc(u16, tokens.len * 2 * hidden);
    defer gpa.free(serial_taps);
    for (0..tokens.len) |row| {
        const one = [1]i32{tokens[row]};
        try forward_file.forward(fx.model, &state2, &b2, &one);
        try cuda.download(b2.taps.?.span(0, 1), std.mem.sliceAsBytes(serial_taps[row * 2 * hidden ..][0 .. 2 * hidden]));
        try forward_file.commit(fx.model, &state2, b2, &one, 1);
    }
    try testing.expectEqualSlices(u16, serial_taps, window_taps);
}

test "DFlash2 proposer halves score the same vocabulary rows as the whole target head" {
    const dflash = @import("../../../src/models/nemotron/dflash2/drafter.zig");
    const gpa = testing.allocator;
    try load(gpa);
    defer unload(gpa);
    const arena = arena_state.allocator();
    const hidden_dim = fx.config.backbone.hidden;
    const vocab = fx.config.backbone.vocab;
    const half = vocab / 2;
    const values = try arena.alloc(u16, 2 * hidden_dim);
    for (values, 0..) |*one, i| one.* = @import("../../../src/weights/numeric.zig").bf16FromF32(@as(f32, @floatFromInt(i % 17)) / 17 - 0.5);
    var x = try gpu.device(u16, .bf16, &.{ 2, hidden_dim }, values);
    defer x.deinit();
    var full = try cuda.Buffer.init(.bf16, &.{ 2, vocab });
    defer full.deinit();
    try fx.model.head.forward(x.view, full.view);
    const want = try gpu.fetch(u16, gpa, full);
    defer gpa.free(want);
    var heads = try dflash.draftHeads(2, arena, gpa, fx.model, .bf16, fx.scratch.rows(2));
    defer for (&heads) |*one| one.deinit();
    var out = try cuda.Buffer.init(.bf16, &.{ 2, half });
    defer out.deinit();
    for (heads, 0..) |head, rank| {
        try head.forward(x.view, out.view);
        const got = try gpu.fetch(u16, gpa, out);
        defer gpa.free(got);
        for (0..2) |row| try testing.expectEqualSlices(u16, want[row * vocab + rank * half ..][0..half], got[row * half ..][0..half]);
    }
}

test "a synthetic DFlash2 drafter beside the backbone replies the serial reply, drafts run and kept" {
    const drafter_testing = @import("../../../src/models/nemotron/checks/random_drafter.zig");
    const drafter_file = @import("../../../src/models/nemotron/dflash2.zig");
    const pass = @import("../../../src/decode/dflash2/pass.zig");
    const gpa = testing.allocator;
    try loadFilled(gpa);
    defer unload(gpa);
    const arena = arena_state.allocator();

    // a two-layer drafter of the backbone's shapes (hidden 64, vocabulary 131,072, four target layers):
    // nemotron's null window and the kernels' heads of 128 kept (q_inner twice hidden's share, a GQA)
    var c = try drafter_testing.config(arena, fx.config.backbone.hidden, fx.config.backbone.vocab, fx.config.backbone.types.len, &.{ 1, 3 });
    c.head_dim = 128;
    c.heads = 16;
    c.kv_heads = 2;
    c.block = 8;
    c.intermediate = 128;
    var source: drafter_testing.Source = .empty;
    var prng: std.Random.DefaultPrng = .init(0xd1a5);
    try drafter_testing.put(arena, prng.random(), c, &source);
    var checkpoint: safetensors.Checkpoint = .{ .shards = &.{}, .tensors = .empty, .gpa = arena };
    var at = source.keyIterator();
    while (at.next()) |name| try checkpoint.tensors.put(arena, name.*, source.get(name.*).?);
    var rig_source: drafter_file.Source = .{ .checkpoint = &checkpoint };
    defer rig_source.deinit(gpa);

    var flash = try gpa.create(@import("../../../src/models/nemotron/dflash2/rig.zig").Rig(2));
    defer {
        flash.deinit();
        gpa.destroy(flash);
    }
    var opened: @import("../../../src/models/nemotron/dflash2/rig.zig").Opened = .{ .config = c };
    try flash.load(arena, gpa, &opened, &rig_source, .{ 0, 1 }, fx.scratch, true, null);

    const steps = 40;
    var random = std.Random.DefaultPrng.init(0xe2e);
    const r = random.random();
    var prompt_toks: [11]i32 = undefined;
    for (&prompt_toks) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    // the same load answers twice: the serial reply on this very model, then the drafted run on its own state
    const serial = try serialReply(gpa, &prompt_toks, steps);
    defer gpa.free(serial);

    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    const heads = try @import("../../../src/models/nemotron/dflash2/drafter.zig").draftHeads(2, arena, gpa, fx.model, .bf16, .{ .pool = fx.scratch, .count = pass.batch });
    try flash.init(gpa, &opened, heads, null, fx.model, &b, fx.capacity, .{ .seed = sampling.seedFor(&prompt_toks, 0), .temperature = 0.0 }, false);
    const drafter = flash.d.drafter();
    const logits = try gpa.alloc(u16, fx.config.backbone.vocab);
    defer gpa.free(logits);
    var sampler: sampling.Sampler = try .init(gpa, logits.len);
    defer sampler.deinit(gpa);
    var decoder: draft_file.Decoder = .{
        .m = fx.model,
        .state = &state,
        .b = &b,
        .graphs = &graphs,
        .sampler = &sampler,
        .settings = .{ .seed = sampling.seedFor(&prompt_toks, 0), .temperature = 0.0 },
        .eos = fx.config.eos,
        .logits = logits,
    };
    var chosen: [draft_file.forward_max_rows]i32 = undefined;
    var drafts: [draft_file.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(drafter, &prompt_toks, &chosen);
    const reply = try gpa.alloc(i32, steps);
    defer gpa.free(reply);
    reply[0] = chosen[kept - 1];
    var made: usize = 1;
    while (made < steps) {
        const n = try decoder.draft(drafter, chosen[0..kept], @min(c.block - 1, steps - made), &drafts);
        kept = try decoder.round(chosen[kept - 1], drafts[0..n], &chosen);
        for (chosen[0..kept]) |t| {
            if (made == steps) break;
            reply[made] = t;
            made += 1;
        }
    }
    // a rejection, reorder or serial drift shows as a token off the serial reply
    try testing.expectEqualSlices(i32, serial, reply);
    // the drafter ran: its block passes proposed drafts, which rounds verified
    try testing.expect(flash.d.passes > 0);
    try testing.expect(decoder.stats.drafted > 0);
}

test "a captured window's mamba meta is the window's own, not the capture's (the graphed row rule)" {
    const gpa = testing.allocator;
    try loadFilled(gpa);
    defer unload(gpa);
    var random = std.Random.DefaultPrng.init(0x9a77);
    const r = random.random();
    // a prompt then several same-width windows: the first captures the graph, the rest replay it with a
    // different pos/parity each — the meta word must be the window's own, not the first window's frozen one
    var prompt_toks: [12]i32 = undefined;
    for (&prompt_toks) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    var window: [8]i32 = undefined;
    for (&window) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    const hidden = fx.config.backbone.hidden;

    // eager reference: prompt + the window committed at keep=3, serial bits
    var ref_state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer ref_state.deinit();
    var ref_b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer ref_b.deinit();
    var ref_graphs: forward_file.Graphs = try .init(gpa);
    defer ref_graphs.deinit();
    try forward_file.forward(fx.model, &ref_state, &ref_b, &prompt_toks);
    try forward_file.commit(fx.model, &ref_state, ref_b, &prompt_toks, prompt_toks.len);
    try forward_file.forward(fx.model, &ref_state, &ref_b, &window);
    try forward_file.commit(fx.model, &ref_state, ref_b, &window, 3);
    const want_hn = try gpa.alloc(u16, hidden);
    defer gpa.free(want_hn);
    try cuda.download(ref_b.hn.span(0, 1), std.mem.sliceAsBytes(want_hn));

    // graphed path: identical ops through Graphs.forward — the second 8-row window replays the first's graph
    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    try forward_file.forward(fx.model, &state, &b, &prompt_toks);
    try forward_file.commit(fx.model, &state, b, &prompt_toks, prompt_toks.len);
    try graphs.forward(fx.model, &state, &b, &window);
    try forward_file.commit(fx.model, &state, b, &window, 3);
    // the captured window itself gives the eager window's bits
    const got1 = try gpa.alloc(u16, hidden);
    defer gpa.free(got1);
    try cuda.download(b.hn.span(0, 1), std.mem.sliceAsBytes(got1));
    try testing.expectEqualSlices(u16, want_hn, got1);
    // replay: a second same-width window under a different pos/parity — the meta staged eagerly must differ
    var window2: [8]i32 = undefined;
    for (&window2) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    try graphs.forward(fx.model, &state, &b, &window2);
    const got_hn = try gpa.alloc(u16, hidden);
    defer gpa.free(got_hn);
    try cuda.download(b.hn.span(0, 1), std.mem.sliceAsBytes(got_hn));
    // the eager reference for the same second window
    try forward_file.forward(fx.model, &ref_state, &ref_b, &window2);
    const want2 = try gpa.alloc(u16, hidden);
    defer gpa.free(want2);
    try cuda.download(ref_b.hn.span(0, 1), std.mem.sliceAsBytes(want2));
    try testing.expectEqualSlices(u16, want2, got_hn);
}

/// The greedy reply `tokens` gets in `n` serial steps (one row a forward, `sampling.argmax` a step), for
/// the drafted run to equal token for token.
fn serialReply(gpa: Allocator, tokens: []const i32, n: usize) ![]i32 {
    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    const logits = try gpa.alloc(u16, fx.config.backbone.vocab);
    defer gpa.free(logits);
    var at: usize = 0;
    while (at < tokens.len) {
        const rows = @min(fx.max_rows, tokens.len - at);
        try forward_file.forwardScoring(fx.model, &state, &b, tokens[at..][0..rows], if (at + rows == tokens.len) .last else .none);
        try forward_file.commit(fx.model, &state, b, tokens[at..][0..rows], rows);
        at += rows;
    }
    const out = try gpa.alloc(i32, n);
    errdefer gpa.free(out);
    var last: i32 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        try forward_file.logitsRow(b, 0, logits);
        last = try sampling.argmax(logits);
        out[i] = last;
        try forward_file.forward(fx.model, &state, &b, &.{last});
        try forward_file.commit(fx.model, &state, b, &.{last}, 1);
    }
    return out;
}

test "the MTP head's drafts verified equal the serial greedy reply, and drafts are kept" {
    const gpa = testing.allocator;
    try loadSeeded(gpa, 0xf17);
    defer unload(gpa);
    // Serving prefills whole prompt windows, wider than a 16-row verify window.
    fx.model.max_rows = fx.prompt_rows;
    var random = std.Random.DefaultPrng.init(0xf00d);
    const r = random.random();
    var prompt_toks: [41]i32 = undefined;
    for (&prompt_toks) |*t| t.* = @intCast(r.intRangeAtMost(u32, 3, 95));
    const steps = 40;
    const serial = try serialReply(gpa, &prompt_toks, steps);
    defer gpa.free(serial);

    // the same prompt and reply through draft.Decoder with the MTP head drafting
    var state: state_file.State = try .init(gpa, &fx.config, fx.capacity);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&fx.config, fx.prompt_rows, fx.max_rows, fx.capacity);
    defer b.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    var head: mtp_file.Head = try .init(&fx.config, b.rows, fx.capacity);
    defer head.deinit();
    try head.bind(fx.model);
    var drafts_ctx: mtp_file.Drafter = .{ .head = &head, .state = &state };
    const drafter = drafts_ctx.drafter();
    const logits = try gpa.alloc(u16, fx.config.backbone.vocab);
    defer gpa.free(logits);
    var sampler: sampling.Sampler = try .init(gpa, logits.len);
    defer sampler.deinit(gpa);
    var decoder: draft_file.Decoder = .{
        .m = fx.model,
        .state = &state,
        .b = &b,
        .graphs = &graphs,
        .sampler = &sampler,
        .settings = .{ .seed = sampling.seedFor(&prompt_toks, 0), .temperature = 0.0 },
        .eos = fx.config.eos,
        .logits = logits,
    };
    var chosen: [draft_file.forward_max_rows]i32 = undefined;
    var drafts: [draft_file.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(drafter, &prompt_toks, &chosen);
    const reply = try gpa.alloc(i32, steps);
    defer gpa.free(reply);
    reply[0] = chosen[kept - 1];
    var made: usize = 1;
    while (made < steps) {
        const n = try decoder.draft(drafter, chosen[0..kept], @min(mtp_file.max_chain, steps - made), &drafts);
        kept = try decoder.round(chosen[kept - 1], drafts[0..n], &chosen);
        for (chosen[0..kept]) |t| {
            if (made == steps) break;
            reply[made] = t;
            made += 1;
        }
    }
    try testing.expectEqualSlices(i32, serial, reply);
    // the seeded weights make every draft the backbone's own choice, so rounds keep more than their first row
    try testing.expect(decoder.stats.drafted > 0);
    try testing.expect(decoder.stats.accepted > 0);
}

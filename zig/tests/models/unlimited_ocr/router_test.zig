//! Unlimited-OCR-2's DASH router (`router.zig`, `kernels/unlimited_ocr/router.cu`, spec 147 WS-03) on its
//! checkpoint's router weights (`$ZSPARK_UNLIMITED_OCR_2`, the 12 layers' `dash_router` alone, 8 MB) against the
//! references of its own HF code with Triton off (`$ZSPARK_REFERENCE_UNLIMITED_OCR_2`,
//! `tools/reference/unlimited_ocr/export_v2.py`), both cases, every layer:
//!
//! - the prompt from each layer's input (`prefill.attn_in`): the index q and k, k_bar of the completed blocks, the
//!   block scores, the 12 ids and their bias;
//! - every reply step from HF's own index q and keys (`step.idx_q`, `cache.idx_k`): k_bar and the scores of the
//!   detailed steps, the ids and bias of every step (the decode candidates, up to 61 blocks); the detailed steps'
//!   index q from their layer input;
//! - the row rule: the prompt in chunks of 1 to 200 rows gives the bits of the prompt in one window.
//!
//! The index q and k are bf16 values on both sides; zspark's differ where its GEMM's sum order rounds a projection
//! the other way (`tol.idx_steps` bf16 steps at most). The scores follow within `tol.scores`. Ids are compared as a
//! set over the valid slots on every row whose 12th and 13th entmax inputs lie `tol.tie` or more apart (gap12;
//! +inf with fewer than 12 candidates), the bias by block where gap16 is too: a closer pair may swap on a last-bit
//! difference of the scores, which is no fault of either side. The figures are printed.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const config_file = @import("../../../src/models/unlimited_ocr/config.zig");
const layout_file = @import("../../../src/models/unlimited_ocr/layout.zig");
const model_file = @import("../../../src/models/unlimited_ocr/model.zig");
const router_file = @import("../../../src/models/unlimited_ocr/router.zig");
const bf16 = @import("../../../src/weights/bf16.zig");

const D = router_file.dim;
const K = 12;
const page_token = 128827;

const tol = struct {
    /// The largest absolute difference of an index q or k value: two bf16 steps of the rows' largest values (4 to 8;
    /// the rows' RMS is about 1). Measured 3.1e-2, one step, on 0.02% and 0.05% of the values.
    const idx = 6.25e-2;
    /// Largest absolute error of a k_bar value, and of a block score (`idx_q . k_bar / sqrt(128)`): measured 1.0e-3
    /// and 1.6e-3.
    const kbar = 4e-3;
    const scores = 5e-3;
    /// The same from HF's own index rows (the reply steps), the sums' order alone: measured 9.5e-7 and 1.1e-6.
    const kbar_exact = 1e-5;
    const scores_exact = 1e-5;
    /// A near tie in entmax-input units (block score times 128 / sqrt(128)): about three times the largest score
    /// error times that factor (0.018). The ids that differ lie 1.2e-5 apart at most.
    const tie = 0.05;
    const tie_exact = 1e-3;
    /// A block's bias against HF's off near ties: measured 1.3e-2, and 1.6e-4 from HF's own rows.
    const bias = 5e-2;
    const bias_exact = 1e-3;
};

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

fn upload(dtype: cuda.DType, shape: []const usize, bytes: []const u8) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(bytes);
    return buffer;
}

fn zeroed(dtype: cuda.DType, shape: []const usize) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.zero();
    return buffer;
}

fn fetch(gpa: std.mem.Allocator, comptime T: type, view: cuda.View) ![]T {
    const out = try gpa.alloc(T, view.count());
    errdefer gpa.free(out);
    try cuda.synchronize();
    try cuda.download(view, std.mem.sliceAsBytes(out));
    return out;
}

/// bf16 bits of fp32 values that are bf16 values (HF's index rows).
fn toBf16(gpa: std.mem.Allocator, values: []const f32) ![]u16 {
    const out = try gpa.alloc(u16, values.len);
    for (out, values) |*o, v| {
        const bits: u32 = @bitCast(v);
        std.debug.assert(bits & 0xFFFF == 0);
        o.* = @intCast(bits >> 16);
    }
    return out;
}

/// The bf16 steps between two bf16 values (as ordered integers).
fn steps(a: u16, b: u16) u32 {
    const key = struct {
        fn of(x: u16) i32 {
            return if (x & 0x8000 != 0) -@as(i32, x & 0x7FFF) else @as(i32, x);
        }
    };
    return @abs(key.of(a) - key.of(b));
}

const Weights = struct {
    routers: []model_file.Router,
    pool: *cuda.Scratch,
    config: config_file.Config,
    rope: router_file.Rope,
    arena: std.heap.ArenaAllocator,

    fn load(gpa: std.mem.Allocator, path: []const u8) !Weights {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
        defer dir.close(testing.io);
        var diagnostic: config_file.Diagnostic = .{};
        const config = try config_file.Config.read(arena.allocator(), testing.io, dir, &diagnostic);
        var checkpoint: safetensors.Checkpoint = try .open(gpa, testing.io, .cwd(), path);
        defer checkpoint.close();
        const layout = try layout_file.Layout.init(arena.allocator(), &config, &checkpoint, &diagnostic);
        const pool = try cuda.Scratch.create(gpa);
        errdefer pool.destroy();
        const routers = try gpa.alloc(model_file.Router, config.layers);
        errdefer gpa.free(routers);
        for (routers, layout.layers) |*r, layer| {
            const t = layer.router;
            var held = try checkpoint.hold(gpa, &.{ t.q.data, t.k.data });
            defer held.release();
            r.* = .{
                .qk = try .initStacked(&.{ t.q.data, t.k.data }, config.hidden, pool.rows(1024)),
                .q_norm = try upload(.bf16, &.{D}, t.q_norm.data),
                .k_norm = try upload(.bf16, &.{D}, t.k_norm.data),
                .head_cls = try upload(.bf16, &.{D}, t.head_cls.data),
            };
        }
        const rope: router_file.Rope = try .init(config.dash.rope);
        return .{ .routers = routers, .pool = pool, .config = config, .rope = rope, .arena = arena };
    }

    fn deinit(self: *Weights, gpa: std.mem.Allocator) void {
        for (self.routers) |*r| {
            r.qk.deinit();
            r.q_norm.deinit();
            r.k_norm.deinit();
            r.head_cls.deinit();
        }
        gpa.free(self.routers);
        self.rope.deinit();
        self.pool.destroy();
        self.arena.deinit();
    }
};

const Case = struct {
    file: safetensors.File,
    name: []const u8,

    fn open(gpa: std.mem.Allocator, name: []const u8) !Case {
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}.safetensors", .{ env("ZSPARK_REFERENCE_UNLIMITED_OCR_2").?, name });
        return .{ .file = try .open(gpa, testing.io, .cwd(), path), .name = name };
    }

    fn tensor(self: Case, name: []const u8) safetensors.Tensor {
        return self.file.get(name) orelse std.debug.panic("{s}: no tensor {s}", .{ self.name, name });
    }

    fn of(self: Case, comptime T: type, name: []const u8) []const T {
        return @alignCast(std.mem.bytesAsSlice(T, self.tensor(name).data));
    }

    /// Layer l's slice of a [L, ...] tensor.
    fn layer(self: Case, comptime T: type, name: []const u8, l: usize) []const T {
        const all = self.of(T, name);
        const n = all.len / self.tensor(name).shape[0];
        return all[l * n ..][0..n];
    }
};

/// What a run found against the reference, over every layer of a case.
const Figures = struct {
    idx_steps: u32 = 0,
    idx_abs: f32 = 0,
    idx_differ: usize = 0,
    idx_values: usize = 0,
    kbar: f32 = 0,
    scores: f32 = 0,
    rows: usize = 0,
    /// rows off near ties, and those whose ids or bias differ
    compared: usize = 0,
    ids_differ: usize = 0,
    /// rows whose ids differ at any gap (a near tie)
    ids_differ_all: usize = 0,
    /// Rows, at any gap, whose valid slots are not HF's count or whose other slots are not -1 (WS-04 reads -1 as empty).
    slots_wrong: usize = 0,
    bias: f32 = 0,
    bias_rows: usize = 0,
    /// the largest gap (the less of gap12 and gap16) of a row whose ids differ
    worst_gap: f32 = 0,

    fn idx(self: *Figures, mine: []const u16, ref: []const f32) void {
        for (mine, ref) |m, r| {
            const b: u16 = @intCast(@as(u32, @bitCast(r)) >> 16);
            const s = steps(m, b);
            self.idx_steps = @max(self.idx_steps, s);
            self.idx_abs = @max(self.idx_abs, @abs(numeric.f32FromBf16(m) - r));
            if (s != 0) self.idx_differ += 1;
        }
        self.idx_values += mine.len;
    }

    /// Ids (sets over valid slots) and bias (by block) of `rows` rows against HF's.
    fn route(self: *Figures, ids: []const i32, bias: []const f32, ref_ids: []const i32, ref_bias: []const f32, ref_valid: []const u8, gap12: []const f32, gap16: []const f32, tie: f32) void {
        for (gap12, gap16, 0..) |g12, g16, r| {
            self.rows += 1;
            var mine: [K]i32 = @splat(-1);
            var theirs: [K]i32 = @splat(-1);
            var nm: usize = 0;
            var nt: usize = 0;
            for (0..K) |k| {
                if (ids[r * K + k] >= 0) {
                    mine[nm] = ids[r * K + k];
                    nm += 1;
                } else if (ids[r * K + k] != -1) self.slots_wrong += 1;
                if (ref_valid[r * K + k] != 0) {
                    theirs[nt] = ref_ids[r * K + k];
                    nt += 1;
                }
            }
            if (nm != nt) self.slots_wrong += 1;
            std.mem.sort(i32, mine[0..nm], {}, std.sort.asc(i32));
            std.mem.sort(i32, theirs[0..nt], {}, std.sort.asc(i32));
            const same = nm == nt and std.mem.eql(i32, mine[0..nm], theirs[0..nt]);
            if (!same) {
                self.ids_differ_all += 1;
                self.worst_gap = @max(self.worst_gap, @min(g12, g16));
            }
            if (!(g12 >= tie)) continue;
            self.compared += 1;
            if (!same) {
                self.ids_differ += 1;
                continue;
            }
            if (!(g16 >= tie)) continue;
            self.bias_rows += 1;
            for (0..K) |k| {
                if (ref_valid[r * K + k] == 0) continue;
                const id = ref_ids[r * K + k];
                for (0..K) |j| if (ids[r * K + j] == id) {
                    self.bias = @max(self.bias, @abs(bias[r * K + j] - ref_bias[r * K + k]));
                };
            }
        }
    }

    fn print(self: Figures, what: []const u8, case: []const u8) void {
        std.debug.print("unlimited_ocr router {s} {s}: idx {d} of {d} values differ, by {e:.2} at most ({d} bf16 steps); k_bar {e:.2}, scores {e:.2} at most; ids differ on {d} of {d} rows off near ties ({d} of {d} at any gap, {d} with slots other than HF's count and -1); bias {e:.2} at most over {d} rows; the widest gap where ids differ {e:.2}\n", .{ case, what, self.idx_differ, self.idx_values, self.idx_abs, self.idx_steps, self.kbar, self.scores, self.ids_differ, self.compared, self.ids_differ_all, self.rows, self.slots_wrong, self.bias, self.bias_rows, self.worst_gap });
    }
};

/// A layer's router over a window of rows on the device.
const Run = struct {
    x: cuda.Buffer,
    qk: cuda.Buffer,
    idx_q: cuda.Buffer,
    idx_k: cuda.Buffer,
    kbar: cuda.Buffer,
    ids: cuda.Buffer,
    bias: cuda.Buffer,
    scores: cuda.Buffer,

    fn init(rows: usize, hidden: usize, blocks: usize, width: usize) !Run {
        var self: Run = undefined;
        self.x = try .init(.bf16, &.{ rows, hidden });
        self.qk = try .init(.bf16, &.{ rows, 2 * D });
        self.idx_q = try .init(.bf16, &.{ rows, D });
        self.idx_k = try zeroed(.bf16, &.{ blocks * router_file.block, D });
        self.kbar = try zeroed(.f32, &.{ blocks, D });
        self.ids = try .init(.i32, &.{ rows, K });
        self.bias = try .init(.f32, &.{ rows, K });
        self.scores = try .init(.f32, &.{ rows, width });
        return self;
    }

    fn deinit(self: *Run) void {
        inline for (comptime std.meta.fieldNames(Run)) |name| @field(self, name).deinit();
    }

    /// Rows [first, first + n) of the window at the position `at` holds.
    fn window(self: *Run, w: *Weights, l: usize, first: usize, n: usize, at: cuda.View, coords: cuda.Buffer, pages: cuda.Buffer) !void {
        const r = w.routers[l];
        try router_file.project(r, self.x.span(first, n), self.qk.span(first, n));
        try router_file.prep(r, w.rope, w.config.eps, self.qk.span(first, n), at, coords.span(first, n), self.idx_q.span(first, n), self.idx_k.view);
        try router_file.kbar(r, self.idx_k.view, at, n, self.kbar.view);
        try router_file.route(.of(w.config.dash, l), self.idx_q.span(first, n), self.kbar.view, at, coords.span(first, n), pages.view, self.ids.span(first, n), self.bias.span(first, n), self.scores.span(first, n));
    }
};

fn coordsOf(gpa: std.mem.Allocator, tok: []const i32, page: []const i32, qp: []const i32) !cuda.Buffer {
    const rows = try gpa.alloc(router_file.Coords, tok.len);
    defer gpa.free(rows);
    for (rows, tok, page, qp) |*c, t, p, q| c.* = .{ .tok = t, .page = p, .qp = q };
    return upload(.i32, &.{ tok.len, 3 }, std.mem.sliceAsBytes(rows));
}

fn pagesOf(gpa: std.mem.Allocator, case: Case, blocks: usize) !cuda.Buffer {
    const least = case.of(i32, "state.bmin");
    const most = case.of(i32, "state.bmax");
    const rows = try gpa.alloc(router_file.Pages, blocks);
    defer gpa.free(rows);
    for (rows, 0..) |*p, b| p.* = if (b < least.len) .{ .least = least[b], .most = most[b] } else .{ .least = -1, .most = -1 };
    return upload(.i32, &.{ blocks, 2 }, std.mem.sliceAsBytes(rows));
}

test "unlimited_ocr router: the prompt, the reply steps and the row rule against HF's, both cases, every layer" {
    const path = env("ZSPARK_UNLIMITED_OCR_2") orelse return error.SkipZigTest;
    if (env("ZSPARK_REFERENCE_UNLIMITED_OCR_2") == null) return error.SkipZigTest;
    const gpa = testing.allocator;
    var w = try Weights.load(gpa, path);
    defer w.deinit(gpa);
    for ([_][]const u8{ "two_pages", "one_page" }) |name| {
        var case = try Case.open(gpa, name);
        defer case.file.close();
        try prompt(gpa, &w, case);
        try reply(gpa, &w, case);
    }
}

/// The prompt from each layer's input, in one window and in chunks.
fn prompt(gpa: std.mem.Allocator, w: *Weights, case: Case) !void {
    const S = case.of(i32, "input_ids").len;
    const blocks = (S + router_file.block - 1) / router_file.block;
    const complete = S / router_file.block;
    var coords = try coordsOf(gpa, case.of(i32, "prefill.tok"), case.of(i32, "prefill.page"), case.of(i32, "prefill.query_page_ids"));
    defer coords.deinit();
    var pages = try pagesOf(gpa, case, blocks);
    defer pages.deinit();
    // chunk starts: 1, 7, 33, 64 and 200 rows in turn
    var starts: std.ArrayList(i32) = .empty;
    defer starts.deinit(gpa);
    var at: usize = 0;
    var k: usize = 0;
    while (at < S) : (k += 1) {
        try starts.append(gpa, @intCast(at));
        at += ([_]usize{ 1, 7, 33, 64, 200 })[k % 5];
    }
    try starts.append(gpa, @intCast(S));
    var firsts = try upload(.i32, &.{starts.items.len}, std.mem.sliceAsBytes(starts.items));
    defer firsts.deinit();
    var whole = try Run.init(S, w.config.hidden, blocks, blocks);
    defer whole.deinit();
    var parts = try Run.init(S, w.config.hidden, blocks, blocks);
    defer parts.deinit();
    var fig: Figures = .{};
    for (0..w.config.layers) |l| {
        const x = std.mem.sliceAsBytes(case.layer(u16, "prefill.attn_in", l));
        try whole.x.upload(x);
        try parts.x.upload(x);
        try whole.window(w, l, 0, S, firsts.view.span(0, 1), coords, pages);
        for (starts.items[0 .. starts.items.len - 1], starts.items[1..], 0..) |a, b, c|
            try parts.window(w, l, @intCast(a), @intCast(b - a), firsts.view.span(c, 1), coords, pages);
        inline for (.{ .{ "idx_q", u16 }, .{ "idx_k", u16 }, .{ "kbar", u32 }, .{ "ids", u32 }, .{ "bias", u32 }, .{ "scores", u32 } }) |pair| {
            const field = pair[0];
            const T = pair[1];
            const one = try fetch(gpa, T, @field(whole, field).view);
            defer gpa.free(one);
            const chunked = try fetch(gpa, T, @field(parts, field).view);
            defer gpa.free(chunked);
            testing.expectEqualSlices(T, one, chunked) catch |err| {
                std.debug.print("unlimited_ocr router {s} layer {d}: {s} differs in chunks\n", .{ case.name, l, field });
                return err;
            };
        }
        const idx_q = try fetch(gpa, u16, whole.idx_q.view);
        defer gpa.free(idx_q);
        fig.idx(idx_q, case.layer(f32, "prefill.idx_q", l));
        const idx_k = try fetch(gpa, u16, whole.idx_k.view);
        defer gpa.free(idx_k);
        fig.idx(idx_k[0 .. S * D], case.layer(f32, "prefill.idx_k", l));
        const kbar = try fetch(gpa, f32, whole.kbar.view);
        defer gpa.free(kbar);
        for (kbar[0 .. complete * D], case.layer(f32, "prefill.k_bar", l)[0 .. complete * D]) |m, r| fig.kbar = @max(fig.kbar, @abs(m - r));
        const scores = try fetch(gpa, f32, whole.scores.view);
        defer gpa.free(scores);
        for (scores, case.layer(f32, "prefill.scores", l)) |m, r| {
            if (r < -1e8) try testing.expectEqual(r, m) else fig.scores = @max(fig.scores, @abs(m - r));
        }
        const ids = try fetch(gpa, i32, whole.ids.view);
        defer gpa.free(ids);
        const bias = try fetch(gpa, f32, whole.bias.view);
        defer gpa.free(bias);
        fig.route(ids, bias, case.layer(i32, "prefill.block_ids", l), case.layer(f32, "prefill.bias", l), case.layer(u8, "prefill.valid", l), case.layer(f32, "prefill.gap12", l), case.layer(f32, "prefill.gap16", l), tol.tie);
    }
    fig.print("prompt", case.name);
    try testing.expect(fig.idx_abs <= tol.idx);
    try testing.expect(fig.kbar <= tol.kbar);
    try testing.expect(fig.scores <= tol.scores);
    try testing.expect(fig.ids_differ == 0);
    try testing.expect(fig.slots_wrong == 0);
    try testing.expect(fig.bias <= tol.bias);
}

/// Every reply step from HF's index rows: the decode candidates and the cache's k_bar; the detailed steps'
/// index q from their layer input.
fn reply(gpa: std.mem.Allocator, w: *Weights, case: Case) !void {
    const S = case.of(i32, "input_ids").len;
    const tokens = case.of(i32, "step.token");
    const C = tokens.len;
    const cache_rows = case.tensor("cache.idx_k").shape[1];
    try testing.expectEqual(S + C, cache_rows);
    const blocks = (cache_rows + router_file.block - 1) / router_file.block;
    const calls = case.of(i32, "detail.calls");
    const width: usize = @intCast(case.tensor("detail.scores").shape[2]);
    // a step's query page: the <PAGE> markers so far, less one and the prompt's pages
    const qp = try gpa.alloc(i32, C);
    defer gpa.free(qp);
    var markers = case.of(i32, "state.mcum")[0];
    const prompt_pages = case.of(i32, "state.pp")[0];
    for (qp, tokens) |*q, t| {
        if (t == page_token) markers += 1;
        q.* = @max(markers - 1 - prompt_pages, -1);
    }
    var coords = try coordsOf(gpa, case.of(i32, "step.tok"), case.of(i32, "step.page"), qp);
    defer coords.deinit();
    var pages = try pagesOf(gpa, case, blocks);
    defer pages.deinit();
    const words = [2]i32{ 0, @intCast(S) };
    var at = try upload(.i32, &.{2}, std.mem.sliceAsBytes(&words));
    defer at.deinit();
    var run = try Run.init(C, w.config.hidden, blocks, @max(width, blocks));
    defer run.deinit();
    var detail = try Run.init(calls.len, w.config.hidden, 1, 1);
    defer detail.deinit();
    var dcoords = dc: {
        const t = try gpa.alloc(router_file.Coords, calls.len);
        defer gpa.free(t);
        for (t, calls) |*c, call| {
            const i: usize = @intCast(call - 1);
            c.* = .{ .tok = case.of(i32, "step.tok")[i], .page = case.of(i32, "step.page")[i], .qp = qp[i] };
        }
        break :dc try upload(.i32, &.{ calls.len, 3 }, std.mem.sliceAsBytes(t));
    };
    defer dcoords.deinit();
    var fig: Figures = .{};
    var dfig: Figures = .{};
    const step_q = case.of(f32, "step.idx_q");
    for (0..w.config.layers) |l| {
        const r = w.routers[l];
        // the cache's keys and the steps' queries, HF's bits
        const keys = try toBf16(gpa, case.layer(f32, "cache.idx_k", l));
        defer gpa.free(keys);
        try run.idx_k.uploadAt(0, std.mem.sliceAsBytes(keys));
        const queries = try gpa.alloc(u16, C * D);
        defer gpa.free(queries);
        for (0..C) |i| for (0..D) |d| {
            const bits: u32 = @bitCast(step_q[(i * w.config.layers + l) * D + d]);
            queries[i * D + d] = @intCast(bits >> 16);
        };
        try run.idx_q.upload(std.mem.sliceAsBytes(queries));
        try router_file.kbar(r, run.idx_k.view, at.view.span(0, 1), cache_rows, run.kbar.view);
        try router_file.route(.of(w.config.dash, l), run.idx_q.view, run.kbar.view, at.view.span(1, 1), coords.view, pages.view, run.ids.view, run.bias.view, run.scores.view);
        const kbar = try fetch(gpa, f32, run.kbar.view);
        defer gpa.free(kbar);
        const scores = try fetch(gpa, f32, run.scores.view);
        defer gpa.free(scores);
        const ref_kbar = case.of(f32, "detail.k_bar");
        const ref_scores = case.of(f32, "detail.scores");
        const L = w.config.layers;
        for (calls, 0..) |call, di| {
            const i: usize = @intCast(call - 1);
            const done = (S + i + 1) / router_file.block;
            for (0..done * D) |e| fig.kbar = @max(fig.kbar, @abs(kbar[e] - ref_kbar[((di * L + l) * width) * D + e]));
            for (0..width) |b| {
                const ref = ref_scores[(di * L + l) * width + b];
                if (std.math.isNan(ref)) continue;
                const m = scores[i * @as(usize, @intCast(run.scores.view.size[1])) + b];
                if (ref < -1e8) try testing.expectEqual(ref, m) else fig.scores = @max(fig.scores, @abs(m - ref));
            }
        }
        const ids = try fetch(gpa, i32, run.ids.view);
        defer gpa.free(ids);
        const bias = try fetch(gpa, f32, run.bias.view);
        defer gpa.free(bias);
        const all = struct {
            fn of(comptime T: type, c: Case, name: []const u8, layer: usize, layers: usize, width_: usize, out: []T) void {
                const src = c.of(T, name);
                for (0..out.len / width_) |i| @memcpy(out[i * width_ ..][0..width_], src[(i * layers + layer) * width_ ..][0..width_]);
            }
        };
        const ref_ids = try gpa.alloc(i32, C * K);
        defer gpa.free(ref_ids);
        const ref_bias = try gpa.alloc(f32, C * K);
        defer gpa.free(ref_bias);
        const ref_valid = try gpa.alloc(u8, C * K);
        defer gpa.free(ref_valid);
        const gap12 = try gpa.alloc(f32, C);
        defer gpa.free(gap12);
        const gap16 = try gpa.alloc(f32, C);
        defer gpa.free(gap16);
        all.of(i32, case, "step.block_ids", l, L, K, ref_ids);
        all.of(f32, case, "step.bias", l, L, K, ref_bias);
        all.of(u8, case, "step.valid", l, L, K, ref_valid);
        all.of(f32, case, "step.gap12", l, L, 1, gap12);
        all.of(f32, case, "step.gap16", l, L, 1, gap16);
        fig.route(ids, bias, ref_ids, ref_bias, ref_valid, gap12, gap16, tol.tie_exact);

        // the detailed steps' index q from their layer input
        const xin = case.of(u16, "detail.attn_in");
        const hidden = w.config.hidden;
        const x = try gpa.alloc(u16, calls.len * hidden);
        defer gpa.free(x);
        for (0..calls.len) |di| @memcpy(x[di * hidden ..][0..hidden], xin[(di * L + l) * hidden ..][0..hidden]);
        try detail.x.upload(std.mem.sliceAsBytes(x));
        try router_file.project(r, detail.x.view, detail.qk.view);
        // positions do not enter the rotation; the keys go to a scratch row past which nothing is written
        try router_file.prep(r, w.rope, w.config.eps, detail.qk.view, at.view.span(0, 1), dcoords.view, detail.idx_q.view, detail.idx_k.view);
        const dq = try fetch(gpa, u16, detail.idx_q.view);
        defer gpa.free(dq);
        for (calls, 0..) |call, di| {
            const i: usize = @intCast(call - 1);
            dfig.idx(dq[di * D ..][0..D], step_q[(i * L + l) * D ..][0..D]);
        }
    }
    fig.print("steps", case.name);
    dfig.print("detailed steps' index q", case.name);
    try testing.expect(fig.kbar <= tol.kbar_exact);
    try testing.expect(fig.scores <= tol.scores_exact);
    try testing.expect(fig.ids_differ == 0);
    try testing.expect(fig.slots_wrong == 0);
    try testing.expect(fig.bias <= tol.bias_exact);
    try testing.expect(dfig.idx_abs <= tol.idx);
}

test "unlimited_ocr router: built ties go to the lower block, a page's rows pick its image blocks, a row alone" {
    const gpa = testing.allocator;
    const blocks = 20;
    // every k_bar the same: equal scores, the ids the lowest 12, the entmax even, the bias 0 (to the logs' rounding)
    var kb: [blocks * D]f32 = @splat(0);
    for (0..blocks) |b| kb[b * D] = 1;
    var kbar = try upload(.f32, &.{ blocks, D }, std.mem.sliceAsBytes(&kb));
    defer kbar.deinit();
    // blocks 0 .. 9 hold page 0's image, 10 .. 19 page 1's
    var pg: [blocks]router_file.Pages = undefined;
    for (&pg, 0..) |*p, b| p.* = if (b < 10) .{ .least = 0, .most = 0 } else .{ .least = 1, .most = 1 };
    var pages = try upload(.i32, &.{ blocks, 2 }, std.mem.sliceAsBytes(&pg));
    defer pages.deinit();
    // rows at 800 .. 803 (22 remote blocks, 20 held): no page, page 1, page 5 (no image block: every block),
    // and at 100 (no remote block)
    const rows = 5;
    const cs = [rows]router_file.Coords{ .{ .tok = 0, .page = 0, .qp = -1 }, .{ .tok = 0, .page = 0, .qp = 1 }, .{ .tok = 0, .page = 0, .qp = 5 }, .{ .tok = 0, .page = 0, .qp = 0 }, .{ .tok = 0, .page = 0, .qp = -1 } };
    var coords = try upload(.i32, &.{ rows, 3 }, std.mem.sliceAsBytes(&cs));
    defer coords.deinit();
    var q: [rows * D]u16 = @splat(0);
    for (0..rows) |r| q[r * D] = numeric.bf16FromF32(1);
    var idx_q = try upload(.bf16, &.{ rows, D }, std.mem.sliceAsBytes(&q));
    defer idx_q.deinit();
    const words = [2]i32{ 800, 100 - 4 };
    var at = try upload(.i32, &.{2}, std.mem.sliceAsBytes(&words));
    defer at.deinit();
    var ids: cuda.Buffer = try .init(.i32, &.{ rows, K });
    defer ids.deinit();
    var bias: cuda.Buffer = try .init(.f32, &.{ rows, K });
    defer bias.deinit();
    const params: router_file.Params = .{ .scale = 128.0 / @sqrt(128.0), .floor = 1e-4, .sigma = 1 };
    try router_file.route(params, idx_q.view.span(0, 4), kbar.view, at.view.span(0, 1), coords.view.span(0, 4), pages.view, ids.view.span(0, 4), bias.view.span(0, 4), null);
    try router_file.route(params, idx_q.view.span(4, 1), kbar.view, at.view.span(1, 1), coords.view.span(4, 1), pages.view, ids.view.span(4, 1), bias.view.span(4, 1), null);
    const got = try fetch(gpa, i32, ids.view);
    defer gpa.free(got);
    const b = try fetch(gpa, f32, bias.view);
    defer gpa.free(b);
    const lowest = [K]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    const page1 = [K]i32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1, -1 };
    const page0 = [K]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1 };
    try testing.expectEqualSlices(i32, &lowest, got[0..K]);
    try testing.expectEqualSlices(i32, &page1, got[K .. 2 * K]);
    try testing.expectEqualSlices(i32, &lowest, got[2 * K .. 3 * K]);
    try testing.expectEqualSlices(i32, &page0, got[3 * K .. 4 * K]);
    try testing.expectEqualSlices(i32, &@as([K]i32, @splat(-1)), got[4 * K ..]);
    for (b) |v| try testing.expect(@abs(v) <= 1e-6);
    // each row alone gives its bits in the window
    for (0..4) |r| {
        var one: cuda.Buffer = try .init(.i32, &.{ 1, K });
        defer one.deinit();
        var one_bias: cuda.Buffer = try .init(.f32, &.{ 1, K });
        defer one_bias.deinit();
        const word = [1]i32{@intCast(800 + r)};
        var at_one = try upload(.i32, &.{1}, std.mem.sliceAsBytes(&word));
        defer at_one.deinit();
        try router_file.route(params, idx_q.view.span(r, 1), kbar.view, at_one.view, coords.view.span(r, 1), pages.view, one.view, one_bias.view, null);
        const alone = try fetch(gpa, i32, one.view);
        defer gpa.free(alone);
        try testing.expectEqualSlices(i32, got[r * K ..][0..K], alone);
    }
}

//! The host side of `zspark compare DIR REF.safetensors` (spec 151 WS-05): reading the tensors
//! `tools/reference/nemotron_transcribe/export_model.py` writes for one clip, lining the fixture's
//! forward rows up with a serial replay, and the distance and WER numbers the command reports.
//!
//! The fixture's row convention: `forward.num_rows[f]` rows per backbone forward (the prefill's prompt
//! rows, then one 9-row MTP verify window a step). `hidden.Lxx`/`residual.Lxx` concatenate every
//! forward's rows in order; a layer's completed output is bf16(hidden + residual) (vLLM's
//! aux-hidden convention, `hidden_states + residual` after the block). Verify rows past a forward's
//! accepted count are speculative: they were computed but never committed, so a serial replay has no
//! row to hold against them — `Rows` maps only the committed positions.

const std = @import("std");
const Allocator = std.mem.Allocator;
const numeric = @import("../../../weights/numeric.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const unicode = @import("tokenizer").unicode;

pub const Error = error{ MissingTensor, UnexpectedShape, UnexpectedDtype } || Allocator.Error;

/// An int32 tensor's values.
pub fn ints(arena: Allocator, file: *const safetensors.File, name: []const u8) Error![]i32 {
    const t = file.get(name) orelse return error.MissingTensor;
    if (t.dtype != .I32) return error.UnexpectedDtype;
    const out = try arena.alloc(i32, @intCast(t.count()));
    @memcpy(std.mem.sliceAsBytes(out), t.data);
    return out;
}

/// Borrowed BF16 layer outputs. Compare completed streams a row at a time without widening the
/// whole hour-long fixture into another allocation.
pub const LayerRef = struct { hidden: safetensors.Tensor, residual: safetensors.Tensor };

pub fn layerTensors(file: *const safetensors.File, layer: usize) Error!LayerRef {
    var name: [32]u8 = undefined;
    const h = file.get(std.fmt.bufPrint(&name, "hidden.L{d:0>2}", .{layer}) catch unreachable) orelse return error.MissingTensor;
    var res_name: [32]u8 = undefined;
    const r = file.get(std.fmt.bufPrint(&res_name, "residual.L{d:0>2}", .{layer}) catch unreachable) orelse return error.MissingTensor;
    if (h.dtype != .BF16 or r.dtype != .BF16 or h.shape.len != 2 or r.shape.len != 2 or h.shape[0] != r.shape[0] or h.shape[1] != r.shape[1]) return error.UnexpectedShape;
    return .{ .hidden = h, .residual = r };
}

/// Complete one captured row with vLLM's BF16(hidden + residual) rounding.
pub fn completedRow(t: LayerRef, row: usize, out: []f32) void {
    const width = out.len;
    const h = std.mem.bytesAsSlice(u16, t.hidden.data)[row * width ..][0..width];
    const r = std.mem.bytesAsSlice(u16, t.residual.data)[row * width ..][0..width];
    for (out, h, r) |*v, a, b| v.* = numeric.f32FromBf16(numeric.bf16FromF32(numeric.f32FromBf16(a) + numeric.f32FromBf16(b)));
}

/// The reference logits one forward's row produced, f32 [vocab or wider]: `logits.step{f}` holds the
/// forward's last `shape[0]` rows only (the prefill's last row; a verify window's every row), stored
/// padded (the tokenizer's added columns past `vocab_size`); callers take `row[0..vocab]` and the
/// rest is said once, not silently dropped.
pub fn logits(arena: Allocator, file: *const safetensors.File, forward_index: usize, forward_rows: usize, row: usize, vocab: usize) Error![]f32 {
    var name: [32]u8 = undefined;
    const t = file.get(std.fmt.bufPrint(&name, "logits.step{d:0>2}", .{forward_index}) catch unreachable) orelse return error.MissingTensor;
    if (t.dtype != .F32 or t.shape.len != 2 or t.shape[1] < vocab or t.shape[0] == 0 or t.shape[0] > forward_rows or row >= forward_rows) return error.UnexpectedShape;
    if (forward_rows - row > t.shape[0]) return error.UnexpectedShape; // the row was not among the scored last rows
    const stored_row = t.shape[0] - (forward_rows - row);
    const bits = std.mem.bytesAsSlice(u32, t.data)[stored_row * t.shape[1] ..][0..vocab];
    const out = try arena.alloc(f32, vocab);
    for (out, bits) |*v, b| v.* = @bitCast(b);
    return out;
}

/// How many of forward `forward_index`'s last rows `logits.step{f}` stores (0 when absent).
pub fn scoredRows(file: *const safetensors.File, forward_index: usize) usize {
    var name: [32]u8 = undefined;
    const t = file.get(std.fmt.bufPrint(&name, "logits.step{d:0>2}", .{forward_index}) catch unreachable) orelse return 0;
    return t.shape[0];
}

/// How far a verify forward reached: the position its next forward started at tells how many rows it
/// committed — `pos_{f+1} = pos_f + 1 + accepted_f`. The last forward's commits are bounded by the
/// reply's length: positions below `prompt + tokens` are real, the rest of its window speculative.
pub const Rows = struct {
    /// Position -> global row index into the concatenated hidden/residual tensors, committed rows only.
    row_of: []i64,
    /// `logits.step{f}[r]` is comparable at `scored_pos[k]` — the position whose serial forward should
    /// equal it — where `scored_pos[k] >= 0`; `scored_f[k]` and `scored_r[k]` say which tensor row.
    /// A row counts only where its own and every earlier row of its window committed on the chain.
    scored_pos: []i64,
    scored_f: []u16,
    scored_r: []u16,
    /// Global row index of forward f's row r.
    first: []usize,
    /// Positions the reply ran that no committed reference row covers (rejected spans).
    unmapped: usize,

    /// Global row index of forward f's row r.
    pub fn at(self: Rows, f: usize, r: usize) usize {
        return self.first[f] + r;
    }
};
/// The committed-row map from `forward.num_rows`, `forward.positions` and each forward's
/// `forward.input_ids`: position p's global row is the row the serial replay of position p is
/// compared with. A verify row is mapped only while its input id continues the committed chain —
/// `prompt.ids` then `tokens.greedy` — so a row computed under a rejected draft's wrong prefix is
/// never held against the honest row even where its position coincidentally matches. The prefill's
/// forward often ran on inputs_embeds with its ids absent; forward 0 then takes `prompt` itself as
/// its ids; every ids tensor absent maps by position alone. `first[f] + r` is row r's global row.
pub fn rows(arena: Allocator, file: *const safetensors.File, num_rows: []const i32, positions: []const i32, input_ids: []const i32, prompt: []const i32, greedy: []const i32) Error!Rows {
    const forwards = num_rows.len;
    var total: usize = 0;
    for (num_rows) |r| total += @intCast(r);
    if (positions.len != total) return error.UnexpectedShape;
    const last_position = prompt.len + greedy.len;
    const row_of = try arena.alloc(i64, last_position);
    @memset(row_of, -1);
    const first = try arena.alloc(usize, forwards);
    var scored_pos: std.ArrayList(i64) = .empty;
    var scored_f: std.ArrayList(u16) = .empty;
    var scored_r: std.ArrayList(u16) = .empty;
    // The committed chain a verify row's id must continue: position p expects this token.
    const chain = try arena.alloc(i32, last_position);
    @memcpy(chain[0..prompt.len], prompt);
    @memcpy(chain[prompt.len..], greedy);
    var offset: usize = 0;
    // Inputs-embeds prefills can omit ids across several chunks; decode ids then follow them.
    if (input_ids.len > total) return error.UnexpectedShape;
    const missing0 = total - input_ids.len;
    var prefill_rows: usize = 0;
    var off0: usize = 0;
    for (num_rows) |nr| {
        const p = positions[off0];
        if (p < 0 or p >= prompt.len) break;
        prefill_rows += @intCast(nr);
        off0 += @intCast(nr);
    }
    if (input_ids.len != 0 and input_ids.len != total and missing0 != prefill_rows) return error.UnexpectedShape;
    for (0..forwards) |f| {
        first[f] = offset;
        const n: usize = @intCast(num_rows[f]);
        const pos0 = positions[offset];
        const is_prefill = pos0 >= 0 and pos0 < prompt.len;
        const ids: ?[]const i32 = if (input_ids.len == 0)
            null
        else if (missing0 != 0 and is_prefill)
            prompt[@intCast(pos0)..][0..n]
        else
            input_ids[offset - missing0 ..][0..n];
        // the furthest position this forward committed: the next forward's start, or the reply's end
        const limit = if (f + 1 < forwards) @as(i64, positions[offset + n]) else @as(i64, @intCast(last_position));
        var committed = true; // while the window's ids follow the chain row by row
        const scored_n = scoredRows(file, f); // the forward's last rows the exporter stored logits for
        for (0..n) |r| {
            const pos = positions[offset + r];
            if (pos < 0 or pos >= last_position) continue;
            if (pos >= limit) break; // candidates: nothing honest follows them in this window
            if (committed and ids != null and ids.?[r] != chain[@intCast(pos)]) committed = false;
            if (!committed) break; // this and later rows ran under a rejected prefix
            if (row_of[@intCast(pos)] == -1) row_of[@intCast(pos)] = @intCast(offset + r);
            // row r's logits predict the token after its position — comparable where it sits
            // among the forward's last `scored_n` (the only rows `logits.step{f}` stored).
            if (r >= n - scored_n) {
                try scored_pos.append(arena, pos);
                try scored_f.append(arena, @intCast(f));
                try scored_r.append(arena, @intCast(r));
            }
        }
        offset += n;
    }
    var unmapped: usize = 0;
    for (row_of) |r| unmapped += @intFromBool(r < 0);
    return .{ .row_of = row_of, .scored_pos = scored_pos.items, .scored_f = scored_f.items, .scored_r = scored_r.items, .first = first, .unmapped = unmapped };
}
/// element's absolute error, cosine similarity over the whole span, and the argmax row disagreement.
pub const Stats = struct {
    /// rms(mine - reference) / rms(reference), the worst row's.
    rel_worst: f64 = 0,
    /// The same over every element together.
    rel_all: f64 = 0,
    max_abs: f64 = 0,
    /// sum(a·b) / (|a||b|) over every element.
    cosine: f64 = 1,
    /// Rows (of `rows` each `width` wide) whose largest element sits at a different index.
    picks_differing: usize = 0,
    rows: usize = 0,
    worst_row: usize = 0,
};

/// `mine` (f32) against `theirs` (f32), [rows, width]: every row's relative rms folded into `Stats`,
/// each row's first-largest index compared where `picks` is set.
pub fn compare(mine: []const f32, theirs: []const f32, rows_n: usize, picks: bool) Stats {
    std.debug.assert(rows_n > 0 and mine.len == theirs.len and mine.len % rows_n == 0);
    const width = mine.len / rows_n;
    var s: Stats = .{ .rows = rows_n };
    var diff_all: f64 = 0;
    var ref_all: f64 = 0;
    var dot: f64 = 0;
    var mine_sq: f64 = 0;
    for (0..rows_n) |r| {
        const a = mine[r * width ..][0..width];
        const b = theirs[r * width ..][0..width];
        var diff: f64 = 0;
        var ref: f64 = 0;
        var top_a: usize = 0;
        var top_b: usize = 0;
        var va: f64 = -std.math.inf(f64);
        var vb: f64 = -std.math.inf(f64);
        for (a, b, 0..) |x, y, i| {
            const d = x - y;
            diff += d * d;
            ref += y * y;
            dot += x * y;
            mine_sq += x * x;
            s.max_abs = @max(s.max_abs, @abs(d));
            if (x > va) {
                va = x;
                top_a = i;
            }
            if (y > vb) {
                vb = y;
                top_b = i;
            }
        }
        if (picks and top_a != top_b) s.picks_differing += 1;
        const rel = if (ref > 0) @sqrt(diff / ref) else @sqrt(diff);
        if (rel >= s.rel_worst) {
            s.rel_worst = rel;
            s.worst_row = r;
        }
        diff_all += diff;
        ref_all += ref;
    }
    s.rel_all = if (ref_all > 0) @sqrt(diff_all / ref_all) else @sqrt(diff_all);
    s.cosine = if (mine_sq > 0 and ref_all > 0) dot / @sqrt(mine_sq * ref_all) else 1;
    return s;
}

/// `theirs` (the fixture's f32 rows) against mine, bf16 bits downloaded from the device: the conversion
/// point between `cuda.download` output and `compare`.
pub fn compareBf16(mine: []const u16, theirs: []const f32, rows_n: usize, picks: bool, arena: Allocator) Allocator.Error!Stats {
    const widened = try arena.alloc(f32, mine.len);
    for (widened, mine) |*v, b| v.* = numeric.f32FromBf16(b);
    return compare(widened, theirs, rows_n, picks);
}

/// The fixed clip set's metrics: lowercase, punctuation-free English WER; Chinese CER with spaces
/// removed and punctuation retained, matching `tools/reference/nemotron_transcribe/export.py`.
pub const Wer = struct {
    /// substitutions + insertions + deletions over reference units.
    distance: usize,
    units: usize,
    /// Chinese/CJK text uses codepoints; a single English word still counts as one word.
    codepoints: bool,

    /// distance/units raw (it can exceed 1: every inserted word counts).
    pub fn rate(self: Wer) f64 {
        return if (self.units > 0) @as(f64, @floatFromInt(self.distance)) / @as(f64, @floatFromInt(self.units)) else @floatFromInt(self.distance);
    }
};

fn hasCjk(text: []const u8) error{InvalidUtf8}!bool {
    var view = try std.unicode.Utf8View.init(text);
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if ((cp >= 0x3040 and cp <= 0x30ff) or (cp >= 0x3400 and cp <= 0x9fff) or
            (cp >= 0xf900 and cp <= 0xfaff) or (cp >= 0x20000 and cp <= 0x323af)) return true;
    }
    return false;
}

fn units(arena: Allocator, text: []const u8, characters: bool) (Allocator.Error || error{InvalidUtf8})!struct { list: [][]const u8, codepoints: bool } {
    var list: std.ArrayList([]const u8) = .empty;
    var view = try std.unicode.Utf8View.init(text);
    var it = view.iterator();
    if (characters) {
        while (it.nextCodepointSlice()) |raw| {
            const cp = std.unicode.utf8Decode(raw) catch unreachable;
            if (!unicode.isSpace(cp) and !(cp >= 0x1c and cp <= 0x1f)) try list.append(arena, raw);
        }
    } else {
        const normalized = try arena.alloc(u8, text.len);
        var n: usize = 0;
        var end: usize = 0;
        while (it.nextCodepointSlice()) |raw| {
            const cp = std.unicode.utf8Decode(raw) catch unreachable;
            if (unicode.bit(unicode.category(cp)) & unicode.punctuation != 0) continue;
            const space = unicode.isSpace(cp) or (cp >= 0x1c and cp <= 0x1f);
            if (n == 0 and space) continue;
            for (raw) |byte| {
                normalized[n] = std.ascii.toLower(byte);
                n += 1;
            }
            if (!space) end = n;
        }
        var words = std.mem.tokenizeScalar(u8, normalized[0..end], ' ');
        while (words.next()) |word| try list.append(arena, word);
    }
    return .{ .list = list.items, .codepoints = characters };
}

/// Word (or codepoint) error rate of `hypothesis` against `reference`.
pub fn wer(arena: Allocator, reference_text: []const u8, hypothesis_text: []const u8) (Allocator.Error || error{InvalidUtf8})!Wer {
    const characters = try hasCjk(reference_text) or (reference_text.len == 0 and try hasCjk(hypothesis_text));
    const ref = try units(arena, reference_text, characters);
    const hyp = try units(arena, hypothesis_text, characters);
    const n = ref.list.len;
    const m = hyp.list.len;
    if (n == 0) return .{ .distance = m, .units = 0, .codepoints = ref.codepoints };
    var prev = try arena.alloc(usize, m + 1);
    var curr = try arena.alloc(usize, m + 1);
    for (prev, 0..) |*v, j| v.* = j;
    for (0..n) |i| {
        curr[0] = i + 1;
        for (0..m) |j| {
            const cost: usize = if (std.mem.eql(u8, ref.list[i], hyp.list[j])) 0 else 1;
            curr[j + 1] = @min(@min(prev[j + 1] + 1, curr[j] + 1), prev[j] + cost);
        }
        std.mem.swap([]usize, &prev, &curr);
    }
    return .{ .distance = prev[m], .units = n, .codepoints = ref.codepoints };
}

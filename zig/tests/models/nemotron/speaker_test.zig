//! The Sortformer speaker encoder's device path against a host reference of NeMo's streaming math
//! (spec 151 WS-11): `speaker.zig` on the fixture's synthetic diarization tensors, a clip's speaker
//! activity window by window (spkcache, FIFO, the sigmoid mask and the `_compress_spkcache` score
//! bookkeeping carried between), then `_fuse_diar_and_asr` beside `asr_norm`. The reference is
//! `sortformer_diar_models.forward_streaming_step` + `streaming_update` + `_fuse_diar_and_asr`
//! (commit 774bc30) written out in f32/bf16: the device matches it to bf16's roundings where the
//! kernels' arithmetic is PyTorch's, and loosely where it is near (LayerNorm's sums, attention).

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const config_file = @import("../../../src/models/nemotron/config.zig");
const layout_file = @import("../../../src/models/nemotron/layout.zig");
const perception_file = @import("../../../src/models/nemotron/perception.zig");
const speaker_file = @import("../../../src/models/nemotron/speaker.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const fixture = @import("fixture.zig");

fn f32v(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn bf(x: f32) f32 {
    return f32v(perception_file.bf16Of(x));
}

fn loadPe(arena: std.mem.Allocator) !struct { c: config_file.Config, checkpoint: safetensors.Checkpoint } {
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.parse(arena, fixture.config_text_pe, fixture.backbone_text, fixture.quant_text, &diagnostic);
    var checkpoint = try fixture.synthetic(arena, c, 11);
    fixture.fill(&checkpoint, 13);
    return .{ .c = c, .checkpoint = checkpoint };
}

/// The host Sortformer: `forward_streaming_step`'s module order on one window, f32 with a bf16
/// rounding where the kernels store one (linears' sums in f64 like `Ref.linear`'s — the comparison
/// is a cosine, not bits, for the same reason perception's is).
const Ref = struct {
    c: config_file.Config,
    checkpoint: *const safetensors.Checkpoint,

    const P = "perception.encoder.diarization_model.";

    fn vec(self: Ref, name: []const u8, a: std.mem.Allocator) ![]f32 {
        const t = self.checkpoint.get(name).?;
        const out = try a.alloc(f32, t.data.len / 2);
        for (out, std.mem.bytesAsSlice(u16, t.data)) |*v, b| v.* = f32v(b);
        return out;
    }

    fn vecN(self: Ref, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) ![]f32 {
        var b: [224]u8 = undefined;
        return self.vec(std.fmt.bufPrint(&b, fmt, args) catch unreachable, a);
    }

    fn norm(x: []f32, at_row: usize, w: []const f32, b: []const f32, out: []f32) void {
        var sum: f64 = 0;
        var sq: f64 = 0;
        for (0..w.len) |t| {
            sum += x[at_row + t];
            sq += @as(f64, x[at_row + t]) * x[at_row + t];
        }
        const mean = sum / @as(f64, @floatFromInt(w.len));
        const rstd = 1.0 / @sqrt(sq / @as(f64, @floatFromInt(w.len)) - mean * mean + 1e-5);
        for (0..w.len) |t| out[at_row + t] = bf(@as(f32, @floatCast((x[at_row + t] - mean) * rstd)) * w[t] + b[t]);
    }

    fn linear(w: []const f32, bias: ?[]const f32, x: []const f32, rows: usize, k: usize, n: usize, out: []f32) void {
        for (0..rows) |r| {
            for (0..n) |j| {
                var acc: f64 = if (bias) |b| b[j] else 0;
                for (0..k) |i| acc += @as(f64, x[r * k + i]) * w[j * k + i];
                out[r * n + j] = bf(@floatCast(acc));
            }
        }
    }

    /// A window's streaming step: `mel` f32 [frames, features] of the clip's normed mel, `cs..ce` the
    /// window's frames, `left_off`/`right_off` its raw context in frames; `state` the streaming
    /// spkcache/FIFO (host f32/bf16). Returns the chunk's native preds f32 [core*sub, speakers].
    fn step(self: Ref, a: std.mem.Allocator, d: config_file.Diar, mel: []const f32, features: usize, cs: usize, ce: usize, valid: usize, left_off: usize, right_off: usize, last: bool, state: *State) ![]f32 {
        const dm = d.d_model;
        const s = d.speakers;
        const sub = d.subsampling;
        const frames = ce - cs;
        const rows = (frames + sub - 1) / sub;
        const packed_n = state.spk_n + state.fifo_n + rows;
        const key_len = state.spk_n + state.fifo_n + @min(rows, (if (cs < valid) @min(frames, valid - cs) + sub - 1 else 0) / sub);
        // pre_encode the chunk's stacked rows (the state rows are already embeddings)
        const emb = try a.alloc(f32, packed_n * dm);
        @memcpy(emb[0 .. state.spk_n * dm], state.spk[0 .. state.spk_n * dm]);
        @memcpy(emb[state.spk_n * dm ..][0 .. state.fifo_n * dm], state.fifo[0 .. state.fifo_n * dm]);
        {
            const w = try self.vecN(a, P ++ "encoder.pre_encode.proj.weight", .{});
            const stacked = try a.alloc(f32, rows * d.stacked());
            @memset(stacked, 0);
            for (0..frames) |f| {
                for (0..features) |m|
                    stacked[(f / sub) * d.stacked() + (f % sub) * features + m] = bf(mel[(cs + f) * features + m]);
            }
            linear(w, null, stacked, rows, d.stacked(), dm, emb[(state.spk_n + state.fifo_n) * dm ..]);
        }
        // the encoder: embed_norm -> pre-norm attention + ffn layers -> final_norm
        const x = try a.dupe(f32, emb); // streaming_update retains the pre-encoder embeddings.
        const y = try a.alloc(f32, packed_n * dm);
        const qkv = try a.alloc(f32, packed_n * 3 * dm);
        const attn = try a.alloc(f32, packed_n * dm);
        const ffn = try a.alloc(f32, packed_n * d.ffWidth());
        {
            const w = try self.vecN(a, P ++ "encoder.embed_norm.weight", .{});
            const b = try self.vecN(a, P ++ "encoder.embed_norm.bias", .{});
            for (0..packed_n) |r| norm(x, r * dm, w, b, x);
        }
        const rot = @as(usize, @intFromFloat(d.rotary_fraction * @as(f64, @floatFromInt(d.headDim()))));
        var inv: [128]f32 = undefined;
        for (0..rot / 2) |i| inv[i] = @floatCast(1.0 / std.math.pow(f64, d.rope_base, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(rot))));
        for (0..d.layers) |l| {
            const n1w = try self.vecN(a, layout_file.diar_prefix ++ "{d}.norm1.weight", .{l});
            const n1b = try self.vecN(a, layout_file.diar_prefix ++ "{d}.norm1.bias", .{l});
            const n2w = try self.vecN(a, layout_file.diar_prefix ++ "{d}.norm2.weight", .{l});
            const n2b = try self.vecN(a, layout_file.diar_prefix ++ "{d}.norm2.bias", .{l});
            for (0..packed_n) |r| norm(x, r * dm, n1w, n1b, y);
            {
                const w = try self.vecN(a, layout_file.diar_prefix ++ "{d}.attn.w_qkv.weight", .{l});
                linear(w, null, y, packed_n, dm, 3 * dm, qkv);
                // RoPE on q,k of each head over the whole packed segment, position = row index
                const hd = d.headDim();
                const half = rot / 2;
                for (0..packed_n) |r| {
                    for (0..d.heads) |h| {
                        for (0..2) |part| {
                            const base = r * 3 * dm + part * dm + h * hd;
                            for (0..half) |p| {
                                const ang = @as(f32, @floatFromInt(r)) * inv[p];
                                const cs_ = bf(@cos(ang));
                                const sn = bf(@sin(ang));
                                const n1 = qkv[base + p];
                                const n2 = qkv[base + p + half];
                                qkv[base + p] = bf(bf(n1 * cs_) + bf(-n2 * sn));
                                qkv[base + p + half] = bf(bf(n2 * cs_) + bf(n1 * sn));
                            }
                        }
                    }
                }
                // softmax attention over keys < key_len
                for (0..packed_n) |r| {
                    for (0..d.heads) |h| {
                        const qs = qkv[r * 3 * dm + h * hd ..][0..hd];
                        var scores = try a.alloc(f32, key_len);
                        var mx: f32 = -std.math.inf(f32);
                        for (0..key_len) |kk| {
                            var dot: f64 = 0;
                            for (0..hd) |t| dot += qs[t] * qkv[kk * 3 * dm + dm + h * hd + t];
                            scores[kk] = @floatCast(dot / @sqrt(@as(f64, @floatFromInt(hd))));
                            mx = @max(mx, scores[kk]);
                        }
                        var den: f64 = 0;
                        for (scores) |*v| {
                            v.* = @exp(v.* - mx);
                            den += v.*;
                        }
                        for (0..hd) |t| {
                            var acc: f64 = 0;
                            for (0..key_len) |kk| acc += scores[kk] / den * qkv[kk * 3 * dm + 2 * dm + h * hd + t];
                            attn[r * dm + h * hd + t] = bf(@floatCast(acc));
                        }
                    }
                }
            }
            {
                const w = try self.vecN(a, layout_file.diar_prefix ++ "{d}.attn.out_proj.weight", .{l});
                const b = try self.vecN(a, layout_file.diar_prefix ++ "{d}.attn.out_proj.bias", .{l});
                linear(w, b, attn, packed_n, dm, dm, y);
                for (0..packed_n * dm) |t| x[t] = bf(x[t] + y[t]);
            }
            for (0..packed_n) |r| norm(x, r * dm, n2w, n2b, y);
            {
                const w0 = try self.vecN(a, layout_file.diar_prefix ++ "{d}.ffn.net.0.weight", .{l});
                const b0 = try self.vecN(a, layout_file.diar_prefix ++ "{d}.ffn.net.0.bias", .{l});
                const w3 = try self.vecN(a, layout_file.diar_prefix ++ "{d}.ffn.net.3.weight", .{l});
                const b3 = try self.vecN(a, layout_file.diar_prefix ++ "{d}.ffn.net.3.bias", .{l});
                linear(w0, b0, y, packed_n, dm, d.ffWidth(), ffn);
                for (ffn) |*v| v.* = bf(gelu(v.*));
                linear(w3, b3, ffn, packed_n, d.ffWidth(), dm, y);
                for (0..packed_n * dm) |t| x[t] = bf(x[t] + y[t]);
            }
        }
        {
            const w = try self.vecN(a, P ++ "encoder.final_norm.weight", .{});
            const b = try self.vecN(a, P ++ "encoder.final_norm.bias", .{});
            for (0..packed_n) |r| norm(x, r * dm, w, b, x);
        }
        // encoder_proj -> masked_fill -> subpixel upsample (x sub) -> relu -> first_hidden -> relu ->
        // single_hidden -> sigmoid masked
        const tf = d.tf_d_model;
        const proj = try a.alloc(f32, packed_n * tf);
        {
            const w = try self.vecN(a, P ++ "sortformer_modules.encoder_proj.weight", .{});
            const b = try self.vecN(a, P ++ "sortformer_modules.encoder_proj.bias", .{});
            linear(w, b, x, packed_n, dm, tf, proj);
            for (0..packed_n) |r| if (r >= key_len) {
                for (0..tf) |t| proj[r * tf + t] = 0;
            };
        }
        const up = sub;
        const hid = try a.alloc(f32, packed_n * up * tf);
        {
            const w = try self.vecN(a, P ++ "sortformer_modules.subpixel_upsample.weight", .{});
            const b = try self.vecN(a, P ++ "sortformer_modules.subpixel_upsample.bias", .{});
            // Ordinary Conv1d(groups=1): every output channel reads every input channel.
            for (0..packed_n) |i| {
                for (0..up) |cc| {
                    for (0..tf) |ch| {
                        const oc = cc * tf + ch;
                        var acc: f64 = b[oc];
                        for (0..tf) |ic| {
                            for (0..3) |k| {
                                const src = @as(i64, @intCast(i)) + @as(i64, @intCast(k)) - 1;
                                const xv: f32 = if (src >= 0 and src < packed_n) proj[@as(usize, @intCast(src)) * tf + ic] else 0;
                                acc += xv * w[(oc * tf + ic) * 3 + k];
                            }
                        }
                        hid[(i * up + cc) * tf + ch] = bf(@floatCast(acc));
                    }
                }
            }
        }
        const preds = try a.alloc(f32, packed_n * up * s);
        {
            const hw = try self.vecN(a, P ++ "sortformer_modules.first_hidden_to_hidden.weight", .{});
            const hb = try self.vecN(a, P ++ "sortformer_modules.first_hidden_to_hidden.bias", .{});
            const sw = try self.vecN(a, P ++ "sortformer_modules.single_hidden_to_spks.weight", .{});
            const sb = try self.vecN(a, P ++ "sortformer_modules.single_hidden_to_spks.bias", .{});
            const h2 = try a.alloc(f32, packed_n * up * tf);
            for (hid) |*v| v.* = @max(0, v.*);
            linear(hw, hb, hid, packed_n * up, tf, tf, h2);
            for (h2) |*v| v.* = @max(0, v.*);
            linear(sw, sb, h2, packed_n * up, tf, s, preds);
            for (preds, 0..) |*v, i| {
                const live = (i / s) < key_len * up;
                v.* = if (live) bf(1.0 / (1.0 + @exp(-v.*))) else 0;
            }
        }
        // the chunk's native preds at (spk+fifo+lc_enc)*up .. core*up
        const lc_enc = (left_off + sub / 2) / sub;
        const rc_enc = rows - lc_enc - ((frames - (lc_enc * sub + right_off)) + sub - 1) / sub; // ceil(right_off/sub)
        const core = rows - lc_enc - rc_enc;
        const native = try a.alloc(f32, core * up * s);
        @memcpy(native, preds[((state.spk_n + state.fifo_n + lc_enc) * up) * s ..][0 .. core * up * s]);
        // streaming_update on the downsampled preds (skipped on the last window)
        if (!last) {
            const down = try a.alloc(f32, packed_n * s);
            for (0..packed_n) |r| {
                for (0..s) |sp| {
                    var sum: f64 = 0;
                    for (0..up) |e| sum += preds[(r * up + e) * s + sp];
                    down[r * s + sp] = @floatCast(sum / @as(f64, @floatFromInt(up)));
                }
            }
            try state.update(a, d, emb[(state.spk_n + state.fifo_n) * dm ..], lc_enc, core, down, self);
        }
        return native;
    }

    fn gelu(x: f32) f32 {
        const t = 1.0 / (1.0 + 0.2316419 * @abs(x));
        const poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
        const phi = 1.0 - @exp(-x * x / 2.0) * poly / @sqrt(2.0 * std.math.pi);
        return x * (if (x >= 0) phi else 1.0 - phi);
    }

    /// `streaming_update`'s host state: spkcache/fifo embeddings (f32; the device keeps bf16), the
    /// preds each holds, and `_compress_spkcache`'s score path.
    const State = struct {
        spk: []f32,
        spk_preds: []f32,
        spk_n: usize,
        compressed: bool,
        fifo: []f32,
        fifo_preds: []f32,
        fifo_n: usize,
        sil: []f32,

        fn init(a: std.mem.Allocator, d: config_file.Diar, ref: Ref) !State {
            const sil = try ref.vecN(a, P ++ "sortformer_modules.learnable_sil_emb", .{});
            return .{
                .spk = try a.alloc(f32, (d.spkcache_len + 256) * d.d_model),
                .spk_preds = try a.alloc(f32, (d.spkcache_len + 256) * d.speakers),
                .spk_n = 0,
                .compressed = false,
                .fifo = try a.alloc(f32, (d.fifo_len + 256) * d.d_model),
                .fifo_preds = try a.alloc(f32, (d.fifo_len + 256) * d.speakers),
                .fifo_n = 0,
                .sil = sil,
            };
        }

        /// `chunk_embs` the window's pre-encoded rows [rows, dm], `lc`/`core` its context and kept
        /// rows, `down` the downsampled packed preds [packed_n, s].
        fn update(self: *State, a: std.mem.Allocator, d: config_file.Diar, chunk_embs: []const f32, lc: usize, core: usize, down: []const f32, ref: Ref) !void {
            const s = d.speakers;
            const dm = d.d_model;
            const spk_before = self.spk_n;
            const fifo_before = self.fifo_n;
            @memcpy(self.fifo_preds[0 .. fifo_before * s], down[spk_before * s ..][0 .. fifo_before * s]);
            @memcpy(self.fifo_preds[fifo_before * s ..][0 .. core * s], down[(spk_before + fifo_before + lc) * s ..][0 .. core * s]);
            @memcpy(self.fifo[fifo_before * dm ..][0 .. core * dm], chunk_embs[lc * dm ..][0 .. core * dm]);
            var fifo_n = fifo_before + core;
            if (fifo_n > d.fifo_len) {
                var pop = @max(d.update_period, core + fifo_before - d.fifo_len);
                pop = @min(pop, fifo_n);
                const cache_new = spk_before + pop;
                const cache_embs = try a.alloc(f32, cache_new * dm);
                @memcpy(cache_embs[0 .. spk_before * dm], self.spk[0 .. spk_before * dm]);
                @memcpy(cache_embs[spk_before * dm ..][0 .. pop * dm], self.fifo[0 .. pop * dm]);
                const cache_preds = try a.alloc(f32, cache_new * s);
                if (self.compressed) {
                    @memcpy(cache_preds[0 .. spk_before * s], self.spk_preds[0 .. spk_before * s]);
                } else {
                    @memcpy(cache_preds[0 .. spk_before * s], down[0 .. spk_before * s]);
                }
                @memcpy(cache_preds[spk_before * s ..][0 .. pop * s], self.fifo_preds[0 .. pop * s]);
                std.mem.copyForwards(f32, self.fifo[0 .. (fifo_n - pop) * dm], self.fifo[pop * dm .. fifo_n * dm]);
                std.mem.copyForwards(f32, self.fifo_preds[0 .. (fifo_n - pop) * s], self.fifo_preds[pop * s .. fifo_n * s]);
                fifo_n -= pop;
                if (cache_new > d.spkcache_len) {
                    try self.compress(a, d, cache_embs, cache_preds, cache_new, ref);
                } else {
                    @memcpy(self.spk[0 .. cache_new * dm], cache_embs);
                    @memcpy(self.spk_preds[0 .. cache_new * s], cache_preds);
                    self.spk_n = cache_new;
                }
            }
            self.fifo_n = fifo_n;
        }

        /// `_compress_spkcache`: the log scores, low-score disable, both `_boost_topk_scores`, the
        /// latest-frames boost, the +inf silence slots, top `spkcache_len` in frame order.
        fn compress(self: *State, a: std.mem.Allocator, d: config_file.Diar, embs: []const f32, preds: []const f32, frames: usize, ref: Ref) !void {
            const s = d.speakers;
            const dm = d.d_model;
            const per_spk = d.spkcache_len / s - 1; // spkcache_sil_frames_per_spk
            const strong: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * 0.75));
            const weak: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * 1.5));
            const min_pos: usize = @intFromFloat(@floor(@as(f64, @floatFromInt(per_spk)) * 0.5));
            const scores = try a.alloc(f64, frames * s);
            for (0..frames) |f| {
                var l1: f64 = 0;
                for (0..s) |sp| l1 += @log(@max(0.25, 1.0 - preds[f * s + sp]));
                for (0..s) |sp| {
                    const pv = preds[f * s + sp];
                    scores[f * s + sp] = @log(@max(0.25, pv)) - @log(@max(0.25, 1 - pv)) + l1 - @log(0.5);
                }
            }
            for (0..s) |sp| {
                var pos: usize = 0;
                for (0..frames) |f| {
                    if (scores[f * s + sp] > 0) pos += 1;
                }
                for (0..frames) |f| {
                    const speech = preds[f * s + sp] > 0.5;
                    if (!speech or (scores[f * s + sp] <= 0 and pos >= min_pos)) scores[f * s + sp] = -std.math.inf(f64);
                }
            }
            for (@min(d.spkcache_len, frames)..frames) |f| {
                for (0..s) |sp| scores[f * s + sp] += 0.05;
            }
            const idx = try a.alloc(usize, frames);
            for (0..s) |sp| {
                for ([_]struct { n: usize, scale: f64 }{ .{ .n = strong, .scale = 2 }, .{ .n = weak, .scale = 1 } }) |b| {
                    if (b.n == 0) continue;
                    for (idx, 0..) |*v, i| v.* = i;
                    const Ctx = struct { sc: []f64, s: usize, sp: usize };
                    std.mem.sort(usize, idx, Ctx{ .sc = scores, .s = s, .sp = sp }, struct {
                        fn lt(ctx: Ctx, aa: usize, bb: usize) bool {
                            return ctx.sc[aa * ctx.s + ctx.sp] > ctx.sc[bb * ctx.s + ctx.sp];
                        }
                    }.lt);
                    for (0..@min(b.n, frames)) |i| scores[idx[i] * s + sp] -= b.scale * @log(0.5);
                }
            }
            const per_spk_len = frames + 1;
            const flat = try a.alloc(f64, s * per_spk_len);
            for (0..s) |sp| {
                for (0..frames) |f| flat[sp * per_spk_len + f] = scores[f * s + sp];
                flat[sp * per_spk_len + frames] = std.math.inf(f64);
            }
            const order = try a.alloc(usize, flat.len);
            for (order, 0..) |*v, i| v.* = i;
            std.mem.sort(usize, order, flat, struct {
                fn lt(sc: []const f64, aa: usize, bb: usize) bool {
                    return sc[aa] > sc[bb];
                }
            }.lt);
            const top = order[0..d.spkcache_len];
            std.mem.sort(usize, top, {}, std.sort.asc(usize));
            for (top, 0..) |fi, i| {
                const f = fi % per_spk_len;
                const disabled = flat[fi] == -std.math.inf(f64) or f >= frames;
                if (disabled) {
                    @memcpy(self.spk[i * dm ..][0..dm], self.sil);
                    for (0..s) |sp| self.spk_preds[i * s + sp] = 0;
                } else {
                    @memcpy(self.spk[i * dm ..][0..dm], embs[f * dm ..][0..dm]);
                    for (0..s) |sp| self.spk_preds[i * s + sp] = preds[f * s + sp];
                }
            }
            self.spk_n = d.spkcache_len;
            self.compressed = true;
            _ = ref;
        }
    };
};

fn cosine(got: []const f32, want: []const f32) f64 {
    var dot: f64 = 0;
    var ga: f64 = 0;
    var wa: f64 = 0;
    for (got, want) |g, w| {
        dot += g * w;
        ga += @as(f64, g) * g;
        wa += @as(f64, w) * w;
    }
    return dot / @sqrt(ga * wa + 1e-30);
}

/// The clip's windows as `encodePe` tiles them for the PE fixture's knobs (pe_core 100 x sub 2 = 200
/// mel frames a core, contexts 12 each).
fn peWindows(a: std.mem.Allocator, c: config_file.Config, width: usize, encoded_len: usize) ![]const [6]usize {
    const p = c.perception;
    const sub = p.subsampling;
    const core_mel = c.pe_core * sub;
    const left_mel = c.pe_left * sub;
    const right_mel = c.pe_right * sub;
    const num = @max(1, (width + core_mel - 1) / core_mel);
    const windows = try a.alloc([6]usize, num);
    var out_row: usize = 0;
    for (windows, 0..) |*w, i| {
        const start = i * core_mel;
        const end = @min(start + core_mel, width);
        const cs = if (start >= left_mel) start - left_mel else 0;
        const ce = @min(end + right_mel, width);
        const core_rows = @min(encoded_len, (end + sub - 1) / sub) - (start + sub - 1) / sub;
        w.* = .{ cs, ce, core_rows, (start - cs) / sub, out_row, start - cs };
        out_row += core_rows;
    }
    return windows;
}

test "speaker: a clip's streaming Sortformer activity against the host reference" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const made = try loadPe(ar);
    const d = made.c.diar.?;
    const e = made.c.perception;
    const sub = e.subsampling;
    var spk = try speaker_file.Speaker.load(a, ar, &made.c, &made.checkpoint, e.d_model, 200);
    defer spk.deinit();
    // a clip: the PE windows over a synthetic normed mel (the mel itself is not exercised here —
    // `Speaker.activity` reads it ready-normed as `encodePe` writes)
    const width = 740; // mel frames: four windows' cores (200 each) under pe_core=200/2? — core_mel
    const valid = 700; // the last window's tail is a partial
    const mel = try ar.alloc(f32, width * e.features);
    var prng: std.Random.DefaultPrng = .init(17);
    for (mel) |*v| v.* = @floatCast(0.4 * prng.random().float(f64) - 0.2);
    @memset(mel[valid * e.features ..], 0);
    const encoded_len = @max(1, (valid + sub - 1) / sub);
    const windows = try peWindows(ar, made.c, width, encoded_len);
    var act = try cuda.Buffer.init(.bf16, &.{ encoded_len, d.speakers });
    defer act.deinit();
    try spk.activity(ar, mel, e.features, valid, windows, encoded_len, act.view);
    const got = try ar.alloc(u16, encoded_len * d.speakers);
    try cuda.download(act.view, std.mem.sliceAsBytes(got));
    // the host: the same windows' steps, downsampled and aligned to each window's core rows
    var state = try Ref.State.init(ar, d, .{ .c = made.c, .checkpoint = &made.checkpoint });
    const want = try ar.alloc(f32, encoded_len * d.speakers);
    @memset(want, 0);
    for (windows, 0..) |w, i| {
        const cs, const ce, const core_rows, const left_off, const out_row = .{ w[0], w[1], w[2], w[5], w[4] };
        const end = @min((i + 1) * made.c.pe_core * sub, width);
        const native = try (Ref{ .c = made.c, .checkpoint = &made.checkpoint }).step(ar, d, mel, e.features, cs, ce, valid, left_off, ce - end, i + 1 == windows.len, &state);
        for (0..core_rows) |r| {
            for (0..d.speakers) |s| {
                var sum: f64 = 0;
                for (0..sub) |j| sum += native[(r * sub + j) * d.speakers + s];
                want[(out_row + r) * d.speakers + s] = @floatCast(sum / @as(f64, @floatFromInt(sub)));
            }
        }
    }
    const got_f = try ar.alloc(f32, encoded_len * d.speakers);
    for (got_f, got) |*v, b| v.* = f32v(b);
    const cos = cosine(got_f, want);
    std.debug.print("speaker activity cosine {d:.6} over {d} rows x {d} spk, {d} windows (spkcache {d}, fifo {d})\n", .{ cos, encoded_len, d.speakers, windows.len, state.spk_n, state.fifo_n });
    try testing.expect(cos > 0.97);
}

test "speaker: the infusion's fuse beside asr_norm against the host" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const made = try loadPe(ar);
    const d = made.c.diar.?;
    var e = made.c.perception;
    e.d_model = 1280; // Published ASR width; the fusion reduction must cover more than one warp's tile.
    var spk = try speaker_file.Speaker.load(a, ar, &made.c, &made.checkpoint, e.d_model, 16);
    defer spk.deinit();
    // At speaker 1 and dimension D/2 the standard sinusoid is sin(1/100), independently of
    // the implementation's logarithm/exponential construction.
    const sinusoid = try ar.alloc(u16, d.speakers * e.d_model);
    try spk.diar_kernel.download(std.mem.sliceAsBytes(sinusoid));
    try testing.expectEqual(perception_file.bf16Of(@floatCast(@sin(@as(f64, 0.01)))), sinusoid[e.d_model + e.d_model / 2]);
    // synthetic states and activity
    const rows = 37;
    const s = d.speakers;
    var prng: std.Random.DefaultPrng = .init(23);
    const rand = prng.random();
    const states_h = try ar.alloc(u16, rows * e.d_model);
    for (states_h) |*v| v.* = perception_file.bf16Of(@floatCast(0.4 * rand.float(f64) - 0.2));
    const act_h = try ar.alloc(u16, rows * s);
    for (act_h) |*v| v.* = perception_file.bf16Of(@floatCast(rand.float(f64)));
    var states = try cuda.Buffer.init(.bf16, &.{ rows, e.d_model });
    defer states.deinit();
    try states.upload(std.mem.sliceAsBytes(states_h));
    var act = try cuda.Buffer.init(.bf16, &.{ rows, s });
    defer act.deinit();
    try act.upload(std.mem.sliceAsBytes(act_h));
    var out = try cuda.Buffer.init(.bf16, &.{ rows, e.d_model });
    defer out.deinit();
    const ref = Ref{ .c = made.c, .checkpoint = &made.checkpoint };
    // the host: y = asr_norm(x) + diar_norm(act>0.5) @ diar_kernel, all bf16-rounds as the kernel's
    const stored_aw = try ref.vec("perception.encoder.asr_norm.weight", ar);
    const stored_ab = try ref.vec("perception.encoder.asr_norm.bias", ar);
    const aw = try ar.alloc(f32, e.d_model);
    const ab = try ar.alloc(f32, e.d_model);
    for (aw, ab, 0..) |*w, *b, i| {
        w.* = stored_aw[i % stored_aw.len];
        b.* = stored_ab[i % stored_ab.len];
    }
    const nw = try ref.vec("perception.encoder.diar_norm.weight", ar);
    const nb = try ref.vec("perception.encoder.diar_norm.bias", ar);
    const x_f = try ar.alloc(f32, rows * e.d_model);
    for (x_f, states_h) |*v, b| v.* = f32v(b);
    const y_f = try ar.alloc(f32, rows * e.d_model);
    for (0..rows) |r| {
        Ref.norm(x_f, r * e.d_model, aw, ab, y_f);
        var feats: [8]f32 = undefined;
        for (0..s) |sp| feats[sp] = if (f32v(act_h[r * s + sp]) > 0.5) 1 else 0;
        Ref.norm(&feats, 0, nw, nb, &feats);
        for (0..e.d_model) |t| {
            var inf: f64 = 0;
            for (0..s) |sp| {
                const ang = @as(f64, @floatFromInt(sp)) / std.math.pow(f64, 10000.0, @as(f64, @floatFromInt(2 * (t / 2))) / @as(f64, @floatFromInt(e.d_model)));
                const kv: f32 = if (t % 2 == 0) bf(@floatCast(@sin(ang))) else bf(@floatCast(@cos(ang)));
                inf += feats[sp] * kv;
            }
            y_f[r * e.d_model + t] = bf(y_f[r * e.d_model + t] + 1.0 * @as(f32, @floatCast(inf)));
        }
    }
    const asr_norm: perception_file.Norm = blk: {
        var b: perception_file.Norm = undefined;
        b.weight = try uploadBf(ar, aw);
        b.bias = try uploadBf(ar, ab);
        break :blk b;
    };
    defer {
        var b = asr_norm;
        b.deinit();
    }
    try spk.fuse(states.view, act.view, &asr_norm, out.view);
    const got = try ar.alloc(u16, rows * e.d_model);
    try cuda.download(out.view, std.mem.sliceAsBytes(got));
    const got_f = try ar.alloc(f32, rows * e.d_model);
    for (got_f, got) |*v, b2| v.* = f32v(b2);
    const cos = cosine(got_f, y_f);
    std.debug.print("speaker fuse cosine {d:.6} over {d} rows\n", .{ cos, rows });
    try testing.expect(cos > 0.99);
    // a row's bits never depend on its neighbours: fuse a single row alone, the same bits
    var out1 = try cuda.Buffer.init(.bf16, &.{ 1, e.d_model });
    defer out1.deinit();
    try spk.fuse(states.span(5, 1), act.span(5, 1), &asr_norm, out1.view);
    const one = try ar.alloc(u16, e.d_model);
    try cuda.download(out1.view, std.mem.sliceAsBytes(one));
    try testing.expectEqualSlices(u16, got[5 * e.d_model .. 6 * e.d_model], one);
}

fn uploadBf(a: std.mem.Allocator, x: []const f32) !cuda.Buffer {
    const bits = try a.alloc(u16, x.len);
    defer a.free(bits);
    for (bits, x) |*v, f| v.* = perception_file.bf16Of(f);
    var b: cuda.Buffer = try .init(.bf16, &.{x.len});
    try b.upload(std.mem.sliceAsBytes(bits));
    return b;
}

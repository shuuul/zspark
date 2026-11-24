//! Host references for `zig/kernels/nemotron/`'s glue kernels: every output's bits as the kernel computes
//! them. From TensorFold 1.1.5 (792cdbab) `src/families/nemotron/glue_math.zig` and `glue_ref.zig`, the parts
//! covering norms.cu, route.cu, ops.cu, keyed.cu and mamba.cu (conv and scan; `convRows` went with its
//! kernel); `embed`'s (its kernel is dropped) and attention's come with their own slices. No device; `numeric.f32ToBf16` is not used — it keeps a NaN's sign
//! where `cvt.rn.bf16.f32` gives 0x7fff.

const std = @import("std");

pub fn fma(a: f32, b: f32, c: f32) f32 {
    return @mulAdd(f32, a, b, c);
}

pub fn bf16ToF32(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// cvt.rn.bf16.f32: round to nearest even, any NaN to 0x7fff.
pub fn f32ToBf16(v: f32) u16 {
    if (std.math.isNan(v)) return 0x7fff;
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// The value rounded through bf16.
pub fn rbf(v: f32) f32 {
    return bf16ToF32(f32ToBf16(v));
}

const fma_ = fma;
const bf = bf16ToF32;
const tobf = f32ToBf16;

const l2e: f32 = 0x1.715476p+0; // log2(e)
const ln2_hi: f32 = 0x1.62e4p-1; // 15 significant bits: j * ln2_hi is exact for |j| <= 255
const ln2_lo: f32 = 0x1.7f7d1cp-20;
const shifter: f32 = 0x1.8p+23; // adding it rounds to an integer

/// e^x within one ulp: x = j ln2 + f, a degree-7 Taylor polynomial in f, then 2^j in two exact factors.
pub fn exp(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    if (x > 104.0) return std.math.inf(f32);
    if (x < -104.0) return 0.0;
    const j = fma(x, l2e, shifter) - shifter;
    var f = fma(j, -ln2_hi, x);
    f = fma(j, -ln2_lo, f);
    var p: f32 = 0x1.a01a02p-13; // 1/7! .. 1/3!, each the nearest float
    p = fma(p, f, 0x1.6c16c2p-10);
    p = fma(p, f, 0x1.111112p-7);
    p = fma(p, f, 0x1.555556p-5);
    p = fma(p, f, 0x1.555556p-3);
    p = fma(p, f, 0.5);
    p = fma(p, f, 1.0);
    p = fma(p, f, 1.0);
    const i: i32 = @intFromFloat(j);
    const a: f32 = @bitCast(@as(u32, if (i > 0) 0x7f000000 else 0x02000000));
    const b: f32 = @bitCast(@as(u32, @intCast(if (i > 0) i << 23 else (i + 250) << 23)));
    return (p * a) * b;
}

/// ln(1 + u) for u in [0, 1], within one ulp: 2 atanh(t / (2 + t)) with the quotient carried in two floats.
pub fn log1pUnit(u: f32) f32 {
    const hi = u > 0.5;
    const t = if (hi) (u - 1.0) * 0.5 else u; // exact for u in (0.5, 1]: ln(1 + u) = ln2 + ln(1 + t)
    const s = 2.0 + t;
    const err = t - (s - 2.0); // 2 + t = s + err exactly
    const zh = t / s;
    const zl = (fma(-zh, s, t) - zh * err) / s; // t / (2 + t) = zh + zl to second order
    const z2 = zh * zh;
    var p: f32 = 0x1.745d18p-4; // 1/11 .. 1/3, each the nearest float
    p = fma(p, z2, 0x1.c71c72p-4);
    p = fma(p, z2, 0x1.24924ap-3);
    p = fma(p, z2, 0x1.99999ap-3);
    p = fma(p, z2, 0x1.555556p-2);
    const zz = zh + zh;
    const r = zz + fma(zz * z2, p, zl + zl);
    if (!hi) return r;
    return ln2_hi + (ln2_lo + r);
}

/// 1 / (1 + e^-x).
pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + exp(-x));
}

/// x / (1 + e^-x): SiLU with one rounding fewer than x * sigmoid(x).
pub fn silu(x: f32) f32 {
    return x / (1.0 + exp(-x));
}

/// max(v, 0) + ln(1 + e^-|v|), the dt softplus, without log(1 + e)'s loss for small e.
pub fn softplus(v: f32) f32 {
    return @max(v, 0.0) + log1pUnit(exp(-@abs(v)));
}

const nt = 256; // the norm kernels' threads a row

/// An xor butterfly over n lanes (n / 2 down to 1), each lane adding its partner to its own value.
fn butterfly(comptime T: type, comptime n: usize, v: [n]T) T {
    var a = v;
    var off: usize = n / 2;
    while (off > 0) : (off >>= 1) {
        var b: [n]T = undefined;
        for (0..n) |l| b[l] = a[l] + a[l ^ off];
        a = b;
    }
    return a[0];
}

/// A row's sum of squares as `threads` threads take it: element i on thread i % threads, warps summed in order.
fn sumSquares(x: []const f32, comptime threads: usize) f32 {
    var lane: [threads]f32 = @splat(0);
    for (x, 0..) |v, i| lane[i % threads] = fma(v, v, lane[i % threads]);
    var ss: f32 = undefined;
    for (0..threads / 32) |w| {
        const s = butterfly(f32, 32, lane[w * 32 ..][0..32].*);
        ss = if (w == 0) s else ss + s;
    }
    return ss;
}

fn invRms(ss: f32, n: usize, eps: f32) f32 {
    return 1.0 / @sqrt(ss / @as(f32, @floatFromInt(n)) + eps);
}

/// y = bf16((x * inv) * w), and each 64-group's sum of the stored values in order.
fn scaleStore(x: []const f32, inv: f32, w: []const u16, out: []u16, xs: []f32) void {
    for (x, w, out) |v, wv, *o| o.* = tobf((v * inv) * bf(wv));
    for (xs, 0..) |*s, g| {
        var acc: f32 = 0;
        for (out[g * 64 ..][0..64]) |y| acc = acc + bf(y);
        s.* = acc;
    }
}

/// tf_nemo_add_rmsnorm: h = bf16(x + r) unless r is null, then y and xs.
pub fn addRmsnorm(gpa: std.mem.Allocator, x: []const u16, r: ?[]const u16, w: []const u16, h: []u16, y: []u16, xs: []f32, rows: usize, d: usize, eps: f32) !void {
    const v = try gpa.alloc(f32, d);
    defer gpa.free(v);
    for (0..rows) |row| {
        for (0..d) |i| {
            v[i] = bf(x[row * d + i]);
            if (r) |rr| {
                h[row * d + i] = tobf(v[i] + bf(rr[row * d + i]));
                v[i] = bf(h[row * d + i]);
            }
        }
        scaleStore(v, invRms(sumSquares(v, nt), d, eps), w, y[row * d ..][0..d], xs[row * (d / 64) ..][0..(d / 64)]);
    }
}

/// tf_nemo_add_moe_norm: delta from the slots in order (routed by fma, shared added), h, then y and xs.
pub fn addMoeNorm(gpa: std.mem.Allocator, x: []const u16, y32: ?[]const f32, y16: ?[]const u16, wt: []const f32, w: []const u16, hn: []u16, out: []u16, xs: []f32, rows: usize, d: usize, eps: f32, nr: usize, ns: usize) !void {
    const v = try gpa.alloc(f32, d);
    defer gpa.free(v);
    for (0..rows) |row| {
        for (0..d) |i| {
            var routed: f32 = 0;
            var shared: f32 = 0;
            for (0..ns) |s| {
                const at = (row * ns + s) * d + i;
                const yv = if (y32) |a| a[at] else bf(y16.?[at]);
                if (s < nr) routed = fma(wt[row * ns + s], yv, routed) else shared = shared + yv;
            }
            hn[row * d + i] = tobf(bf(x[row * d + i]) + rbf(routed + shared));
            v[i] = bf(hn[row * d + i]);
        }
        scaleStore(v, invRms(sumSquares(v, nt), d, eps), w, out[row * d ..][0..d], xs[row * (d / 64) ..][0..(d / 64)]);
    }
}

/// tf_nemo_concat_norms: [rmsnorm(e) * we | rmsnorm(h) * wh] and its group sums.
pub fn concatNorms(gpa: std.mem.Allocator, e: []const u16, h: []const u16, we: []const u16, wh: []const u16, out: []u16, xs: []f32, rows: usize, d: usize, eps: f32) !void {
    const v = try gpa.alloc(f32, d);
    defer gpa.free(v);
    for (0..rows) |row| for (0..2) |part| {
        const src = if (part == 0) e else h;
        for (0..d) |i| v[i] = bf(src[row * d + i]);
        const g = d / 64;
        scaleStore(v, invRms(sumSquares(v, nt), d, eps), if (part == 0) we else wh, out[row * 2 * d + part * d ..][0..d], xs[row * 2 * g + part * g ..][0..g]);
    };
}

/// tf_nemo_group_rmsnorm: per group of gs, n = bf16(v * inv), out = bf16(w * n), and the 64-group sums.
pub fn groupRmsnorm(gpa: std.mem.Allocator, x: []const u16, w: []const u16, out: []u16, xs: []f32, rows: usize, xd: usize, groups: usize, eps: f32) !void {
    const gs = xd / groups;
    const v = try gpa.alloc(f32, gs);
    defer gpa.free(v);
    for (0..rows) |row| for (0..groups) |grp| {
        const at = row * xd + grp * gs;
        for (0..gs) |i| v[i] = bf(x[at + i]);
        const inv = invRms(sumSquares(v, 128), gs, eps);
        for (0..gs) |i| out[at + i] = tobf(bf(w[grp * gs + i]) * rbf(v[i] * inv));
        for (0..gs / 64) |g| {
            var acc: f32 = 0;
            for (out[at + g * 64 ..][0..64]) |o| acc = acc + bf(o);
            xs[at / 64 + g] = acc;
        }
    };
}

/// tf_nemo_router: PART[s][row][e], one fma chain over the slice's inputs in order.
pub fn router(x: []const u16, w: []const u16, part: []f32, rows: usize, d: usize, e: usize, sk: usize) void {
    const per = d / sk;
    for (0..sk) |s| for (0..rows) |r| for (0..e) |ex| {
        var acc: f32 = 0;
        for (s * per..(s + 1) * per) |k| acc = fma(bf(x[r * d + k]), bf(w[ex * d + k]), acc);
        part[(s * rows + r) * e + ex] = acc;
    };
}

/// tf_nemo_topk: sum slices in order, rank sigmoid scores plus bias (lowest id on ties), then shared slots.
pub fn topk(part: []const f32, bias: []const f32, idx: []i32, wt: []f32, rows: usize, scaling: f32, e: usize, sk: usize, top_k: usize, ns: usize, norm: bool) void {
    var score: [256]f32 = undefined;
    var sel: [256]f32 = undefined;
    for (0..rows) |r| {
        for (0..e) |ex| {
            var logit: f32 = 0;
            for (0..sk) |s| logit = logit + part[(s * rows + r) * e + ex];
            score[ex] = sigmoid(logit);
            const v = score[ex] + bias[ex];
            sel[ex] = if (std.math.isNan(v)) -std.math.inf(f32) else v;
        }
        var total: f32 = 0;
        var probs: [8]f32 = undefined;
        var ids: [8]i32 = undefined;
        for (0..top_k) |k| {
            var bi: usize = 0;
            for (1..e) |ex| if (sel[ex] > sel[bi]) {
                bi = ex;
            };
            probs[k] = score[bi];
            ids[k] = @intCast(bi);
            total = total + score[bi];
            sel[bi] = -std.math.inf(f32);
        }
        for (0..ns) |k| {
            idx[r * ns + k] = if (k < top_k) ids[k] else @intCast(e + k - top_k);
            wt[r * ns + k] = if (k >= top_k) 1.0 else if (norm) (probs[k] / (total + 1e-20)) * scaling else probs[k] * scaling;
        }
    }
}

/// tf_serial_feed: the sampled token into ids and history at pos + 1; meta becomes {pos + 1, parity ^ 1, 1}.
pub fn serialFeed(sampled: []const i32, ids: []i32, meta: []i32, history: []i32) void {
    const tok = sampled[0];
    const pos: usize = @intCast(meta[0]);
    ids[0] = tok;
    history[pos + 1] = tok;
    meta[0] += 1;
    meta[1] ^= 1;
    meta[2] = 1;
}

/// tf_plan_routed: per expert in id order, its routed picks' indices (member order kept), its items of at
/// most `tile` members, and counts = {items, experts used}. Picks at slots >= `routed` are skipped.
pub fn planRouted(picks: []const i32, members: []i32, items: []i32, counts: []i32, rows: usize, slots: usize, routed: usize, e: usize, tile: usize) void {
    var off: usize = 0;
    var ioff: usize = 0;
    var nused: usize = 0;
    for (0..e) |ex| {
        var c: usize = 0;
        for (0..rows * slots) |p| {
            if (p % slots < routed and picks[p] == @as(i32, @intCast(ex))) c += 1;
        }
        const tiles = if (c == 0) 0 else (c + tile - 1) / tile;
        for (0..tiles) |j| {
            const it = items[3 * (ioff + j) ..];
            it[0] = @intCast(ex);
            it[1] = @intCast(off + tile * j);
            it[2] = @intCast(@min(tile, c - tile * j));
        }
        var k = off;
        for (0..rows * slots) |p| {
            if (k >= off + c) break;
            if (p % slots < routed and picks[p] == @as(i32, @intCast(ex))) {
                members[k] = @intCast(p);
                k += 1;
            }
        }
        off += c;
        ioff += tiles;
        nused += @intFromBool(c > 0);
    }
    counts[0] = @intCast(ioff);
    counts[1] = @intCast(nused);
}

/// tf_rest_rows: y[row * row_mul + row_add][j] += r[j] . x[row], a column's dot by 32 lanes of 8-value chunks
/// (each chunk an fma chain), the lanes' sums by xor butterfly; y fp32, or bf16 with one rounding.
pub fn restRows(x: []const u16, x_stride: usize, r: []const u16, y32: ?[]f32, y16: ?[]u16, y_stride: usize, rows: usize, row_mul: usize, row_add: usize, n: usize, k: usize) void {
    for (0..n) |j| {
        for (0..rows) |rr| {
            const row = rr * row_mul + row_add;
            var lanes: [32]f32 = @splat(0);
            for (0..32) |lane| {
                var i = lane * 8;
                while (i < k) : (i += 256) {
                    var s: f32 = 0;
                    for (0..8) |q| s = fma(bf(r[j * k + i + q]), bf(x[row * x_stride + i + q]), s);
                    lanes[lane] += s;
                }
            }
            const v = butterfly(f32, 32, lanes);
            const at = row * y_stride + j;
            if (y32) |a| a[at] += v else y16.?[at] = tobf(bf(y16.?[at]) + v);
        }
    }
}

/// tf_nemo_keyed_greedy: rank 0 by (value desc, id asc) and, given `prob`, 1 / sum over the top k of
/// e^(v - v0) in f64.
pub fn keyedGreedy(vals: []const f32, ids: []const i64, out: []i32, prob: ?[]f32, rows: usize, count: usize, k: usize) void {
    for (0..rows) |r| {
        var v: [32]f64 = @splat(-std.math.inf(f64));
        var id: [32]i64 = @splat(1 << 40);
        for (0..count) |i| {
            v[i] = vals[r * count + i];
            id[i] = ids[r * count + i];
        }
        var e: [32]f64 = @splat(0);
        var top = -std.math.inf(f64);
        var rank: [32]usize = undefined;
        for (0..32) |i| {
            rank[i] = 0;
            for (0..count) |j| {
                if (v[j] > v[i] or (v[j] == v[i] and id[j] < id[i])) rank[i] += 1;
            }
            if (i < count and rank[i] == 0) out[r] = @intCast(id[i]);
            if (i < count and rank[i] < k) top = @max(top, v[i]);
        }
        const p = prob orelse continue;
        for (0..count) |i| e[i] = if (rank[i] < k) @exp(v[i] - top) else 0;
        p[r] = @floatCast(1.0 / butterfly(f64, 32, e));
    }
}

/// The conv taps in fma order, rounded to bf16, through SiLU, rounded again.
fn conv4(bias: f32, w: [4]f32, t0: f32, t1: f32, t2: f32, cur: f32) u16 {
    var acc = bias;
    acc = fma(w[0], t0, acc);
    acc = fma(w[1], t1, acc);
    acc = fma(w[2], t2, acc);
    acc = fma(w[3], cur, acc);
    return tobf(silu(rbf(acc)));
}

/// The Mamba layer's shapes as the kernels take them (Tri.Shape's fields).
pub const Shape = struct { proj: usize, xd: usize, cd: usize, heads: usize, dh: usize, groups: usize, rmax: usize };

/// tf_nemo_conv: kept rows of RAW[1 - parity] replay, then the window's rows; BASE commits after the last kept row.
pub fn conv(p: []const u16, base: []u16, raw: []u16, xc: []u16, cw: []const f32, cb: []const f32, parity: usize, pk: usize, rows: usize, s: Shape) void {
    const cd = s.cd;
    for (0..cd) |ch| {
        var t0 = bf(base[ch]);
        var t1 = bf(base[cd + ch]);
        var t2 = bf(base[2 * cd + ch]);
        const w = [4]f32{ cw[ch], cw[cd + ch], cw[2 * cd + ch], cw[3 * cd + ch] };
        for (0..pk + rows) |i| {
            const prev = i < pk;
            const cur = if (prev) bf(raw[((1 - parity) * s.rmax + i) * cd + ch]) else bf(p[(i - pk) * s.proj + s.xd + ch]);
            if (!prev) {
                const at = (parity * s.rmax + (i - pk)) * cd + ch;
                xc[at] = conv4(cb[ch], w, t0, t1, t2, cur);
                raw[at] = tobf(cur);
            }
            t0 = t1;
            t1 = t2;
            t2 = cur;
            if (i + 1 == pk) {
                base[ch] = tobf(t0);
                base[cd + ch] = tobf(t1);
                base[2 * cd + ch] = tobf(t2);
            }
        }
    }
}

/// tf_nemo_scan: per head and value row, the state replays the kept rows, then the window's; y as the kernel sums it.
pub fn scan(p: []const u16, xc: []const u16, dt: []f32, state: []f32, a: []const f32, dsk: []const f32, dtb: []const f32, y: []u16, parity: usize, pk: usize, rows: usize, lo: f32, hi: f32, s: Shape) void {
    const ds = 128;
    for (0..s.heads) |h| for (0..s.dh) |d| {
        const g = h / (s.heads / s.groups);
        var st: [ds]f32 = state[(h * s.dh + d) * ds ..][0..ds].*;
        var dt_row: f32 = 0;
        for (0..pk + rows) |i| {
            const prev = i < pk;
            const buf = if (prev) 1 - parity else parity;
            const row = if (prev) i else i - pk;
            const base = (buf * s.rmax + row) * s.cd;
            const dt_at = (buf * s.rmax + row) * s.heads + h;
            if (prev) dt_row = dt[dt_at] else {
                const v = bf(p[row * s.proj + s.xd + s.cd + h]) + dtb[h];
                dt_row = @min(@max(softplus(v), lo), hi);
                if (d == 0) dt[dt_at] = dt_row;
            }
            const x = bf(xc[base + h * s.dh + d]);
            const da = exp(a[h] * dt_row);
            const xdt = x * dt_row;
            var outs: [4]f32 = undefined;
            for (0..4) |q| {
                var acc = [4]f32{ 0, 0, 0, 0 };
                for (0..8) |j| for (0..4) |c| {
                    const n = 4 * (4 * j + q) + c;
                    st[n] = fma(xdt, bf(xc[base + s.xd + g * ds + n]), st[n] * da);
                    acc[c] = fma(st[n], bf(xc[base + s.xd + s.groups * ds + g * ds + n]), acc[c]);
                };
                outs[q] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            }
            if (i + 1 == pk) state[(h * s.dh + d) * ds ..][0..ds].* = st;
            if (prev) continue;
            const out = (outs[0] + outs[1]) + (outs[2] + outs[3]);
            const yv = rbf(fma(x, dsk[h], out));
            const gz = rbf(silu(bf(p[row * s.proj + h * s.dh + d])));
            y[row * s.xd + h * s.dh + d] = tobf(gz * yv);
        }
    };
}

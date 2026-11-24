//! A long clip cut into pieces at its quietest points, as vLLM cuts a transcription's audio past the model's
//! `max_audio_clip_s` (spec 142 WS-06): `vllm/multimodal/audio.py`'s `split_audio` and `find_split_point` (the fork
//! `local-inference-lab/vllm` 8244d1c44 and upstream main a6c898e2 alike; SGLang's `audio_chunking.py` copies them
//! verbatim). The pieces are contiguous and do not overlap: the "overlap" second is only the region at the end
//! of each `max` window where the cut may land. In it, the RMS energy of strides of `window` samples is taken and
//! the cut is the start of the first quietest one; the loop bound leaves the region's last stride unread, as
//! vLLM's does. The last piece is what remains once a window would reach the end.
//!
//! The energy is numpy's on float32 samples: `(w ** 2).mean() ** 0.5`, the squares summed pairwise in f32 as
//! numpy's `pairwise_sum` does, divided in f32, then the root (`@sqrt`, where numpy's scalar power takes
//! `powf(x, 0.5)`: the two can part by an ulp, which changes a cut only when two strides' energies are a few
//! ulps apart).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// vLLM's `SpeechToTextConfig` defaults: `overlap_chunk_second` and `min_energy_split_window_size` (100 ms).
pub const overlap_seconds = 1;
pub const energy_window = 1600;

/// The pieces' ends, in samples: piece i is `[ends[i - 1] (0 for the first), ends[i])`, the last ending at
/// `n`. One piece, `n`, where `n` is at most `max` samples.
pub fn cuts(a: Allocator, samples: []const f32, rate: u32, max_seconds: f64, overlap: f64, window: usize) Allocator.Error![]usize {
    const chunk: usize = @intFromFloat(@as(f64, @floatFromInt(rate)) * max_seconds);
    const over: usize = @intFromFloat(@as(f64, @floatFromInt(rate)) * overlap);
    var ends: std.ArrayList(usize) = .empty;
    errdefer ends.deinit(a);
    const n = samples.len;
    var i: usize = 0;
    while (i < n) {
        if (i + chunk >= n) {
            try ends.append(a, n);
            break;
        }
        const start = i + chunk - over;
        const end = @min(i + chunk, n);
        var split = quietest(samples, start, end, window);
        if (split <= i) split = @min(i + chunk, n);
        try ends.append(a, split);
        i = split;
    }
    return ends.toOwnedSlice(a);
}

/// `find_split_point`: the start of the quietest `window`-sample stride of `samples[start..end]`, the first where
/// two are as quiet; `start` where no stride is read.
pub fn quietest(samples: []const f32, start: usize, end: usize, window: usize) usize {
    const len = end - start;
    var least = std.math.inf(f32);
    var at = start;
    var i: usize = 0;
    // python's range(0, len - window, window): nothing where len - window is not positive
    while (len > window and i < len - window) : (i += window) {
        const energy = rms(samples[start + i ..][0..window]);
        if (!std.math.isNan(energy) and energy < least) {
            at = start + i;
            least = energy;
        }
    }
    return at;
}

/// numpy's `(w ** 2).mean() ** 0.5` on float32.
pub fn rms(w: []const f32) f32 {
    const sum = pairwiseSquares(w);
    const mean: f32 = sum / @as(f32, @floatFromInt(w.len));
    return @sqrt(mean);
}

/// numpy's `pairwise_sum` (`loops_utils.h.src`) in f32 over the squares: under 8 a running sum; to 128 eight
/// running sums over a stride of 8, added in a tree, the rest after; past that the halves (the first a multiple
/// of 8) each so and added.
fn pairwiseSquares(w: []const f32) f32 {
    const n = w.len;
    if (n < 8) {
        var res: f32 = 0;
        for (w) |x| res += x * x;
        return res;
    }
    if (n <= 128) {
        var r: [8]f32 = undefined;
        for (&r, 0..) |*v, j| v.* = w[j] * w[j];
        var i: usize = 8;
        while (i < n - n % 8) : (i += 8) {
            for (&r, 0..) |*v, j| v.* += w[i + j] * w[i + j];
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < n) : (i += 1) res += w[i] * w[i];
        return res;
    }
    var half = n / 2;
    half -= half % 8;
    return pairwiseSquares(w[0..half]) + pairwiseSquares(w[half..]);
}

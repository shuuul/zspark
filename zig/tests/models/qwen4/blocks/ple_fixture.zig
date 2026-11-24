//! What `ple_test.zig` and `ple_gather_test.zig` call: a view read back and an n-gram table's constants.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const qwen4_config = @import("../../../../src/models/qwen4/config.zig");
const Config = qwen4_config.Config;
const Ngram = qwen4_config.Ngram;

pub fn fetch(comptime T: type, gpa: Allocator, view: cuda.View) ![]T {
    try cuda.synchronize();
    const host = try gpa.alloc(T, view.count());
    errdefer gpa.free(host);
    const part: cuda.Buffer = .{ .view = view };
    try part.download(std.mem.sliceAsBytes(host));
    return host;
}

/// An n-gram table's constants from the config values that define them.
pub fn ngramOf(arena: Allocator, vocab: u32, ngram_size: u32, heads_per_ngram: u32, base: u32, divisor: u32, seed: u64, ple_dim: u32, ple_index: usize) !Ngram {
    var config: Config = undefined;
    config.vocab = vocab;
    config.ngram_size = ngram_size;
    config.heads_per_ngram = heads_per_ngram;
    config.ngram_base = base;
    config.ngram_divisor = divisor;
    config.seed = seed;
    config.ple_dim = ple_dim;
    return .init(arena, config, ple_index);
}

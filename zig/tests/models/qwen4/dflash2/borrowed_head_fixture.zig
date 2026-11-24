//! Spec 155's reference is the old download-and-reinitialize head construction. Keep its outputs beside
//! the borrowed path: every block's logits, candidates, rescored values, projected rows and draft trail.
const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const weights = @import("../../../../src/decode/dflash2/weights.zig");
const pass = @import("../../../../src/decode/dflash2/pass.zig");

pub const Evidence = struct {
    legacy: bool,
    mixed: bool,
    hash: std.crypto.hash.sha2.Sha256 = .init(.{}),
    copies: [2]?weights.Linear = @splat(null),
    proposers: [2]?weights.Linear = @splat(null),
    blocks: usize = 0,

    pub fn install(self: *Evidence, d: anytype, arena: std.mem.Allocator, gpa: std.mem.Allocator) !void {
        const mains = d.mains();
        var heads: [2]weights.Linear = undefined;
        var widened: [2]cuda.View = undefined;
        for (mains, 0..) |m, r| {
            if (!self.legacy and !self.mixed) continue;
            const stored: cuda.Buffer = .{ .view = m.rows };
            const bytes = try gpa.alloc(u8, stored.bytes());
            defer gpa.free(bytes);
            try cuda.synchronize();
            try stored.download(bytes);
            const rows: cuda.Scratch.Rows = .{ .pool = m.head.bf16.scratch, .count = pass.batch };
            if (self.legacy) {
                self.copies[r] = try .init(arena, gpa, &.{bytes}, d.c.hidden, .bf16, rows);
                widened[r] = self.copies[r].?.bf16.weight.view;
                heads[r] = self.copies[r].?;
            }
            if (self.mixed) {
                self.proposers[r] = try .init(arena, gpa, &.{bytes}, d.c.hidden, .four, rows);
                heads[r] = self.proposers[r].?;
            }
        }
        d.heads = if (self.legacy or self.mixed) heads else null;
        d.widened = if (self.legacy) widened else null;
    }

    pub fn deinit(self: *Evidence) void {
        for (&self.proposers) |*one| if (one.*) |*head| head.deinit();
        for (&self.copies) |*one| if (one.*) |*head| head.deinit();
    }

    pub fn capture(self: *Evidence, d: anytype) !void {
        if (d.trailed == 0) return;
        try cuda.synchronize();
        self.blocks += 1;
        try self.buffer(d.pair.tops);
        for (d.pair.ranks) |rank| {
            try self.buffer(rank.logits);
            try self.buffer(rank.selected);
        }
        self.hash.update(std.mem.sliceAsBytes(d.half_values));
        self.hash.update(std.mem.sliceAsBytes(d.half_tokens));
        self.hash.update(std.mem.sliceAsBytes(d.values));
        self.hash.update(std.mem.sliceAsBytes(d.projected));
        for (d.trail[0..d.trailed]) |pick| {
            self.hash.update(std.mem.asBytes(&pick.token));
            self.hash.update(std.mem.asBytes(&pick.p));
        }
        self.hash.update(std.mem.sliceAsBytes(d.noised[0..d.trailed]));
    }

    fn buffer(self: *Evidence, value: cuda.Buffer) !void {
        const bytes = try std.testing.allocator.alloc(u8, value.bytes());
        defer std.testing.allocator.free(bytes);
        try value.download(bytes);
        self.hash.update(bytes);
    }

    pub fn digest(self: *Evidence) [32]u8 {
        var out: [32]u8 = undefined;
        self.hash.final(&out);
        return out;
    }
};

//! What the tests of replies decoded several at once share (`hall_test.zig`, `pool_test.zig`,
//! `hall_pack_test.zig`): cases served through a hall with what each reply was (`serve`), the random model
//! with its MTP layer and a hall over it (`Fixture`), and the two samplings.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const sampling = @import("../../../../src/decode/sampling.zig");
const hall_file = @import("../../../../src/models/glm5/pool/hall.zig");
const layout = @import("../../../../src/models/glm5/layers/layout.zig");
const mtp = @import("../../../../src/models/glm5/mtp/mtp.zig");
const random_file = @import("../../../../src/models/glm5/checks/random.zig");
const shared = @import("../testing.zig");

const fx = random_file.fx;
const Hall = hall_file.HallOf(layout.world);

/// A reply asked in a test: its request, and the round it is offered from.
pub const Case = struct {
    request: hall_file.Request,
    start: usize = 0,
};

/// What the cases' replies were: each one's tokens, how it ended, the slot it took and what the slot held.
pub const Served = struct {
    tokens: []std.ArrayList(i32),
    ended: []?hall_file.Ended,
    admitted: []?hall_file.Admitted,
    rounds: usize = 0,
    /// The most streams live in one round.
    beside: usize = 0,

    fn init(arena: Allocator, count: usize) !Served {
        const self: Served = .{ .tokens = try arena.alloc(std.ArrayList(i32), count), .ended = try arena.alloc(?hall_file.Ended, count), .admitted = try arena.alloc(?hall_file.Admitted, count) };
        @memset(self.tokens, .empty);
        @memset(self.ended, null);
        @memset(self.admitted, null);
        return self;
    }
};

/// Every case through `hall`, each offered from its round on and admitted when a slot is free, in order.
pub fn serve(arena: Allocator, hall: anytype, cases: []const Case) !Served {
    var served: Served = try .init(arena, cases.len);
    const Sink = struct {
        arena: Allocator,
        served: *Served,
        /// The case each slot decodes.
        whose: []usize,
        left: std.ArrayList(usize) = .empty,

        pub fn tokens(self: *@This(), slot: usize, given: []const i32) void {
            self.served.tokens[self.whose[slot]].appendSlice(self.arena, given) catch @panic("out of memory");
        }

        pub fn ended(self: *@This(), slot: usize, how: hall_file.Ended) void {
            self.served.ended[self.whose[slot]] = how;
            self.left.append(self.arena, slot) catch @panic("out of memory");
        }
    };
    var sink: Sink = .{ .arena = arena, .served = &served, .whose = try arena.alloc(usize, hall.streams.len) };
    var next: usize = 0;
    var done: usize = 0;
    while (done < cases.len) : (served.rounds += 1) {
        while (next < cases.len and cases[next].start <= served.rounds) : (next += 1) {
            const admitted = try hall.admit(cases[next].request) orelse break;
            served.admitted[next] = admitted;
            sink.whose[admitted.slot] = next;
        }
        served.beside = @max(served.beside, hall.live());
        if (!try hall.round(&sink)) continue;
        for (sink.left.items) |slot| hall.retire(slot);
        done += sink.left.items.len;
        sink.left.clearRetainingCapacity();
        try testing.expect(served.rounds < 10_000);
    }
    return served;
}

/// The random model with its MTP layer, a sampler and the head: `mtp_test.zig`'s, with a hall over it.
pub const Fixture = struct {
    fixture: shared.Fixture,
    loop: shared.Loop,
    pair: mtp.Pair,
    sampler: sampling.Sampler,
    head: mtp.Head,
    logits: []u16,

    pub fn init(self: *Fixture, seed: u64, tokens: usize, capacity: usize) !void {
        const gpa = testing.allocator;
        try self.fixture.init(seed, tokens);
        errdefer self.fixture.deinit();
        self.loop = try self.fixture.loop(capacity);
        errdefer self.loop.deinit();
        self.pair = try .init(self.fixture.arena_state.allocator(), gpa, self.fixture.source, &self.fixture.c, self.loop.m, capacity, self.fixture.rows());
        errdefer self.pair.deinit();
        self.sampler = try .init(gpa, fx.vocab);
        errdefer self.sampler.deinit(gpa);
        self.head = try .init(gpa, self.loop.m, &self.pair, &self.loop.b, &self.sampler, greedy, fx.vocab);
        self.logits = try self.fixture.arena_state.allocator().alloc(u16, fx.vocab);
    }

    pub fn deinit(self: *Fixture) void {
        self.head.deinit();
        self.sampler.deinit(testing.allocator);
        self.pair.deinit();
        self.loop.deinit();
        self.fixture.deinit();
    }

    pub fn hall(self: *Fixture, slots: usize, capacity: usize, drafted: bool) !Hall {
        return .init(testing.allocator, self.loop.m, &self.loop.b, slots, capacity, fx.vocab, &self.sampler, self.logits, if (drafted) .{ .mtp = &self.head } else .none);
    }
};

pub const greedy: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };

pub fn sampled(seed: u64) sampling.Sampling {
    return .{ .seed = seed, .temperature = 1.4, .top_k = 40, .top_p = 0.95, .min_p = 0 };
}

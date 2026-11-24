//! `ranks-check DIR reference=REF [layers=L] [drafter=DIR] [kv=fp8] [rank=R at=ADDR:PORT]` where REF is the vLLM container's (spec 107
//! WS-02: `NAME.json` and `score.json` from `tools/reference/vllm_reference_ranks.sh`, no TensorFold export):
//! FP4-Local held to vLLM at `--tensor-parallel-size 2`. The whole model is a share a Spark; its first layers
//! may also run with both ranks in one process, and print the digest a pair of the same layers prints, so every
//! bit over NCCL is the loopback's. Both ranks of a pair decode alike (`session.zig`'s `alike`) and run every
//! step: the first case's prompt as one window against its serial steps, bit for bit; each case's greedy and
//! sampled replies drafted by the MTP layer, or by DFlash2 with `drafter=`, against the serial ones, bit for bit; the greedy reply against
//! vLLM's, leaving it only where the serial choice along vLLM's own tokens does, and there at a tie of vLLM's
//! (`tie`); then the score text in windows of `score_window`, a prompt's form (spec 115), its losses against vLLM's
//! two runs (`decode/vllm_reference.zig`). Against vLLM only the whole model is judged; the first layers' figures
//! are printed. With `kv=fp8` the DSA caches hold FP8 records (`--kv-dtype fp8`, the card's `--kv-cache-dtype fp8`
//! and serve's default), judged as under bf16.

const std = @import("std");
const units = @import("../../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const sampling = @import("../../../decode/sampling.zig");
const vllm = @import("../../../decode/vllm_reference.zig");
const layout = @import("../layers/layout.zig");
const session_file = @import("../session/session.zig");
const dsa = @import("../layers/dsa.zig");
const common = @import("../commands/common.zig");
const rig = @import("../dflash2/rig.zig");
const numeric = @import("../../../weights/numeric.zig");

/// A greedy reply vLLM gave (`NAME.json`; the tokens its runs agree on), and the tokens of it zspark's begins with, at
/// least, where its serial choice along vLLM's tokens does not part earlier at a tie; past them it may part anywhere.
const Case = vllm.Case;

/// Measured 2026-10-18 (spec 107 WS-02, zspark 778e99a against the card's image `sha256:03a956544e9f…3595` at
/// TP 2): `primes` every token vLLM's; `sky` and `sky-long` (one prompt) vLLM's to token 30, where vLLM's first
/// two are 13 at -0.5769 and 11 at -0.8269 and zspark takes 11; along vLLM's tokens 63 of 64 and 245 of 251
/// choices zspark's. vLLM's own two runs of `sky-long` part at 251 (572 and 10444, 0.25 apart). Since the experts'
/// scale fold (fc4d8be5, zspark 5c72d453, spec 107 WS-04) zspark takes 13 at 30: `sky` all 64 vLLM's, `sky-long` to
/// token 119, where vLLM has 576 at -0.4516 and 1446 at -1.5766 and zspark's logits are 19.75 and 20.625 (no tie of
/// either), along vLLM's tokens 246 of 251 choices zspark's. vLLM at its image default (W4A4, spec 114 WS-06): its
/// runs agree on the first 15 of `sky` and `sky-long` and 17 of `primes` (where the A16 reply's 3681 and 7772 lie
/// near), each case held to those; zspark's W4A4 replies begin with all of them.
const cases = [_]Case{
    .{ .name = "sky", .least = 30 },
    .{ .name = "primes", .least = 32 },
    .{ .name = "sky-long", .least = 30 },
};

/// Where zspark's reply may leave vLLM's: where vLLM's token and zspark's lie within this of each other, in
/// vLLM's logprobs or in zspark's own logits (a near-tie either side, which a bf16 rounding of the other settles).
/// Twice vLLM's own spread, the score text's mean |run - run| a token, 0.148 (measured as `cases`' notes say).
const tie = 0.30;
/// The score text's nll a token, zspark's against vLLM's runs' mean, at most this apart: measured -0.0038
/// (2.22185 against 2.22567 and 2.22472), a token's mean |zspark - vLLM| 0.132. At vLLM's image default (W4A4,
/// spec 114): 2.22273 against 2.22802 and 2.22879; zspark 0.2349 and 0.2338 from the runs, 0.2334 between them, so
/// zspark is held as a third run (`vllm.Sums.exchangeable`). At the card's command (spec 115, FP8 caches, a
/// prompt's form): 2.22407 against 2.22877 and 2.22805.
const nll_apart = 0.02;
/// The least distance a token zspark is held to, as vLLM's runs at the card's command are from its own runs in other
/// forms of FP4-Local (spec 115 WS-04, the score text, 2026-10-19): A16 experts 0.2005 to 0.2028, image-default W4A4
/// 0.2230 to 0.2304, against the card's runs 0.1272 apart. zspark's A4 prompt form is the card's, read in its
/// source, run on zspark's own kernels: as far from the card's kernel as vLLM's other kernels are. The rule is
/// the forms (`forward.zig` `card_bf16_rows`); this bound and `nll_apart` catch what departs from them.
const across = 0.2304;

/// The windows' rows: the score text's, a prompt's window of more than `forward.card_bf16_rows`, so its rows take the
/// quantized forms the card's vLLM scored them in (spec 115); the cases' prompts are shorter.
const rows = score_window;
const score_window = 512;
comptime {
    std.debug.assert(score_window > @import("../layers/forward.zig").card_bf16_rows);
}
/// The score text and the longest reply after its prompt.
const capacity = 8192;

const greedy_settings = vllm.greedy;
const sampled_settings = vllm.sampled;

/// Whether `reference_path` holds vLLM's files rather than TensorFold's export.
pub fn holds(io: std.Io, reference_path: []const u8) bool {
    return vllm.holds(io, reference_path, "primes.window.f32.safetensors");
}

fn Check(comptime held: usize) type {
    return struct {
        const Self = @This();
        const Session = session_file.SessionOf(held);
        const H = Session.H;

        arena: Allocator,
        stdout: *std.Io.Writer,
        session: *Session,
        /// The whole model: held to vLLM. Its first layers are only printed beside it.
        judged: bool,
        /// What drafts: "the MTP layer" or "DFlash2".
        drafts: []const u8,
        digest: std.hash.Wyhash = .init(107),
        failures: usize = 0,

        fn who(self: *const Self) []const u8 {
            return if (Session.paired) (if (self.session.first == 0) "rank 0" else "rank 1") else "both ranks";
        }

        fn fail(self: *Self, comptime format: []const u8, args: anytype) !void {
            self.failures += 1;
            try self.stdout.print("{s}: FAILED: " ++ format ++ "\n", .{self.who()} ++ args);
            try self.stdout.flush();
        }

        /// Against vLLM: a failure on the whole model over bf16 caches, a line on its first layers or over FP8.
        fn apart(self: *Self, comptime format: []const u8, args: anytype) !void {
            if (self.judged) return self.fail(format, args);
            try self.say("not judged: " ++ format, args);
        }

        fn say(self: *Self, comptime format: []const u8, args: anytype) !void {
            try self.stdout.print("{s}: " ++ format ++ "\n", .{self.who()} ++ args);
            try self.stdout.flush();
        }

        /// The prompt as one window, then as serial steps: every row's logits the same bits.
        fn window(self: *Self, prompt: []const i32) !void {
            const s = self.session;
            const vocab = s.logits.len;
            const whole = try self.arena.alloc(u16, prompt.len * vocab);
            try s.state.reset();
            try H.forwardScoring(s.m, &s.state, &s.b, prompt, .all);
            for (0..prompt.len) |r| try H.logitsRow(s.b, r, whole[r * vocab ..][0..vocab]);
            try s.state.reset();
            var differing: usize = 0;
            for (prompt, 0..) |_, r| {
                try H.forwardScoring(s.m, &s.state, &s.b, prompt[r..][0..1], .all);
                try H.logitsRow(s.b, 0, s.logits);
                differing += @intFromBool(!std.mem.eql(u16, s.logits, whole[r * vocab ..][0..vocab]));
                try H.commit(s.m, &s.state, s.b, prompt[r..][0..1], 1);
            }
            self.digest.update(std.mem.sliceAsBytes(whole));
            try self.say("the prompt's {d} rows as one window: {d} of them other bits as serial steps", .{ prompt.len, differing });
            if (differing != 0) try self.fail("a window's rows are not its serial steps'", .{});
        }

        fn case(self: *Self, one: Case, theirs: vllm.Reply) !void {
            const s = self.session;
            // what vLLM's runs agree on: its kernels' order is not fixed, and a long reply may part at a tie of its own
            var agreed = theirs.runs[0].tokens.len;
            for (theirs.runs[1..]) |other| agreed = @min(agreed, std.mem.indexOfDiff(i32, other.tokens, theirs.runs[0].tokens) orelse agreed);
            const expected = theirs.runs[0].tokens[0..agreed];
            if (agreed < theirs.runs[0].tokens.len) try self.say("vLLM {s}: its runs agree on the first {d} of {d} tokens, held to those", .{ one.name, agreed, theirs.runs[0].tokens.len });
            // a reference whose runs part early (W4A4's quantized activations) holds zspark to what they agree on
            const least = @min(one.least, agreed);
            if (theirs.prompt.len > rows or theirs.prompt.len + expected.len > capacity) return self.fail("{s}: vLLM's case does not fit the check", .{one.name});
            const serial = try self.arena.alloc(i32, expected.len);
            const out = try self.arena.alloc(i32, expected.len);
            var greedy: []const i32 = &.{};
            for ([_]sampling.Sampling{ greedy_settings, sampled_settings }) |settings| {
                const how = if (settings.greedy()) "greedy" else "sampled";
                const base = try s.reply(theirs.prompt, settings, .{ .eos = &.{}, .fresh = true, .drafts = false }, serial);
                const done = try s.reply(theirs.prompt, settings, .{ .eos = &.{}, .fresh = true }, out);
                self.digest.update(std.mem.sliceAsBytes(serial[0..base.made]));
                self.digest.update(std.mem.asBytes(&[_]u64{ done.stats.rounds, done.stats.accepted, done.stats.drafted }));
                const same = std.mem.eql(i32, out[0..done.made], serial[0..base.made]);
                try self.say("{s} {s}: serial {d} tokens, {d:.1} ms a token; drafted by {s} {s}, {d} rounds, {d} of {d} drafts kept, {d:.1} ms a token", .{ one.name, how, base.made, ms(base.rest_ns, base.made), self.drafts, if (same) "the serial tokens" else "OTHER TOKENS", done.stats.rounds, done.stats.accepted, done.stats.drafted, ms(done.rest_ns, done.made) });
                if (!same) try self.fail("{s} {s}: the drafted reply is not the serial one", .{ one.name, how });
                if (settings.greedy()) greedy = try self.arena.dupe(i32, serial[0..base.made]);
            }
            // along vLLM's reply: zspark leaves it where its serial choice after vLLM's tokens does
            const along = try self.forced(theirs.prompt, expected);
            var same: usize = 0;
            while (same < greedy.len and same < expected.len and greedy[same] == expected[same]) same += 1;
            try self.say("vLLM {s}: the first {d} of {d} tokens are vLLM's; after its tokens {d} of {d} choices are zspark's own", .{ one.name, same, greedy.len, along.agree, expected.len });
            if ((along.first orelse greedy.len) != same) return self.fail("{s}: the reply leaves vLLM's elsewhere than the serial choice along vLLM's tokens", .{one.name});
            if (along.first) |at| {
                const want = vllm.logprobOf(theirs, at, expected[at]) orelse return self.apart("{s}: vLLM's token is not among its own top", .{one.name});
                const mine = vllm.logprobOf(theirs, at, greedy[at]) orelse return self.apart("{s}: zspark's token {d} at {d} is not among vLLM's top", .{ one.name, greedy[at], at });
                try self.say("vLLM {s}: at token {d} vLLM's {d} at {d:.4}, zspark's {d} at {d:.4}; zspark's logits {d:.4} and {d:.4}", .{ one.name, at, expected[at], want, greedy[at], mine, along.theirs, along.own });
                if (at < least and want - mine > tie and along.own - along.theirs > tie) try self.apart("{s}: the reply leaves vLLM's before token {d} at no tie of vLLM's or of zspark's", .{ one.name, least });
            } else if (same < least) try self.apart("{s}: the reply leaves vLLM's before token {d}", .{ one.name, least });
        }

        /// `first`'s logits, zspark's: its own choice's and vLLM's token's.
        const Forced = struct { agree: usize = 0, first: ?usize = null, own: f32 = 0, theirs: f32 = 0 };

        /// The greedy choice at each of vLLM's tokens, fed vLLM's tokens before it: a step a token.
        fn forced(self: *Self, prompt: []const i32, theirs: []const i32) !Forced {
            const s = self.session;
            try s.state.reset();
            try H.forwardScoring(s.m, &s.state, &s.b, prompt, .last);
            try H.logitsRow(s.b, 0, s.logits);
            try H.commit(s.m, &s.state, s.b, prompt, prompt.len);
            var result: Forced = .{};
            for (theirs, 0..) |want, i| {
                const token = try sampling.argmax(s.logits);
                if (token == want) result.agree += 1 else if (result.first == null) {
                    result.first = i;
                    result.own = numeric.f32FromBf16(s.logits[@intCast(token)]);
                    result.theirs = numeric.f32FromBf16(s.logits[@intCast(want)]);
                }
                if (i + 1 == theirs.len) break;
                const step = [1]i32{want};
                try H.forwardScoring(s.m, &s.state, &s.b, &step, .last);
                try H.logitsRow(s.b, 0, s.logits);
                try H.commit(s.m, &s.state, s.b, &step, 1);
            }
            self.digest.update(std.mem.asBytes(&[_]u64{ result.agree, result.first orelse theirs.len }));
            return result;
        }

        /// The score text in windows of `score_window`, every row's logits, against vLLM's prompt logprobs.
        fn score(self: *Self, scored: vllm.Scored) !void {
            const s = self.session;
            if (scored.runs.len < 2 or scored.prompt.len > capacity) return self.fail("vLLM's score does not fit the check", .{});
            const tokens = scored.prompt;
            try s.state.reset();
            var sums: vllm.Sums = .{};
            var at: usize = 0;
            while (at < tokens.len) {
                const count = @min(score_window, tokens.len - at);
                const chunk = tokens[at..][0..count];
                try H.forwardScoring(s.m, &s.state, &s.b, chunk, .all);
                for (0..count) |r| {
                    const next = at + r + 1;
                    if (next >= tokens.len) break;
                    try H.logitsRow(s.b, r, s.logits);
                    self.digest.update(std.mem.sliceAsBytes(s.logits));
                    sums.add(scored, s.logits, next);
                }
                try H.commit(s.m, &s.state, s.b, chunk, count);
                at += count;
            }
            try self.stdout.print("{s}: ", .{self.who()});
            try sums.print(self.stdout);
            try self.stdout.flush();
            try self.say("score: zspark from either run at most {d:.4} a token, the runs' own distance or vLLM's across forms, and two standard errors", .{sums.bound(across)});
            if (!sums.exchangeable(nll_apart, across)) try self.apart("the score text is outside vLLM's bounds", .{});
        }
    };
}

fn ms(ns: u64, made: usize) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(@max(made, 2) - 1));
}

/// `layers` of the model: the whole, a rank a process (`place`); the first ones may be both ranks here too.
pub fn run(init: std.process.Init, stdout: *std.Io.Writer, opened: *common.Opened, drafter: ?*const rig.Opened, place: join_file.Place, reference_path: []const u8, layers: usize, kv: dsa.Storage) !u8 {
    const paired = place.rank != null or place.at != null;
    if (paired) return check(1, init, stdout, opened, drafter, place, reference_path, layers, kv);
    if (layers == opened.config.layers) {
        std.log.err("ranks-check: the whole model is a rank a Spark (rank=R at=ADDR:PORT)", .{});
        return 1;
    }
    return check(layout.world, init, stdout, opened, drafter, place, reference_path, layers, kv);
}

fn check(comptime held: usize, init: std.process.Init, stdout: *std.Io.Writer, opened: *common.Opened, drafter: ?*const rig.Opened, place: join_file.Place, reference_path: []const u8, layers: usize, kv: dsa.Storage) !u8 {
    const C = Check(held);
    const gpa = init.gpa;
    const io = init.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const c = &opened.config;
    var references = std.Io.Dir.cwd().openDir(io, reference_path, .{}) catch |err| {
        std.log.err("{s}: {t}", .{ reference_path, err });
        return 1;
    };
    defer references.close(io);
    var replies: [cases.len]vllm.Reply = undefined;
    for (cases, &replies) |one, *theirs| {
        const name = try std.fmt.allocPrint(arena, "{s}.json", .{one.name});
        const text = references.readFileAlloc(io, name, arena, .limited(1 << 26)) catch |err| {
            std.log.err("{s}/{s}: {t}", .{ reference_path, name, err });
            return 1;
        };
        theirs.* = vllm.reply(arena, text) catch |err| {
            std.log.err("{s}/{s}: {t}", .{ reference_path, name, err });
            return 1;
        };
    }
    const score_text = references.readFileAlloc(io, "score.json", arena, .limited(1 << 26)) catch |err| {
        std.log.err("{s}/score.json: {t}", .{ reference_path, err });
        return 1;
    };
    const scored = vllm.scored(arena, score_text) catch |err| {
        std.log.err("{s}/score.json: {t}", .{ reference_path, err });
        return 1;
    };

    const what = try std.fmt.allocPrint(arena, "ranks-check vllm layers={d}", .{layers});
    const session = try C.Session.create(gpa, arena, io, &opened.checkpoint.?, c, place, .{ .layers = layers, .rows = rows, .capacity = capacity, .alike = true, .kv = kv, .drafter = drafter }, what);
    defer session.destroy();
    const gib = units.gib(cuda.account.bytes);
    var one: C = .{ .arena = arena, .stdout = stdout, .session = session, .judged = layers == c.layers, .drafts = if (drafter != null) "DFlash2" else "the MTP layer" };
    if (C.Session.paired) {
        try one.say("zspark {s}{s}, NCCL {d}, {d} of {d} layers, its share {d:.2} GiB on the device, loaded in {d:.1} s", .{ cli.stamp.commit, if (cli.stamp.dirty) " dirty" else "", try nccl.version(), layers, c.layers, gib, units.seconds(session.load_ns) });
    } else try one.say("{d} of {d} layers, {d:.2} GiB on the device, loaded in {d:.1} s", .{ layers, c.layers, gib, units.seconds(session.load_ns) });
    const channel = &session.joined.channel;
    session.meet() catch |err| return channel.gone(session.first, @errorCast(err));

    try one.window(replies[0].prompt);
    for (cases, replies) |case, theirs| try one.case(case, theirs);
    try one.score(scored);
    const digest = one.digest.final();
    try one.say("digest {x} of what the ranks hold alike", .{digest});
    // both ranks ran every step whatever they found; here they hold the same, or say so on both
    if (C.Session.paired) {
        channel.barrier(digest | 1 << 63) catch |err| {
            if (err == error.OutOfStep) std.log.err("rank {d}: the other rank's digest is another", .{session.first});
            return channel.gone(session.first, err);
        };
        try one.say("the other rank holds the same digest", .{});
    }
    try stdout.print("{s}\n", .{if (one.failures == 0) "ranks-check: passed" else "ranks-check: FAILED"});
    return @intFromBool(one.failures != 0);
}

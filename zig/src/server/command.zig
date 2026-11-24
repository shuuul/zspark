//! What the models' `serve` commands share (spec 069): each loads its model its own way and hands
//! TensorFold's host `served.zig`'s engine; between the two stand the same words parted from the host's, the
//! engine's own flag of admission (`bounded`), the host's flags read (and those zspark does not serve refused),
//! the grammar engine a constrained reply is compiled by, and the sampling a request that names none is given;
//! `Startup` loads the persistent tool-ID key before the model and resolves thinking-off sampling before
//! serving. Shared checks and logs (spec 136 WS-04): the ranks asked for (`ranked`), the bounds of `--parallel`
//! and `--context`, `--spill-gib`'s three flags (`checkSpill`, `askedStates`), the flags not read (`warnUnread`),
//! the version, the front end with its grammar and think tags (`Front`), and the text and tokenizer loads'
//! diagnostics (`loadText`, `loadTokenizer`). A model's defaults stay in its command.

const std = @import("std");
const Allocator = std.mem.Allocator;
const host = @import("server");
const cli = @import("../cli.zig");
const learned_parts = @import("../decode/learned_parts.zig");
const request_file = @import("request.zig");
const structured = @import("structured.zig");
const text_frontend = @import("../text/frontend.zig");
const call_gate = @import("../text/call_gate.zig");
const json_schema = @import("../text/json_schema.zig");
const tokenizer_file = @import("../text/tokenizer.zig");

/// Whose a word of the command line is: TensorFold's, or zspark's own, alone or with the word after it.
pub const Kind = enum { host, own, own_valued };

pub const Parted = struct { own: []const []const u8, rest: []const []const u8 };

/// `args` parted into zspark's own words, a flag's value after it where one follows, and the rest, which are
/// TensorFold's; each in its order.
pub fn part(arena: Allocator, args: []const []const u8, comptime kindOf: fn (word: []const u8) Kind) Allocator.Error!Parted {
    var own: std.ArrayList([]const u8) = .empty;
    var rest: std.ArrayList([]const u8) = .empty;
    var at: usize = 0;
    while (at < args.len) : (at += 1) {
        const kind = kindOf(args[at]);
        if (kind == .host) {
            try rest.append(arena, args[at]);
            continue;
        }
        try own.append(arena, args[at]);
        if (kind == .own_valued and at + 1 < args.len) {
            at += 1;
            try own.append(arena, args[at]);
        }
    }
    return .{ .own = own.items, .rest = rest.items };
}

/// The host's words without zspark's own flag of admission, and what it said.
pub const Bounded = struct {
    rest: []const []const u8,
    /// `Served.max_queued`: how many requests may wait beyond the `--parallel` being decoded; null, no bound.
    max_queued: ?usize = null,
};

/// `--max-queued N` (or `--max-queued=N`, as the host's flags may be written) taken out of `rest`, the words
/// `part` left the host: a flag of every model's `serve`, read here because it is the engine's (`served.zig`)
/// and none of TensorFold's. Not given, nothing is bounded: vLLM's `--max-num-queued-reqs` and SGLang's
/// `--max-queued-requests` are unset by default too. Null with the fault said, for the exit code 2.
pub fn bounded(arena: Allocator, rest: []const []const u8) Allocator.Error!?Bounded {
    const flag = "--max-queued";
    var kept: std.ArrayList([]const u8) = .empty;
    var out: Bounded = .{ .rest = rest };
    var at: usize = 0;
    while (at < rest.len) : (at += 1) {
        const word = rest[at];
        const value = if (std.mem.eql(u8, word, flag)) value: {
            at += 1;
            if (at == rest.len) {
                std.log.err("serve: {s} takes a value", .{flag});
                return null;
            }
            break :value rest[at];
        } else if (std.mem.startsWith(u8, word, flag ++ "=")) word[flag.len + 1 ..] else {
            try kept.append(arena, word);
            continue;
        };
        out.max_queued = std.fmt.parseInt(usize, value, 10) catch {
            std.log.err("serve: {s} {s}: a count of requests, 0 for none", .{ flag, value });
            return null;
        };
    }
    out.rest = kept.items;
    return out;
}

/// TensorFold's `serve` flags read from `path` and `rest`; null with the fault said, for the exit code 2.
pub fn hostArgs(arena: Allocator, path: []const u8, rest: []const []const u8) Allocator.Error!?host.cli.Args {
    var usage: host.cli.Usage = .{};
    const args = host.cli.parse(arena, try std.mem.concat(arena, []const u8, &.{ &.{path}, rest }), &usage) catch |err| switch (err) {
        error.Usage => {
            std.log.err("serve: {s}", .{usage.message});
            return null;
        },
        error.OutOfMemory => |e| return e,
    };
    if (unserved(args, false)) |flag| {
        std.log.err("serve: {s} is not served by zspark", .{flag});
        return null;
    }
    return args;
}

/// The first flag `args` names that the host reads since TensorFold 1.0.4 or 1.0.5 and no model of zspark
/// serves, or null. Since 1.0.4: `--slide` and `--slide-graph` (Sliding Weights: the engine has no `learn`),
/// and `--drafter-bits` (the host's `Open.drafter`, which zspark's engine is not opened by). `--drafter` is
/// each model's own, parted before the host's words (GLM's, and Flash Next's since spec 126, the owner's
/// 2026-10-25 word), and never arrives here. Since 1.0.5: `--chat-template`
/// (a template in the checkpoint's place: the host's text would render it while `text/frontend.zig` and the
/// call gate still read the checkpoint's own, so the commands load the text with no override) and
/// `--learn-min-free-gib` (the free disk TensorFold's learned states leave; zspark's `--learn` is
/// `learnt.zig`'s and has no such floor). Since 1.0.12 (spec 106): `--learn-conversations` and
/// `--learn-daily-gib` (TensorFold's learned states again), `--recall` and its `--recall-dir`, `--recall-gib`
/// and `--recall-read-only` (Recall's record: the engine has no `recall`), `--record` (the drafter's training
/// rows, `Open.record`), `--precision` (`Open.precision`, NVFP4 activations at stored scales), `--cortheon`
/// (the host's research tools around a reply) and `--vision` with `--vision-max-images`, `--vision-image-tokens`
/// and `--vision-video-tokens` (spec 019 WS-04: images are served without a flag where the model offers
/// `Info.vision`, at 0.6.5's budgets; no model reads a video). `--vision-urls` is served: it lets the host
/// fetch an image part's public HTTPS URL. `--tp`, `--rank`, `--master`, `--policy`, `--mtp-*` and
/// `--checkpoint-slots` the host itself refuses where its root names no `serves`. Since 1.1.1 (spec 113):
/// `--vision-max-videos` (no model reads a video) and `--kv-cache fp8` (`Open.kv8`, e4m3 attention caches);
/// `--allowed-host` and `--allowed-origin` are the host's own guard and served, `--prompt-seed` is always on
/// (`hostArgs`). Since 1.1.4 (spec 141): the speech server's flags (`--voices`, `--default-voice`,
/// `--allowed-controls`, `--max-frames`, the `--silence-*` guard and its redraws, `--short-reply-*`,
/// `--register-*`: no model of zspark speaks), `--idle-unload` (the host closes and reopens the engine through
/// `Open`, which zspark's commands do not call) and `--prompt-precision` (`Open.prompt_int8`: the checkpoint's
/// activations are vLLM's, root `AGENTS.md`); `--thinking-reserve` is served as inert, the host saying at start
/// that it is not applied (`Info.answer_limit` false). Since 1.1.5 (spec 145): `--loops` (a looped model's
/// recurrent passes, `Open.loops`: no model of zspark loops); the `--cfg-*` guidance flags the host refuses
/// itself off macOS. Since 1.2.1 (spec 152): `--name-priority` (the host's lane priorities), `--loop-guard-stop`
/// (`Request.loop_stop`: zspark's guard closes the think block and answers after it, as under 1.1.5),
/// `--rope-yarn`, `--decode-share` and `--spill-gib` are host-native now — zspark serves none of the first four
/// and `--spill-gib` stays its own, parted before the host's words — and the FSR residency flags
/// `--ssd-experts`, `--ple-on-ssd` and `--fsr-route-trace` are macOS Metal's. The `--max-inflight*`/
/// `--requests-per-second`/`--request-burst` admission flags are the host's own 429s, off by default, and
/// served. Since 1.2.8 (spec 174): the connection flags (`--max-connections`,
/// `--max-connections-per-address`, `--trusted-proxy`) are the host's own accept-layer admission, built
/// into `Config` by `host.serve.run` itself, off by default, and served. Each was a usage error under
/// 1.0.2's host and stays one: taken and left unread it would promise what is not done. A value that is the
/// host's default (`--drafter-bits 4`, `--learn-min-free-gib 4`) cannot be told from the flag left out.
/// Once the engine is open, `greedy_only` also refuses an instruct set it cannot sample from.
pub fn unserved(args: host.cli.Args, greedy_only: bool) ?[]const u8 {
    const defaults: host.cli.Args = .{};
    if (args.slide) return "--slide";
    if (args.slide_graph != null) return "--slide-graph";
    if (args.drafter_bits != defaults.drafter_bits) return "--drafter-bits";
    if (args.chat_template != null) return "--chat-template";
    if (args.learn_min_free_gib != defaults.learn_min_free_gib) return "--learn-min-free-gib";
    if (args.learn_conversations) return "--learn-conversations";
    if (args.learn_daily_gib != defaults.learn_daily_gib) return "--learn-daily-gib";
    if (args.recall_dir != null) return "--recall-dir";
    if (args.recall_gib != defaults.recall_gib) return "--recall-gib";
    if (args.recall_read_only) return "--recall-read-only";
    if (args.recall) return "--recall";
    if (args.record != null) return "--record";
    if (args.precision != null) return "--precision";
    if (args.cortheon) return "--cortheon";
    if (args.cache_salt_per_key) return "--cache-salt-per-key";
    if (args.vision_max_images != defaults.vision_max_images) return "--vision-max-images";
    if (args.vision_image_tokens_set) return "--vision-image-tokens";
    if (args.vision_video_tokens != defaults.vision_video_tokens) return "--vision-video-tokens";
    if (args.vision_max_videos != defaults.vision_max_videos) return "--vision-max-videos";
    if (!std.mem.eql(u8, args.kv_cache, defaults.kv_cache)) return "--kv-cache";
    if (args.vision) return "--vision";
    if (args.voices != null) return "--voices";
    if (!std.mem.eql(u8, args.default_voice, defaults.default_voice)) return "--default-voice";
    if (args.allowed_controls != null) return "--allowed-controls";
    if (args.max_frames != null) return "--max-frames";
    if (!std.meta.eql(args.silence, defaults.silence)) return "--silence-seconds";
    if (args.redraw.max != null) return "--silence-redraws";
    if (args.redraw.budget_s != null) return "--silence-redraw-budget-s";
    if (args.redraw.controls != null) return "--silence-redraw-controls";
    if (args.redraw.short_ratio != null or args.redraw.short_chars_per_s != null) return "--short-reply-ratio";
    if (!std.meta.eql(args.redraw.register, defaults.redraw.register)) return "--register-redraw-st";
    if (args.idle_unload != defaults.idle_unload) return "--idle-unload";
    if (!std.mem.eql(u8, args.prompt_precision, defaults.prompt_precision)) return "--prompt-precision";
    if (args.loops != null) return "--loops";
    // TensorFold 1.2.1 (spec 152 WS-03): these are now the host's own and no model of zspark reads them.
    if (args.name_priority.len != 0) return "--name-priority";
    if (args.loop_guard_stop) return "--loop-guard-stop";
    if (args.rope_yarn != null) return "--rope-yarn";
    if (args.decode_share != null) return "--decode-share";
    // the FSR residency flags (`--ssd-experts`, `--ple-on-ssd`, `--fsr-route-trace`) are macOS-only in the
    // flag table: off macOS the parser refuses them itself, as `--cfg-*`, and none reach here.
    if (args.ssd_experts_gib != null) return "--ssd-experts";
    if (args.ple_on_ssd) return "--ple-on-ssd";
    if (args.fsr_trace != null) return "--fsr-route-trace";
    // `--spill-gib` is zspark's own on qwen4's and glm5's `serve`, parted before the host's words; on a command
    // that does not part it the parser refuses it already (not `.native`), so this line only names it.
    if (args.spill_gib != null) return "--spill-gib";
    // The host can resolve an instruct set, but a greedy-only engine cannot draw from it. `none` turns it off.
    if (greedy_only) if (args.instruct_sampling) |text| {
        if (!std.mem.eql(u8, std.mem.trim(u8, text, " "), "none")) return "--instruct-sampling";
    };
    return null;
}

/// A request names the tokens that end its reply, and they are the server's reading of the checkpoint
/// (`theirs`); the constraints and the head's chains end at zspark's (`ours`). Each difference is said.
pub fn warnEnds(theirs: []const u32, ours: []const u32) void {
    for (theirs) |id| if (std.mem.indexOfScalar(u32, ours, id) == null) std.log.warn("the server ends a reply at token {d}, which is none of the checkpoint's end tokens as zspark reads them", .{id});
    for (ours) |id| if (std.mem.indexOfScalar(u32, theirs, id) == null) std.log.warn("the checkpoint's end token {d} does not end a reply for the server", .{id});
}

/// The checkpoint's text (`HfText`: its tokenizer, template and end tokens), loaded as every `serve` did:
/// `HfText.load` with no template override, the fault said for the exit code 1 on null. `pa` is the
/// caller's arena, which `load` allocs its diagnostics under. The caller holds the `HfText` for the run
/// and `deinit`s it.
pub fn loadText(gpa: Allocator, io: std.Io, path: []const u8, pa: Allocator) ?*host.hf_text.HfText {
    var problem: []const u8 = "";
    return host.hf_text.HfText.load(gpa, io, path, null, pa, &problem) catch |err| {
        std.log.err("{s}: {s}", .{ path, if (problem.len > 0) problem else @errorName(err) });
        return null;
    };
}

/// zspark's own tokenizer over the host's (`text/tokenizer.zig`'s `Tokenizer.over`), the fault said for
/// the exit code 1 on null. The caller `deinit`s it under `gpa` before the `loaded` it stands on goes.
pub fn loadTokenizer(gpa: Allocator, loaded: *const tokenizer_file.Loaded, path: []const u8) ?tokenizer_file.Tokenizer {
    var diagnostic: tokenizer_file.Diagnostic = .{};
    return tokenizer_file.Tokenizer.over(gpa, loaded, &diagnostic) catch |err| {
        std.log.err("{s}: the tokenizer: {s}", .{ path, if (diagnostic.len > 0) diagnostic.message() else @errorName(err) });
        return null;
    };
}

/// The grammar engine a structured request's constraint is compiled by, and what a required call's values
/// may hold: zspark's, over its own reading of the tokenizer. A request that asks for nothing reaches neither.
/// Made in place (`compiler` points into it) and kept until the engine is closed.
pub const Grammar = struct {
    gpa: Allocator,
    engine: json_schema.Engine,
    strings: call_gate.Strings,
    tokenizer: *const tokenizer_file.Tokenizer,
    stops: []const u32,

    pub fn init(self: *Grammar, gpa: Allocator, tokenizer: *const tokenizer_file.Tokenizer, vocab: usize, stops: []const u32) Allocator.Error!void {
        self.* = .{ .gpa = gpa, .engine = try .init(gpa, tokenizer, vocab, stops), .strings = undefined, .tokenizer = tokenizer, .stops = stops };
        errdefer self.engine.deinit(gpa);
        self.strings = try .init(gpa, tokenizer, self.engine.interface().words, stops);
    }

    pub fn deinit(self: *Grammar) void {
        self.strings.deinit();
        self.engine.deinit(self.gpa);
    }

    pub fn compiler(self: *Grammar) structured.Compiler {
        return .{ .engine = self.engine.interface(), .tokenizer = self.tokenizer, .stops = self.stops, .gate = .{ .strings = &self.strings } };
    }
};

/// `generation_config.json`'s sampling (`do_sample` false is greedy), then the serve flags over it: the
/// server's defaults for a request that names none. TensorFold's `zig/src/server/main.zig` `sampling`.
pub fn samplingDefaults(a: Allocator, io: std.Io, dir: []const u8, args: host.cli.Args) !?host.json.Value {
    const json = host.json;
    const out = try json.newObject(a);
    const path = try std.fs.path.join(a, &.{ dir, "generation_config.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20))) |bytes| {
        const parsed = try json.parse(a, bytes);
        if (parsed == .ok) {
            const cfg = parsed.ok;
            for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
            if (cfg.get("do_sample")) |d| if (d == .bool) {
                if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
            };
        }
    } else |_| {}
    if (args.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (args.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (args.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (args.min_p) |t| try out.put(a, "min_p", .{ .float = t });
    return if (out.count() > 0) json.Value{ .object = out } else null;
}

/// TensorFold 1.2.7's startup contract (`server/main.zig`): one key load on the HTTP rank, before the model;
/// the server takes the key when it starts. Keep this in place until `run` returns, then clear any untaken key.
pub const Startup = struct {
    key: ?host.ids.Key = null,

    /// A bad configured key refuses startup by the setting's name, without printing its path or contents.
    pub fn open(a: Allocator, environ: *const std.process.Environ.Map) host.tool_id_key.Error!?Startup {
        var problem: []const u8 = "";
        const key = host.tool_id_key.load(a, environ.get(host.tool_id_key.name), &problem) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => {
                std.debug.print("zspark serve: {s}\n", .{problem});
                return null;
            },
        };
        return .{ .key = key };
    }

    pub fn deinit(self: *Startup) void {
        if (self.key) |*key| std.crypto.secureZero(u8, std.mem.asBytes(key));
        self.key = null;
    }

    /// Preserve each model's ordinary defaults; only an explicit --instruct-sampling asks for the second set.
    /// Resolve against the checkpoint's actual model_type and the engine's advertised penalty support.
    pub fn run(self: *Startup, gpa: Allocator, io: std.Io, args: host.cli.Args, setup: host.serve.Setup) u8 {
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const a = arena_state.allocator();
        var ready = setup;
        if (args.instruct_sampling) |text| {
            const info = setup.engine.info();
            if (unserved(args, info.greedy_only)) |flag| {
                std.debug.print("zspark serve: {s} is not served by this engine\n", .{flag});
                return 2;
            }
            const path = std.fs.path.join(a, &.{ setup.model_dir, "generation_config.json" }) catch return 1;
            const generation = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch null;
            const got = host.family_sampling.startup(a, setup.model_type, generation, .{
                .temperature = args.temperature,
                .top_p = args.top_p,
                .top_k = args.top_k,
                .min_p = args.min_p,
            }, info.capability(.penalties), text) catch return 1;
            if (got.note) |note| host.log.line("{s}", .{note});
            if (got.problem) |problem| {
                std.debug.print("zspark serve: {s}\n", .{problem});
                return 1;
            }
            ready.instruct = got.set;
        }
        ready.tool_id_key = if (self.key) |*key| key else null;
        return host.serve.run(gpa, io, args, ready);
    }
};

/// Whether a command runs a rank of two, or the exit code it ends with when the ranks asked for cannot be had.
pub const Ranked = union(enum) { alone, paired, exit: u8 };

/// How a command spells its ranks in the line that refuses an incomplete pair.
pub const Spelled = enum { flags, words };

/// `place` as asked: neither `rank` nor `at` is one process; both and a rank of 0 or 1 is a pair, which needs
/// NCCL in this build (`linked`, `nccl.linked`: this file imports neither, so that the tests of the host
/// build without it). A half pair is refused with 2, a build without NCCL with 1; each said under `verb`.
/// `place` is `ranks/join.zig`'s `Place`.
pub fn ranked(comptime verb: []const u8, place: anytype, linked: bool, comptime spelled: Spelled) Ranked {
    if (place.rank == null and place.at == null) return .alone;
    if (!place.complete()) {
        std.log.err(verb ++ ": a rank is " ++ switch (spelled) {
            .flags => "--rank 0 or 1 with --at ADDR:PORT",
            .words => "rank=0 or 1 with at=ADDR:PORT",
        } ++ ", where rank 1 listens", .{});
        return .{ .exit = 2 };
    }
    if (!linked) {
        std.log.err(verb ++ ": this build has no NCCL; build with -Dnccl=PREFIX (/usr for the apt packages)", .{});
        return .{ .exit = 1 };
    }
    return .paired;
}

/// `--parallel` as the host read it: `auto` is `auto_streams`, anything else the number given. Null, said, when
/// it is above `most`, or (with `least` above 0) below `least`.
pub fn parallelWithin(args: host.cli.Args, auto_streams: usize, least: usize, most: usize) ?usize {
    const parallel: usize = if (host.cli.parallelFixed(args.parallel)) host.cli.parallel(args.parallel).? else auto_streams;
    if (parallel > most or parallel < least) {
        if (least > 0) std.log.err("--parallel {d}: from {d} to {d} replies at once", .{ parallel, least, most }) else std.log.err("--parallel {d}: {d} streams at most", .{ parallel, most });
        return null;
    }
    return parallel;
}

/// `--context` as asked, or null where it is not given; error.Refused, said, below `least` tokens.
pub fn askedContext(args: host.cli.Args, least: usize) error{Refused}!?usize {
    const context: usize = if (args.context) |asked| std.math.cast(usize, asked) orelse 0 else return null;
    if (context < least) {
        std.log.err("--context {d}: {d} tokens at least", .{ context, least });
        return error.Refused;
    }
    return context;
}

/// `--spill-gib`, `--learn-min-free-gib` and `--spill-dir` as a `serve` read them; false, said, for the exit
/// code 2: a number of GiB from 0 below 1 Mi, and a directory only where a size is given for it.
pub fn checkSpill(spill_gib: f64, learn_min_free_gib: f64, spill_dir: ?[]const u8) bool {
    if (!(spill_gib >= 0 and spill_gib < 1 << 20)) {
        std.log.err("--spill-gib {d}: GiB from 0 (off)", .{spill_gib});
        return false;
    }
    if (!(learn_min_free_gib >= 0 and learn_min_free_gib < 1 << 20)) {
        std.log.err("--learn-min-free-gib {d}: GiB from 0", .{learn_min_free_gib});
        return false;
    }
    if (spill_dir != null and spill_gib == 0) {
        std.log.err("--spill-dir: with --spill-gib N, the GiB the states that left may hold there", .{});
        return false;
    }
    return true;
}

/// What `learnt.open` is asked: `learn` and the states that leave (`--spill-gib` and `--spill-dir`, none at
/// 0), under the build's commit and the disk's floor (`--learn-min-free-gib`).
pub fn askedStates(init: std.process.Init, path: []const u8, learn: ?learned_parts.Root, spill_gib: f64, spill_dir: ?[]const u8, learn_min_free_gib: f64) learned_parts.Asked {
    return .{
        .learn = learn,
        .spill = if (spill_gib > 0) .{ .dir = spill_dir, .gib = spill_gib } else null,
        .home = init.minimal.environ.getPosix("HOME"),
        .checkpoint = path,
        .commit = cli.stamp.commit,
        .dirty = cli.stamp.dirty,
        .min_free = @intFromFloat(learn_min_free_gib * (1 << 30)),
    };
}

/// A flag of the host's that a command does not read, and whether it was given.
pub const Unread = struct { name: []const u8, given: bool };

/// The host's flags that no model reads (`--speed-up`, `--backend`, `--device`, `--segments`) and `more`, each
/// given one said not to be read, as `tail` finishes the line.
pub fn warnUnread(args: host.cli.Args, tail: []const u8, more: []const Unread) void {
    const own = [_]Unread{
        .{ .name = "--speed-up", .given = args.speed_up != null },
        .{ .name = "--backend", .given = !std.mem.eql(u8, args.backend, "auto") },
        .{ .name = "--device", .given = args.device != null },
        .{ .name = "--segments", .given = args.segments != null },
    };
    for (own) |unread| if (unread.given) std.log.warn("{s} {s}", .{ unread.name, tail });
    for (more) |unread| if (unread.given) std.log.warn("{s} {s}", .{ unread.name, tail });
}

/// The line every `serve` says first: the version and the commit it was built from.
pub fn logVersion() void {
    std.log.info("zspark {s} {s}{s}", .{ cli.version, cli.stamp.commit, if (cli.stamp.dirty) " dirty" else "" });
}

/// The open `/health` identity: the exact `zspark version` line, without its newline, and the engine backend.
/// Loads retain no device name; leave it omitted rather than launch `cuda.probe` just for that field.
pub fn identity() host.server.Identity {
    return .{
        .version = "zspark " ++ cli.version ++ " " ++ cli.stamp.commit ++ (if (cli.stamp.dirty) " dirty" else ""),
        .backend = "cuda",
    };
}

/// The tokenizer and template a `serve` hands the host's request path, the grammar engine over the same
/// tokenizer, and the `<think>` tags it holds. Made in place, as `Grammar` is.
pub const Front = struct {
    frontend: text_frontend.Frontend,
    grammar: Grammar,

    /// Loads the front end from `dir` (`shared`: a tokenizer already read, `cli.loadFrontend`'s) and makes the
    /// grammar; false with the reason logged where it does not load.
    pub fn open(self: *Front, gpa: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, shared: ?*const text_frontend.Loaded, vocab: usize, stops: []const u32) !bool {
        return self.openWith(gpa, io, dir, path, shared, null, vocab, stops);
    }

    /// `open` with `own` the template where it is not null (`Frontend.loadWith`).
    pub fn openWith(self: *Front, gpa: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, shared: ?*const text_frontend.Loaded, own: ?[]const u8, vocab: usize, stops: []const u32) !bool {
        self.frontend = cli.loadFrontendWith(gpa, io, dir, path, shared, own) orelse return false;
        errdefer self.frontend.deinit(gpa);
        try self.grammar.init(gpa, &self.frontend.tokenizer, vocab, stops);
        return true;
    }

    pub fn deinit(self: *Front, gpa: Allocator) void {
        self.grammar.deinit();
        self.frontend.deinit(gpa);
    }

    /// `<think>` and `</think>` as the tokenizer holds them, or null where it holds either not.
    pub fn think(self: *const Front) ?request_file.Think {
        const open_id = self.frontend.tokenizer.addedId("<think>") orelse return null;
        const close_id = self.frontend.tokenizer.addedId("</think>") orelse return null;
        return .{ .open = @intCast(open_id), .close = @intCast(close_id) };
    }
};

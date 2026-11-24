//! `zspark chat DIR [rank=R at=ADDR:PORT] [OPTION..] MESSAGE`: one user message through the checkpoint's chat
//! template, a reply decoded on the session (`../session/session.zig`) and printed as it comes (spec 045 WS-04; Flash
//! Next's is `../../qwen4/commands/chat.zig`). With a rank the process is one of two, and both are given the
//! same words: rank 1 renders the prompt only for its length, which the ranks hold alike, and follows.
//! Without one the model's first layers alone fit (`--layers`), and their reply is no text.

const std = @import("std");
const cli = @import("../../../cli.zig");
const nccl = @import("../../../ranks/nccl.zig");
const command = @import("../../../server/command.zig");
const cuda = @import("../../../cuda/cuda.zig");
const join_file = @import("../../../ranks/join.zig");
const sampling = @import("../../../decode/sampling.zig");
const text_frontend = @import("../../../text/frontend.zig");
const text_tokenizer = @import("../../../text/tokenizer.zig");
const glm5_layout = @import("../layers/layout.zig");
const rig = @import("../dflash2/rig.zig");
const session_file = @import("../session/session.zig");
const snapshot_file = @import("../session/snapshot.zig");
const dsa = @import("../layers/dsa.zig");
const common = @import("common.zig");
const serve_file = @import("serve.zig");
const world = glm5_layout.world;

pub const usage =
    \\chat DIR [rank=R at=ADDR:PORT] [OPTION..] MESSAGE
    \\  --system TEXT      a system message before MESSAGE
    \\  --max-tokens N     the most reply tokens (default 512)
    \\  --no-thinking      close the think block in the prompt
    \\  --seed N           the sampling key (default: derived from the prompt, so a prompt repeats its reply)
    \\  --temperature X    0 decodes greedily (default: the checkpoint's generation config, else 0)
    \\  --serial           no drafts
    \\  --drafter DIR      the DFlash2 drafter in the MTP layer's place
    \\  --layers L         the first layers alone, both ranks in this process
    \\  --rows N           a prompt window's rows (default 4096)
    \\  --kv-dtype bf16|fp8  how the DSA caches hold their latents (default fp8; the MTP layer's stays bf16)
    \\  --render           print the prompt's text and ids, and stop before loading the model
;

const Options = struct {
    system: ?[]const u8 = null,
    message: ?[]const u8 = null,
    max_tokens: usize = 512,
    thinking: bool = true,
    seed: ?i64 = null,
    temperature: ?f64 = null,
    serial: bool = false,
    drafter: ?[]const u8 = null,
    layers: ?usize = null,
    rows: ?usize = null,
    render: bool = false,
    kv: ?dsa.Storage = null,

    const flags = [_]cli.Flag{
        .{ .name = "--no-thinking", .field = "thinking", .on = false },
        .{ .name = "--serial", .field = "serial" },
        .{ .name = "--render", .field = "render" },
        .{ .name = "--system", .field = "system" },
        .{ .name = "--drafter", .field = "drafter" },
        .{ .name = "--max-tokens", .field = "max_tokens", .min = 1 },
        .{ .name = "--layers", .field = "layers", .min = 1 },
        .{ .name = "--rows", .field = "rows", .min = 1, .max = 4096 },
        .{ .name = "--kv-dtype", .field = "kv" },
        .{ .name = "--seed", .field = "seed" },
        .{ .name = "--temperature", .field = "temperature", .min = 0 },
    };

    /// `args` read, or null with the reason said. `place` takes the words `rank=R at=ADDR:PORT`.
    fn parse(arena: std.mem.Allocator, args: []const []const u8, place: *join_file.Place) !?Options {
        var kept: std.ArrayList([]const u8) = .empty;
        for (args) |word| if (!place.take(word)) try kept.append(arena, word);
        const self = cli.parse(Options, &flags, .{ .positional = .{ .field = "message", .again = "one MESSAGE; quote it" }, .quoted = false, .prefix = "chat: " }, kept.items) orelse return null;
        if (self.message == null) {
            std.log.err("usage: zspark " ++ usage, .{});
            return null;
        }
        return self;
    }
};

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    const arena = init.arena.allocator();
    const io = init.io;
    const path = words[0];
    var place: join_file.Place = .{};
    const options = (try Options.parse(arena, words[1..], &place)) orelse return 2;
    switch (command.ranked("chat", place, nccl.linked, .words)) {
        .exit => |code| return code,
        .alone, .paired => {},
    }
    const paired = place.complete();
    var opened: common.Opened = .{};
    defer opened.close(io);
    if (!try opened.enter(io, path, .say)) return 1;
    var frontend = cli.loadFrontend(init.gpa, io, opened.dir.?, path, null) orelse return 1;
    defer frontend.deinit(init.gpa);

    const made = switch (try cli.chatPrompt(init, stdout, &frontend, .{
        .system = options.system,
        .message = options.message,
        .thinking = options.thinking,
        .render = options.render,
        .seed = options.seed,
        .temperature = options.temperature,
    }, "chat")) {
        .exit => |code| return code,
        .ready => |ready| ready,
    };
    const prompt = made.ids;
    const settings = made.settings;

    if (!try opened.configure(init) or !try opened.resolve(init)) return 1;
    const layers = options.layers orelse opened.config.layers;
    if (layers > opened.config.layers) {
        std.log.err("--layers {d}: the model has {d}", .{ layers, opened.config.layers });
        return 2;
    }
    if (!paired and layers == opened.config.layers) {
        std.log.err("GLM-5.3-Flash does not fit one process: give both Sparks the command with rank=1 at=ADDR:PORT on the second, started first, and rank=0 on the first (tools/two_ranks.sh chat), or --layers L to try the first layers here", .{});
        return 1;
    }
    var drafting: rig.Opened = .{};
    defer drafting.close(init.io);
    if (options.drafter) |directory| if (!try rig.enter(init, &drafting, directory, &opened.config, layers)) return 1;
    const session_options: session_file.Options = .{ .layers = layers, .rows = options.rows orelse serve_file.defaultRows(&opened.config), .capacity = prompt.len + options.max_tokens, .serial = options.serial, .drafter = if (options.drafter != null) &drafting else null, .snapshots = snapshot_file.held_default, .stride = snapshot_file.stride_default, .kv = options.kv orelse .stored(&opened.config) };
    return if (paired)
        reply(1, init, stdout, &opened, &frontend, place, session_options, settings, prompt, options.max_tokens)
    else
        reply(world, init, stdout, &opened, &frontend, place, session_options, settings, prompt, options.max_tokens);
}

/// The reply's tokens made text and written as they come.
const Spoken = struct {
    decoder: text_tokenizer.StreamDecoder,
    stdout: *std.Io.Writer,
    eos: []const u32,

    fn take(context: *anyopaque, token: i32) void {
        const self: *Spoken = @ptrCast(@alignCast(context));
        if (std.mem.indexOfScalar(u32, self.eos, @intCast(token)) != null) return;
        self.stdout.writeAll(self.decoder.next(@intCast(token))) catch {};
        self.stdout.flush() catch {};
    }

    fn halt(_: *anyopaque) bool {
        return false;
    }

    fn held(_: *anyopaque, _: usize) void {}
};

fn reply(comptime held: usize, init: std.process.Init, stdout: *std.Io.Writer, opened: *common.Opened, frontend: *text_frontend.Frontend, place: join_file.Place, session_options: session_file.Options, settings: sampling.Sampling, prompt: []const i32, max_tokens: usize) !u8 {
    const Session = session_file.SessionOf(held);
    const arena = init.arena.allocator();
    const config = &opened.config;
    const what = try std.fmt.allocPrint(arena, "chat t={d} seed={d} prompt={x}", .{ settings.temperature, settings.seed, std.hash.Wyhash.hash(prompt.len, std.mem.sliceAsBytes(prompt)) });
    const session = Session.create(init.gpa, arena, init.io, &opened.checkpoint.?, config, place, session_options, what) catch |err| {
        std.log.err("chat: {t}", .{err});
        return 1;
    };
    defer session.destroy();
    std.log.info("{s}{d} of {d} layers, {d:.2} GiB in {d:.1} s", .{ if (Session.paired) "this rank's share: " else "both ranks in this process: ", session_options.layers, config.layers, @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0), @as(f64, @floatFromInt(session.load_ns)) / 1e9 });
    session.meet() catch |err| return common.gone(session, err);
    if (session.follows()) {
        _ = session.follow() catch |err| return common.gone(session, err);
        return 0;
    }

    var spoken: Spoken = .{ .decoder = try .init(init.gpa, &frontend.tokenizer, .{}), .stdout = stdout, .eos = config.eos };
    defer spoken.decoder.deinit(init.gpa);
    const out = try arena.alloc(i32, max_tokens);
    const done = session.reply(prompt, settings, .{ .eos = config.eos, .tap = .{ .context = &spoken, .take = Spoken.take, .halt = Spoken.halt, .held = Spoken.held } }, out) catch |err| {
        std.log.err("chat: {t}", .{err});
        return 1;
    };
    try stdout.writeAll(spoken.decoder.finish());
    try stdout.writeAll("\n");
    try stdout.flush();
    const rate = if (done.made > 1) @as(f64, @floatFromInt(done.made - 1)) * 1e9 / @as(f64, @floatFromInt(@max(done.rest_ns, 1))) else 0;
    std.log.info("{d} tokens of prompt, the first of the reply after {d:.0} ms; {d} tokens ({t}) at {d:.1} a second; {d} rounds, {d} of {d} drafts kept", .{ prompt.len, @as(f64, @floatFromInt(done.first_ns)) / 1e6, done.made, done.finish, rate, done.stats.rounds, done.stats.accepted, done.stats.drafted });
    if (session.leads()) _ = session.end() catch |err| return common.gone(session, err);
    return 0;
}

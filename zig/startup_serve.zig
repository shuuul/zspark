//! `startup_serve serve [--says TEXT] [--model-type NAME] DIR [serve flags]`: the shared `command.Startup`
//! over a scripted engine, as `command_test.zig`'s smoke drives it in a process of its own (so SIGTERM, the
//! installed signal handlers, the listener and the persistent tool-id key live and die with it). It announces
//! `PORT` on stderr as upstream's `fake_serve` does on stdout, and answers every request with `--says`'s
//! bytes ("hi" where it is not given).
const std = @import("std");
const server = @import("server");
const api = @import("engine_api");
const command = @import("src/server/command.zig");
const fixture = @import("tests/server/served_fixture.zig");

fn announce(port: u16) void {
    std.debug.print("PORT {d}\n", .{port});
}

/// A byte a token, like `served_fixture.zig`'s `Hi`, but an `api.Engine` directly (no `Served`): what the
/// shared startup is driven over where no test of it needs the decoded text.
const Echo = struct {
    says: []const u8,
    tokens: [512]u32 = undefined,

    fn engine(e: *Echo) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(ctx: *anyopaque) api.Info {
        _ = ctx;
        return .{ .context_window = 8192 };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Echo = @ptrCast(@alignCast(ctx));
        _ = r;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        var n: usize = 0;
        for (e.says) |byte| {
            if (n == e.tokens.len) break;
            e.tokens[n] = byte;
            n += 1;
        }
        sink.event(sink.ctx, id, &.{ .tokens = e.tokens[0..n] });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    server.log.init(io, false);
    if (argv.len < 2 or !std.mem.eql(u8, argv[1], "serve")) {
        std.debug.print("usage: startup_serve serve [--says TEXT] [--model-type NAME] DIR [serve flags]\n", .{});
        return 2;
    }
    // The driver's own options come off before the host's parse sees the rest.
    var says: []const u8 = "hi";
    var model_type: []const u8 = "hi";
    var words: std.ArrayList([]const u8) = .empty;
    var i: usize = 2;
    while (i < argv.len) : (i += 1) {
        if (std.mem.eql(u8, argv[i], "--says") and i + 1 < argv.len) {
            says = argv[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, argv[i], "--model-type") and i + 1 < argv.len) {
            model_type = argv[i + 1];
            i += 1;
        } else try words.append(a, argv[i]);
    }
    var u: server.cli.Usage = .{};
    const args = server.cli.parse(a, words.items, &u) catch {
        std.debug.print("startup_serve: error: {s}\n", .{u.message});
        return 2;
    };
    var text: fixture.Bytes = .{};
    var echo: Echo = .{ .says = says };
    const sampling = command.samplingDefaults(a, io, args.model, args) catch null;
    var startup = (try command.Startup.open(a, init.environ_map)) orelse return 1;
    defer startup.deinit();
    return startup.run(gpa, io, args, .{
        .identity = command.identity(),
        .engine = echo.engine(),
        .text = text.text(),
        .served = server.hub.servedName(args.name, args.model, args.model),
        .model_dir = args.model,
        .model_type = model_type,
        .sampling = sampling,
        .environ = init.environ_map,
        .on_listen = announce,
    });
}

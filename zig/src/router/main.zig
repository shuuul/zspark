//! `zspark-router`: one address over whole copies of `zspark serve` (spec 088). It names no checkpoint and
//! loads no CUDA; a copy a machine runs `zspark serve` as it always does, and this stands between them and
//! the client, a conversation kept on the copy that holds its prefix. Each reply is logged with its copy,
//! how it was placed and its key.

const std = @import("std");
const router_file = @import("router.zig");
const http = @import("http.zig");
const stamp = @import("stamp");

const Router = router_file.Router;
const net = std.Io.net;

const usage =
    \\usage: zspark-router version
    \\       zspark-router [--listen HOST:PORT] COPY [COPY..] [FLAG..]
    \\
    \\  One address over whole copies of `zspark serve` (spec 088). A COPY is NAME=HOST:PORT, one a copy,
    \\  named in the order arrivals tie in. A request stays on the copy that holds its conversation: by the
    \\  hash of its opening, the `x-zspark-routing-key` header that names one instead, or the
    \\  `previous_response_id` a reply of Responses gave; what none of those names goes to the least
    \\  in-flight copy. A copy that fails 3 times running is left 30 s, a health probe a poll telling it
    \\  back; its conversations are met elsewhere and prefill again there.
    \\
    \\  Copies need not be one model. Each copy's `GET /v1/models` is read when the router starts and when a
    \\  copy comes back; a JSON request (chat completions, completions, responses, messages, tokenize) goes by
    \\  its body's `model` to the copies that list it, and then as above among them. A request without `model`
    \\  goes to the --default copy's model (vLLM serves its first name); an unknown model gets vLLM's 404.
    \\  Any /v1/audio/ path goes to the --audio copies, its body unread. GET /v1/models lists every copy's
    \\  models; /health is 200 only when every model has a copy up (else 503). Copies that list no model are
    \\  one model.
    \\
    \\  --listen HOST:PORT        the address served (127.0.0.1:18887)
    \\  --audio NAME              this copy serves /v1/audio/* (repeat for more)
    \\  --default NAME            the copy whose model a request without `model` gets (the first copy)
    \\  --parallel N              the most requests in flight at one copy, the copies' own --parallel (4)
    \\  --parallel-of NAME=N      that copy's own, where it is not --parallel's (repeat for more)
    \\  --poll-ms N               the health poll's period (2000)
    \\  --open-ms N               the least a copy is left after its failures (30000)
    \\  --failures N              the failures running that leave a copy (3)
    \\  --connect-timeout-ms N    the connect to a copy gives up after (2000)
    \\  --health-timeout-ms N     a health probe's answer must come within (1000)
    \\  --max-body BYTES          a request body past this is refused (67108864)
    \\  --table N                 the routing keys and reply ids kept, oldest turned over first (4096)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_file: std.Io.File.Writer = .initStreaming(.stdout(), init.io, &stdout_buffer);
    const stdout = &stdout_file.interface;
    defer stdout.flush() catch {};

    if (args.len > 1 and std.mem.eql(u8, args[1], "version")) {
        try stdout.print("zspark-router {s}+{s} {s}{s}\n", .{ stamp.version, stamp.optimize, stamp.commit, if (stamp.dirty) " dirty" else "" });
        return 0;
    }

    var given: router_file.Config = .{ .io = init.io };
    var listen_address: ?net.IpAddress = null;
    var specs: std.ArrayList(router_file.CopySpec) = .empty;
    var audio_names: std.ArrayList([]const u8) = .empty;
    var default_name: ?[]const u8 = null;
    var limits: std.ArrayList([]const u8) = .empty;
    var bad: ?[]const u8 = null;
    var at: usize = 1;
    while (at < args.len) : (at += 1) {
        const word = args[at];
        if (std.mem.eql(u8, word, "-h") or std.mem.eql(u8, word, "--help")) {
            try stdout.writeAll(usage);
            return 0;
        }
        if (std.mem.eql(u8, word, "--listen")) {
            at += 1;
            if (at >= args.len) {
                bad = "--listen takes HOST:PORT";
                break;
            }
            listen_address = parseAddress(args[at]) orelse {
                bad = "the listen address is not HOST:PORT";
                break;
            };
            continue;
        }
        if (std.mem.eql(u8, word, "--parallel-of")) {
            at += 1;
            if (at >= args.len) {
                bad = "--parallel-of takes NAME=N";
                break;
            }
            try limits.append(init.arena.allocator(), args[at]);
            continue;
        }
        if (std.mem.eql(u8, word, "--audio") or std.mem.eql(u8, word, "--default")) {
            at += 1;
            if (at >= args.len) {
                bad = std.fmt.allocPrint(init.arena.allocator(), "{s} takes a copy's NAME", .{word}) catch "a flag was left without a value";
                break;
            }
            if (word[2] == 'a') {
                try audio_names.append(init.arena.allocator(), args[at]);
            } else {
                default_name = args[at];
            }
            continue;
        }
        if (std.mem.startsWith(u8, word, "--")) {
            at += 1;
            if (at >= args.len) {
                bad = std.fmt.allocPrint(init.arena.allocator(), "{s} takes a number", .{word}) catch "a flag was left without a value";
                break;
            }
            const number = std.fmt.parseInt(u64, args[at], 10) catch {
                bad = std.fmt.allocPrint(init.arena.allocator(), "{s}: {s} is not a number", .{ word, args[at] }) catch "a flag's value was not a number";
                break;
            };
            setFlag(&given, word[2..], number) orelse {
                bad = std.fmt.allocPrint(init.arena.allocator(), "no flag {s}", .{word}) catch "a flag was not known";
                break;
            };
            continue;
        }
        // a copy: NAME=HOST:PORT
        const eq = std.mem.indexOfScalar(u8, word, '=') orelse {
            bad = std.fmt.allocPrint(init.arena.allocator(), "{s}: a copy is NAME=HOST:PORT", .{word}) catch "a copy was not NAME=HOST:PORT";
            break;
        };
        const address = parseAddress(word[eq + 1 ..]) orelse {
            bad = std.fmt.allocPrint(init.arena.allocator(), "{s}: the address is not HOST:PORT", .{word}) catch "a copy's address was not HOST:PORT";
            break;
        };
        try specs.append(init.arena.allocator(), .{ .name = word[0..eq], .address = address });
    }
    if (bad) |said| {
        std.log.err("{s}", .{said});
        try stdout.writeAll(usage);
        return 2;
    }
    if (specs.items.len == 0) {
        try stdout.writeAll(usage);
        return 2;
    }
    for (audio_names.items) |name| {
        const spec = find(specs.items, name) orelse {
            std.log.err("--audio {s}: no such copy", .{name});
            return 2;
        };
        spec.audio = true;
    }
    for (limits.items) |word| {
        const eq = std.mem.indexOfScalar(u8, word, '=') orelse {
            std.log.err("--parallel-of {s}: NAME=N", .{word});
            return 2;
        };
        const spec = find(specs.items, word[0..eq]) orelse {
            std.log.err("--parallel-of {s}: no such copy", .{word});
            return 2;
        };
        spec.parallel = std.fmt.parseInt(usize, word[eq + 1 ..], 10) catch 0;
        if (spec.parallel.? == 0) {
            std.log.err("--parallel-of {s}: N is a number above 0", .{word});
            return 2;
        }
    }
    if (default_name) |name| {
        const spec = find(specs.items, name) orelse {
            std.log.err("--default {s}: no such copy", .{name});
            return 2;
        };
        given.default_copy = spec - specs.items.ptr;
    }

    // The router's own address: the recipe's and the script's 18887 unless given.
    const listen = listen_address orelse parseAddress("127.0.0.1:18887").?;
    const listener = try http.listen(listen);
    var said: [64]u8 = undefined;
    std.log.info("listening on {s}, {d} copies behind it", .{ http.addressText(&said, listen), specs.items.len });
    const router = try Router.create(init.gpa, given, specs.items);
    Router.serve(router, listener); // until killed
    return 0;
}

/// The copy named `name`.
fn find(specs: []router_file.CopySpec, name: []const u8) ?*router_file.CopySpec {
    for (specs) |*spec| if (std.mem.eql(u8, spec.name, name)) return spec;
    return null;
}

/// `--flag N` onto the config's number it names; null for a flag there is none of.
fn setFlag(given: *router_file.Config, name: []const u8, number: u64) ?void {
    if (std.mem.eql(u8, name, "parallel")) {
        given.parallel = @intCast(number);
    } else if (std.mem.eql(u8, name, "poll-ms")) {
        given.poll_ms = number;
    } else if (std.mem.eql(u8, name, "open-ms")) {
        given.open_ms = number;
    } else if (std.mem.eql(u8, name, "failures")) {
        given.failure_limit = @intCast(number);
    } else if (std.mem.eql(u8, name, "connect-timeout-ms")) {
        given.connect_timeout_ms = @intCast(number);
    } else if (std.mem.eql(u8, name, "health-timeout-ms")) {
        given.health_timeout_ms = @intCast(number);
    } else if (std.mem.eql(u8, name, "max-body")) {
        given.max_body = number;
    } else if (std.mem.eql(u8, name, "table")) {
        given.table_entries = number;
    } else {
        return null;
    }
}

/// `HOST:PORT`, the host an IPv4 literal, as the recipe and the script name addresses.
fn parseAddress(text: []const u8) ?net.IpAddress {
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return null;
    const port = std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return null;
    return net.IpAddress.parse(text[0..colon], port) catch null;
}

//! The router's tests (spec 088 WS-01): two fake copies over loopback sockets, after
//! `tests/server/served_test.zig`'s harness, and the routing key's unit cases. No GPU, no checkpoint.

const std = @import("std");
const testing = std.testing;
const router_file = @import("../../src/router/router.zig");
const http = @import("../../src/router/http.zig");
const sent = @import("../server/served_fixture.zig").sent;
const posix = std.posix;
const Allocator = std.mem.Allocator;

const Router = router_file.Router;
const Config = router_file.Config;
const net = std.Io.net;

/// A fake copy: a loopback listener whose thread answers every request as it is told to, keeping a record
/// of the requests it took. A held fake records a request and answers nothing until released, so the
/// router sees it in flight.
const Fake = struct {
    gpa: Allocator,
    name: []const u8,
    listener: posix.socket_t,
    port: u16,
    address: net.IpAddress,
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    hold: std.atomic.Value(bool) = .init(false),
    count: std.atomic.Value(usize) = .init(0),
    /// When set, a /health GET is met with silence: the router's probe times out (a copy too busy
    /// to answer its own health, the breaker's failure without a forward).
    deaf_health: std.atomic.Value(bool) = .init(false),
    /// When set, the reply's bytes go out in two writes a pause apart, the second taking this many of
    /// the reply's trailing bytes: a read's edge falls inside the id when the body is cut there.
    trailing_second: usize = 0,
    /// The body of `GET /v1/models`, answered and not kept; null leaves it to the general answer.
    models: ?[]const u8 = null,

    // the reply's line and body, as the next request is given them
    status_line: []const u8 = "HTTP/1.1 200 OK",
    body: []const u8 = "{}",
    content_type: []const u8 = "application/json",

    requests_mutex: std.Io.Mutex = .init,
    requests: std.ArrayList(Request) = .empty,

    const Request = struct { path: []u8, body: []u8 };

    fn start(gpa: Allocator, name: []const u8) !*Fake {
        const self = try gpa.create(Fake);
        self.* = .{ .gpa = gpa, .name = name, .listener = 0, .port = 0, .address = undefined };
        self.listener = try http.listen(.{ .ip4 = .loopback(0) });
        self.port = portOf(self.listener);
        self.address = .{ .ip4 = .loopback(self.port) };
        self.thread = try std.Thread.spawn(.{}, serve, .{self});
        return self;
    }

    /// Stop answering (a dead copy), keeping the port for the take-back test's restart.
    fn die(self: *Fake) void {
        self.stop.store(true, .release);
        _ = posix.system.close(self.listener);
        self.thread.?.join();
        self.thread = null;
        self.stop = .init(false);
    }

    fn restart(self: *Fake) !void {
        self.listener = try http.listen(.{ .ip4 = .loopback(self.port) });
        self.thread = try std.Thread.spawn(.{}, serve, .{self});
    }

    fn end(self: *Fake) void {
        if (self.thread) |thread| {
            self.stop.store(true, .release);
            _ = posix.system.close(self.listener);
            thread.join();
            self.thread = null;
        }
        for (self.requests.items) |request| {
            self.gpa.free(request.path);
            self.gpa.free(request.body);
        }
        self.requests.deinit(self.gpa);
        self.gpa.destroy(self);
    }

    /// Waits until `n` requests were taken, at most ten seconds.
    fn waitFor(self: *Fake, n: usize) void {
        var waited: usize = 0;
        while (self.count.load(.acquire) < n and waited < 10_000) : (waited += 1) {
            _ = std.posix.poll(&[_]posix.pollfd{}, 1) catch 0;
        }
    }

    fn serve(self: *Fake) void {
        while (!self.stop.load(.acquire)) {
            var fds = [_]posix.pollfd{.{ .fd = self.listener, .events = posix.POLL.IN, .revents = 0 }};
            const ready = posix.poll(&fds, 50) catch break;
            if (ready == 0) continue;
            const rc = posix.system.accept(self.listener, null, null);
            if (posix.errno(rc) != .SUCCESS) continue;
            const fd: posix.socket_t = @intCast(rc);
            self.answerOne(fd);
            _ = posix.system.close(fd);
        }
    }

    fn answerOne(self: *Fake, fd: posix.socket_t) void {
        var head_buf: [16 * 1024]u8 = undefined;
        const head = http.readHead(fd, &head_buf) catch return;
        // the router's health probe is answered and not kept, as a copy's own /health is
        const path = if (std.mem.indexOfScalar(u8, head.target, '?')) |q| head.target[0..q] else head.target;
        // as a `serve` bound to the loopback (the vendored `request_guard.zig`): a host that is not the
        // loopback's is refused, so the router's own reads must name the copy's address (spec 139 WS-01)
        var own_buf: [32]u8 = undefined;
        const own = std.fmt.bufPrint(&own_buf, "127.0.0.1:{d}", .{self.port}) catch return;
        if (std.mem.eql(u8, head.method, "GET") and !std.mem.eql(u8, head.header("host") orelse "", own)) {
            http.writeAll(fd, "HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\nconnection: close\r\n\r\n") catch {};
            return;
        }
        if (std.mem.eql(u8, head.method, "GET") and (std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/healthz"))) {
            if (self.deaf_health.load(.acquire)) return; // silence: the probe's read times out
            http.writeAll(fd, "HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n") catch {};
            return;
        }
        // the router's own read of the models is answered and not kept; a copy with no list says `{}`
        if (std.mem.eql(u8, head.method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
            const models = self.models orelse "{}";
            var reply_buf: [2048]u8 = undefined;
            const reply = std.fmt.bufPrint(&reply_buf, "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ models.len, models }) catch return;
            http.writeAll(fd, reply) catch {};
            return;
        }
        const length = http.contentLength(head) catch 0;
        const body = self.gpa.alloc(u8, length) catch return;
        defer self.gpa.free(body);
        http.readBody(fd, head.extra, body) catch return;

        self.requests_mutex.lockUncancelable(testing.io);
        self.requests.append(self.gpa, .{
            .path = self.gpa.dupe(u8, head.target) catch @panic("oom"),
            .body = self.gpa.dupe(u8, body) catch @panic("oom"),
        }) catch @panic("oom");
        self.requests_mutex.unlock(testing.io);
        _ = self.count.fetchAdd(1, .acq_rel);

        if (self.hold.load(.acquire)) {
            // the request is in flight until the test lets it go
            var waited: usize = 0;
            while (self.hold.load(.acquire) and waited < 60_000) : (waited += 1) {
                _ = std.posix.poll(&[_]posix.pollfd{}, 1) catch 0;
            }
        }
        var out: std.ArrayList(u8) = .empty;
        var num: [32]u8 = undefined;
        defer out.deinit(self.gpa);
        out.appendSlice(self.gpa, self.status_line) catch return;
        out.appendSlice(self.gpa, "\r\ncontent-type: ") catch return;
        out.appendSlice(self.gpa, self.content_type) catch return;
        out.appendSlice(self.gpa, "\r\ncontent-length: ") catch return;
        out.appendSlice(self.gpa, std.fmt.bufPrint(&num, "{d}", .{self.body.len}) catch return) catch return;
        out.appendSlice(self.gpa, "\r\nconnection: close\r\n\r\n") catch return;
        out.appendSlice(self.gpa, self.body) catch return;
        if (self.trailing_second > 0 and out.items.len > self.trailing_second) {
            // the reply in two writes, the pause so the router's read ends between them
            const at = out.items.len - self.trailing_second;
            http.writeAll(fd, out.items[0..at]) catch return;
            _ = posix.poll(&[_]posix.pollfd{}, 200) catch 0;
            http.writeAll(fd, out.items[at..]) catch {};
            return;
        }
        http.writeAll(fd, out.items) catch {};
    }
};

fn portOf(listener: posix.socket_t) u16 {
    var storage align(4) = std.mem.zeroes([64]u8);
    var len: posix.socklen_t = 64;
    if (posix.errno(posix.system.getsockname(listener, @ptrCast(&storage), &len)) != .SUCCESS) return 0;
    const sin: *const posix.sockaddr.in = @ptrCast(@alignCast(&storage));
    return std.mem.bigToNative(u16, sin.port);
}

/// The two fake copies and the router over them, its listener on the loopback.
const Stand = struct {
    gpa: Allocator,
    a: *Fake,
    b: *Fake,
    router: *Router,
    listener: posix.socket_t,
    port: u16,
    serve_thread: std.Thread,

    fn start(gpa: Allocator, given: Config) !*Stand {
        const self = try gpa.create(Stand);
        self.* = .{ .gpa = gpa, .a = undefined, .b = undefined, .router = undefined, .listener = 0, .port = 0, .serve_thread = undefined };
        self.a = try Fake.start(gpa, "a");
        errdefer self.a.end();
        self.b = try Fake.start(gpa, "b");
        errdefer self.b.end();
        self.router = try Router.create(gpa, given, &.{
            .{ .name = "a", .address = self.a.address },
            .{ .name = "b", .address = self.b.address },
        });
        errdefer self.router.deinit();
        self.listener = try http.listen(.{ .ip4 = .loopback(0) });
        self.port = portOf(self.listener);
        self.serve_thread = try std.Thread.spawn(.{}, Router.serve, .{ self.router, self.listener });
        return self;
    }

    fn end(self: *Stand) void {
        self.router.deinit();
        _ = posix.system.close(self.listener);
        self.serve_thread.join();
        self.a.end();
        self.b.end();
        self.gpa.destroy(self);
    }

    /// The router's health line for one copy, as `healthJson` says it.
    fn stateOf(self: *Stand, name: []const u8) []const u8 {
        var buf: [4096]u8 = undefined;
        const said = self.router.healthJson(&buf);
        var needle_buf: [64]u8 = undefined;
        const needle = std.fmt.bufPrint(&needle_buf, "\"name\":\"{s}\"", .{name}) catch return "";
        const at = std.mem.indexOf(u8, said, needle) orelse return "";
        const rest = said[at..];
        const end_at = std.mem.indexOf(u8, rest, "}") orelse return "";
        return rest[0..end_at];
    }
};

/// A request sent to the router and its whole reply, to the connection's end.
fn post(arena: Allocator, port: u16, path: []const u8, body: []const u8, headers: []const u8) ![]const u8 {
    const asked = try std.fmt.allocPrint(arena, "POST {s} HTTP/1.1\r\nhost: test\r\n{s}content-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ path, headers, body.len, body });
    const fd = try sent(port, asked);
    defer _ = posix.system.close(fd);
    return heard(arena, fd);
}

/// A request whose connection stays open: the test reads its reply after the fake is released.
fn postOpen(arena: Allocator, port: u16, path: []const u8, body: []const u8) !posix.socket_t {
    const asked = try std.fmt.allocPrint(arena, "POST {s} HTTP/1.1\r\nhost: test\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ path, body.len, body });
    return sent(port, asked);
}

fn heard(arena: Allocator, fd: posix.socket_t) ![]const u8 {
    var reply: std.ArrayList(u8) = .empty;
    var buffer: [4096]u8 = undefined;
    while (true) {
        const got = posix.read(fd, &buffer) catch break;
        if (got == 0) break;
        try reply.appendSlice(arena, buffer[0..got]);
    }
    return reply.items;
}

fn bodyOf(reply: []const u8) []const u8 {
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return "";
    return reply[at + 4 ..];
}

fn config() Config {
    return .{
        .io = testing.io,
        .parallel = 4,
        .poll_ms = 30,
        .open_ms = 300,
        .failure_limit = 3,
        .connect_timeout_ms = 300,
        .health_timeout_ms = 300,
        .table_entries = 64,
    };
}

fn chatBody(arena: Allocator, system: []const u8, user: []const u8) []const u8 {
    return std.fmt.allocPrint(arena, "{{\"messages\":[{{\"role\":\"system\",\"content\":\"{s}\"}},{{\"role\":\"user\",\"content\":\"{s}\"}}]}}", .{ system, user }) catch @panic("oom");
}

test "a second turn reaches its copy, a header's key and a previous_response_id theirs, a miss the idler one" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const stand = try Stand.start(gpa, config());
    defer stand.end();

    // A conversation's second turn reaches the copy its first turn went to (a fresh key's tie is the
    // lower index, so the copy is a).
    const turn1 = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the sea"), "");
    try testing.expect(std.mem.indexOf(u8, turn1, "HTTP/1.1 200") != null);
    const turn2 = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the sea"), "");
    try testing.expect(std.mem.indexOf(u8, turn2, "HTTP/1.1 200") != null);
    stand.a.waitFor(2);
    try testing.expectEqual(@as(usize, 0), stand.b.count.load(.acquire));

    // A header's key goes to its copy whatever the opening under it: a's key count is one higher, so the
    // keyed request's first sight is b.
    const keyed1 = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the mountain"), "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, keyed1, "HTTP/1.1 200") != null);
    stand.b.waitFor(1);
    const keyed2 = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "long winded", "another thing"), "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, keyed2, "HTTP/1.1 200") != null);
    stand.b.waitFor(2);

    // A Responses conversation follows its previous_response_id past a changed opening: the create went
    // by alpha's key (to b, as keyed1's), the continuation is routed by the id alone.
    stand.b.body = "{\"id\":\"resp_1\",\"copy\":\"b\"}";
    const made = try post(arena, stand.port, "/v1/responses", "{\"input\":\"a thing\"}", "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, made, "HTTP/1.1 200") != null);
    stand.b.waitFor(3);
    const followed = try post(arena, stand.port, "/v1/responses", "{\"input\":\"another thing entirely\",\"previous_response_id\":\"resp_1\"}", "");
    try testing.expect(std.mem.indexOf(u8, followed, "\"copy\":\"b\"") != null);
    stand.b.waitFor(4);
    try testing.expectEqual(@as(usize, 2), stand.a.count.load(.acquire));

    // A miss goes to the idler copy: a held at one request in flight, a fresh opening's request arrives.
    stand.a.hold.store(true, .release);
    const held_fd = try postOpen(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the held one"));
    stand.a.waitFor(3);
    const elsewhere = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "a third thing"), "");
    try testing.expect(std.mem.indexOf(u8, elsewhere, "HTTP/1.1 200") != null);
    stand.b.waitFor(5);
    stand.a.hold.store(false, .release);
    const held_reply = try heard(arena, held_fd);
    try testing.expect(std.mem.indexOf(u8, held_reply, "HTTP/1.1 200") != null);
}

test "a dead copy is left and retaken" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const stand = try Stand.start(gpa, config());
    defer stand.end();

    // The beta key's copy is b (a holds one key from the first, keyless request).
    stand.a.body = "{\"copy\":\"a\"}";
    stand.b.body = "{\"copy\":\"b\"}";
    _ = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "s", "the first"), "");
    stand.a.waitFor(1);
    const first = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "s", "the sea"), "x-zspark-routing-key: beta\r\n");
    try testing.expect(std.mem.indexOf(u8, first, "\"copy\":\"b\"") != null);
    stand.b.waitFor(1);

    // It dies: three probes in a row leave it, and its key's request goes to the survivor, which has the
    // key moved to it.
    stand.b.die();
    var waited: usize = 0;
    while (waited < 5_000 and std.mem.indexOf(u8, stand.stateOf("b"), "open") == null) : (waited += 1) {
        _ = std.posix.poll(&[_]posix.pollfd{}, 2) catch 0;
    }
    try testing.expect(std.mem.indexOf(u8, stand.stateOf("b"), "open") != null);
    stand.b.count.store(0, .release);
    const survivor = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "s", "the sea"), "x-zspark-routing-key: beta\r\n");
    try testing.expect(std.mem.indexOf(u8, survivor, "HTTP/1.1 200") != null);
    try testing.expect(std.mem.indexOf(u8, survivor, "\"copy\":\"a\"") != null);
    _ = std.posix.poll(&[_]posix.pollfd{}, 20) catch 0;
    try testing.expectEqual(@as(usize, 0), stand.b.count.load(.acquire));

    // It is back: the poll's probe takes it, and a keyless conversation is met there (its keys are still
    // all on a, so the fewer-keys tie-break gives b the next one). Every try is its own opening, so no
    // try is a key's hit on a.
    try stand.b.restart();
    stand.b.body = "{\"copy\":\"b\"}";
    waited = 0;
    while (waited < 30_000) : (waited += 1) {
        const said = try std.fmt.allocPrint(arena, "the sea after the return {d}", .{waited});
        const reply = post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "s", said), "") catch break;
        if (std.mem.indexOf(u8, reply, "\"copy\":\"b\"") != null) break;
        _ = std.posix.poll(&[_]posix.pollfd{}, 2) catch 0;
    }
    try testing.expect(std.mem.indexOf(u8, stand.stateOf("b"), "closed") != null);
    try testing.expect(stand.b.count.load(.acquire) > 0);
}

test "a deaf copy is left and the poll's own probe takes it back" {
    const gpa = testing.allocator;

    const stand = try Stand.start(gpa, config());
    defer stand.end();

    // Three silent /healths leave the copy; no request of a client ever fails, none is even sent.
    stand.b.deaf_health.store(true, .release);
    var waited: usize = 0;
    while (waited < 10_000 and std.mem.indexOf(u8, stand.stateOf("b"), "open") == null) : (waited += 1) {
        _ = std.posix.poll(&[_]posix.pollfd{}, 2) catch 0;
    }
    try testing.expect(std.mem.indexOf(u8, stand.stateOf("b"), "open") != null);
    try testing.expectEqual(@as(usize, 0), stand.a.count.load(.acquire));
    try testing.expectEqual(@as(usize, 0), stand.b.count.load(.acquire));

    // The copy answers its health again: the poll's probe (the half-open one once open_ms has gone)
    // takes it back on its own, and no client request took part in that.
    stand.b.deaf_health.store(false, .release);
    waited = 0;
    while (waited < 30_000 and std.mem.indexOf(u8, stand.stateOf("b"), "closed") == null) : (waited += 1) {
        _ = std.posix.poll(&[_]posix.pollfd{}, 2) catch 0;
    }
    try testing.expect(std.mem.indexOf(u8, stand.stateOf("b"), "closed") != null);
    try testing.expectEqual(@as(usize, 0), stand.a.count.load(.acquire));
    try testing.expectEqual(@as(usize, 0), stand.b.count.load(.acquire));
}

test "a host's reply, its id spaced after the colon, is kept on the copy that made it" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const stand = try Stand.start(gpa, config());
    defer stand.end();

    // The real host's JSON writes the id `"id": "`, a space after the colon; the vendored host's
    // reply is that shape. The create lands on b (a holds one key from one turn, alpha's first sight
    // is b, whose side is then even), and the continuation must find its id on b: were the id lost,
    // the continuation would be a miss whose tie-break is the lower index, a.
    stand.a.body = "{\"copy\":\"a\"}";
    _ = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the sea"), "");
    stand.a.waitFor(1);
    stand.b.body = "{\"id\": \"resp_spaced\",\"copy\":\"b\"}";
    const made = try post(arena, stand.port, "/v1/responses", "{\"input\":\"a thing\"}", "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, made, "HTTP/1.1 200") != null);
    try testing.expect(std.mem.indexOf(u8, made, "\"copy\":\"b\"") != null);
    stand.b.waitFor(1);
    const followed = try post(arena, stand.port, "/v1/responses", "{\"input\":\"another thing entirely\",\"previous_response_id\":\"resp_spaced\"}", "");
    try testing.expect(std.mem.indexOf(u8, followed, "\"copy\":\"b\"") != null);
}

test "an id cut by a read's edge is taken whole" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const stand = try Stand.start(gpa, config());
    defer stand.end();

    // The fake's reply goes out in two writes a pause apart, the second taking the reply's last 18
    // bytes, from `split` on: the router's first read ends inside the id, and the capture must go on
    // where it stopped.
    stand.a.body = "{\"copy\":\"a\"}";
    stand.b.body = "{\"id\": \"resp_split\",\"copy\":\"b\"}";
    stand.b.trailing_second = 18;
    _ = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the sea"), "");
    stand.a.waitFor(1);
    _ = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the sea"), "");
    stand.a.waitFor(2);
    const made = try post(arena, stand.port, "/v1/responses", "{\"input\":\"a thing\"}", "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, made, "HTTP/1.1 200") != null);
    stand.b.waitFor(1);
    const followed = try post(arena, stand.port, "/v1/responses", "{\"input\":\"another thing entirely\",\"previous_response_id\":\"resp_split\"}", "");
    try testing.expect(std.mem.indexOf(u8, followed, "\"copy\":\"b\"") != null);
}

test "a GET of a response goes where the response lives" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const stand = try Stand.start(gpa, config());
    defer stand.end();

    // The create on b (a warmed as the other tests warm it), then the id fetched by GET: it must go
    // to the copy that holds the response, not to the idler.
    _ = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the sea"), "");
    stand.a.waitFor(1);
    _ = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "the sea"), "");
    stand.a.waitFor(2);
    stand.a.body = "{\"copy\":\"a\"}";
    stand.b.body = "{\"id\": \"resp_stored\",\"copy\":\"b\"}";
    const made = try post(arena, stand.port, "/v1/responses", "{\"input\":\"a thing\"}", "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, made, "HTTP/1.1 200") != null);
    stand.b.waitFor(1);
    const asked = try std.fmt.allocPrint(arena, "GET /v1/responses/resp_stored HTTP/1.1\r\nhost: test\r\nconnection: close\r\n\r\n", .{});
    const fd = try sent(stand.port, asked);
    defer _ = posix.system.close(fd);
    const fetched = try heard(arena, fd);
    try testing.expect(std.mem.indexOf(u8, fetched, "HTTP/1.1 200") != null);
    try testing.expect(std.mem.indexOf(u8, fetched, "\"copy\":\"b\"") != null);
}

test "a session keeps its re-routed copy across the table's wrap" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const stand = try Stand.start(gpa, config()); // the key table's 64 entries
    defer stand.end();

    stand.a.body = "{\"copy\":\"a\"}";
    stand.b.body = "{\"copy\":\"b\"}";
    // Fresh keys fill the table to one short of full, so alpha's entry lands on the ring's last slot.
    // Each synchronous reply confirms its request was taken, whichever copy handled it.
    var i: usize = 0;
    while (i < 63) : (i += 1) {
        const said = try std.fmt.allocPrint(arena, "the sea {d}", .{i});
        const filled = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", said), "");
        try testing.expect(std.mem.indexOf(u8, filled, "HTTP/1.1 200") != null);
    }
    const on_b = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "alpha's own"), "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, on_b, "\"copy\":\"b\"") != null);

    // b is left: alpha's next turn goes to a. A blind append would write its new entry at index 0
    // while its stale b record lives on at the ring's last slot.
    stand.b.die();
    var waited: usize = 0;
    while (waited < 5_000 and std.mem.indexOf(u8, stand.stateOf("b"), "open") == null) : (waited += 1) {
        _ = std.posix.poll(&[_]posix.pollfd{}, 2) catch 0;
    }
    try testing.expect(std.mem.indexOf(u8, stand.stateOf("b"), "open") != null);
    const moved = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "alpha's own"), "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, moved, "\"copy\":\"a\"") != null);

    // b is taken back: the session stays on a — the table holds alpha once, at a
    try stand.b.restart();
    waited = 0;
    while (waited < 10_000 and std.mem.indexOf(u8, stand.stateOf("b"), "closed") == null) : (waited += 1) {
        _ = std.posix.poll(&[_]posix.pollfd{}, 2) catch 0;
    }
    try testing.expect(std.mem.indexOf(u8, stand.stateOf("b"), "closed") != null);
    const stayed = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "be brief", "alpha's own"), "x-zspark-routing-key: alpha\r\n");
    try testing.expect(std.mem.indexOf(u8, stayed, "\"copy\":\"a\"") != null);
}

test "a stream's bytes equal the copy's" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const stand = try Stand.start(gpa, config());
    defer stand.end();

    stand.a.content_type = "text/event-stream";
    stand.a.body = "data: {\"delta\":\"a\"}\n\ndata: {\"delta\":\"b\"}\n\ndata: [DONE]\n\n";
    const reply = try post(arena, stand.port, "/v1/chat/completions", chatBody(arena, "s", "u"), "");
    try testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200 OK\r\n"));
    try testing.expect(std.mem.indexOf(u8, reply, "text/event-stream") != null);
    try testing.expect(std.mem.indexOf(u8, reply, "connection: close") != null);
    try testing.expectEqualStrings(stand.a.body, bodyOf(reply));
}

test "a full copy queues its requests in arrival order" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var one = config();
    one.parallel = 1;
    const stand = try Stand.start(gpa, one);
    defer stand.end();

    // The copy is full with one held request; the next with its opening waits for it; a new opening's
    // request goes to the other copy at once.
    stand.a.body = "{\"copy\":\"a\"}";
    stand.b.body = "{\"copy\":\"b\"}";
    const body1 = chatBody(arena, "s", "u");
    const body2 = chatBody(arena, "s", "another user ask");
    stand.a.hold.store(true, .release);
    const held_fd = try postOpen(arena, stand.port, "/v1/chat/completions", body1);
    stand.a.waitFor(1);

    const Waiter = struct {
        fn run(s: *Stand, body: []const u8, out: *std.atomic.Value(bool)) void {
            var a: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
            defer a.deinit();
            const reply = post(a.allocator(), s.port, "/v1/chat/completions", body, "") catch return;
            out.store(std.mem.indexOf(u8, reply, "HTTP/1.1 200") != null, .release);
        }
    };
    var queued_ok = std.atomic.Value(bool).init(false);
    var missed_ok = std.atomic.Value(bool).init(false);
    const queued = try std.Thread.spawn(.{}, Waiter.run, .{ stand, body1, &queued_ok });
    const missed = try std.Thread.spawn(.{}, Waiter.run, .{ stand, body2, &missed_ok });
    _ = std.posix.poll(&[_]posix.pollfd{}, 300) catch 0; // the queued request had its time to queue and not run
    stand.a.hold.store(false, .release);
    queued.join();
    missed.join();
    try testing.expect(queued_ok.load(.acquire));
    try testing.expect(missed_ok.load(.acquire));
    stand.a.waitFor(2);
    stand.b.waitFor(1);
    const held_reply = try heard(arena, held_fd);
    try testing.expect(std.mem.indexOf(u8, held_reply, "HTTP/1.1 200") != null);
    try testing.expectEqual(@as(usize, 2), stand.a.count.load(.acquire));
    try testing.expectEqualStrings(body1, stand.a.requests.items[1].body);
    try testing.expectEqual(@as(usize, 1), stand.b.count.load(.acquire));
    try testing.expectEqualStrings(body2, stand.b.requests.items[0].body);
}

test "the routing key is the opening's: its system, tools and first user message" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const open = router_file.openingKey;
    const api = router_file.apiOf;

    const a = "{\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},{\"role\":\"user\",\"content\":\"the sea\"},{\"role\":\"user\",\"content\":\"the mountain\"}]}";
    // the same conversation a turn later: the opening is the same where the system and the first user
    // message are, so the key is
    const later = "{\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},{\"role\":\"user\",\"content\":\"the sea\"},{\"role\":\"assistant\",\"content\":\"...\"},{\"role\":\"user\",\"content\":\"more\"}]}";
    try testing.expectEqual(open(arena, .chat, try parse(arena, a)).?, open(arena, .chat, try parse(arena, later)).?);

    // a different first user message, a different key
    const b = "{\"messages\":[{\"role\":\"system\",\"content\":\"be brief\"},{\"role\":\"user\",\"content\":\"the mountain\"}]}";
    try testing.expect(open(arena, .chat, try parse(arena, a)).? != open(arena, .chat, try parse(arena, b)).?);

    // tools are part of the opening
    const with_tools = "{\"tools\":[{\"name\":\"f\"}],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    const with_tools_2 = "{\"tools\":[{\"name\":\"g\"}],\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]}";
    try testing.expect(open(arena, .chat, try parse(arena, with_tools)).? != open(arena, .chat, try parse(arena, with_tools_2)).?);

    // Responses: instructions, tools and the first user input; the string input and the one-item message
    // array that carries it are the same key
    const responses_body = "{\"instructions\":\"be brief\",\"input\":\"the sea\"}";
    const responses_body2 = "{\"instructions\":\"be brief\",\"input\":[{\"role\":\"user\",\"content\":\"the sea\"}]}";
    try testing.expectEqual(open(arena, .responses, try parse(arena, responses_body)).?, open(arena, .responses, try parse(arena, responses_body2)).?);

    // the same opening on two APIs is two different keys
    const chat_key = open(arena, .chat, try parse(arena, "{\"messages\":[{\"role\":\"user\",\"content\":\"the sea\"}]}")).?;
    const messages_key = open(arena, .messages, try parse(arena, "{\"system\":null,\"messages\":[{\"role\":\"user\",\"content\":\"the sea\"}]}")).?;
    try testing.expect(chat_key != messages_key);

    try testing.expectEqual(api("/v1/chat/completions"), .chat);
    try testing.expectEqual(api("/v1/responses"), .responses);
    try testing.expectEqual(api("/v1/messages"), .messages);
    try testing.expectEqual(api("/v1/completions"), .completions);
    try testing.expectEqual(api("/v1/other"), .none);
}

fn parse(arena: Allocator, text: []const u8) !std.json.Value {
    return std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{});
}

// ---- models (spec 139 WS-02): copies of different models behind one address ------------------------ //

/// n fakes and the router over them; the first copy is the default, `audio` marks the audio ones.
const Fleet = struct {
    gpa: Allocator,
    fakes: []*Fake,
    router: *Router,
    listener: posix.socket_t,
    port: u16,
    serve_thread: std.Thread,

    const Member = struct { name: []const u8, ids: []const u8, audio: bool = false, parallel: ?usize = null };

    fn start(gpa: Allocator, members: []const Member) !*Fleet {
        const self = try gpa.create(Fleet);
        self.gpa = gpa;
        self.fakes = try gpa.alloc(*Fake, members.len);
        const specs = try gpa.alloc(router_file.CopySpec, members.len);
        defer gpa.free(specs);
        for (members, 0..) |member, at| {
            const fake = try Fake.start(gpa, member.name);
            fake.body = try std.fmt.allocPrint(gpa, "{{\"from\":\"{s}\"}}", .{member.name});
            // the card list as `serve` writes it: one object a name
            var cards: std.ArrayList(u8) = .empty;
            defer cards.deinit(gpa);
            var names = std.mem.tokenizeScalar(u8, member.ids, ',');
            while (names.next()) |id| {
                if (cards.items.len > 0) try cards.append(gpa, ',');
                try cards.print(gpa, "{{\"id\":\"{s}\",\"object\":\"model\",\"owned_by\":\"tensorfold\"}}", .{id});
            }
            fake.models = try std.fmt.allocPrint(gpa, "{{\"object\":\"list\",\"data\":[{s}]}}", .{cards.items});
            self.fakes[at] = fake;
            specs[at] = .{ .name = member.name, .address = fake.address, .audio = member.audio, .parallel = member.parallel };
        }
        self.router = try Router.create(gpa, config(), specs);
        self.listener = try http.listen(.{ .ip4 = .loopback(0) });
        self.port = portOf(self.listener);
        self.serve_thread = try std.Thread.spawn(.{}, Router.serve, .{ self.router, self.listener });
        return self;
    }

    fn end(self: *Fleet) void {
        self.router.deinit();
        _ = posix.system.close(self.listener);
        self.serve_thread.join();
        for (self.fakes) |fake| {
            self.gpa.free(fake.body);
            self.gpa.free(fake.models.?);
            fake.end();
        }
        self.gpa.free(self.fakes);
        self.gpa.destroy(self);
    }

    /// Waits until the router has read every copy's models (its `/health` is 200), at most ten seconds.
    fn ready(self: *Fleet, arena: Allocator) !void {
        var waited: usize = 0;
        while (waited < 1000) : (waited += 1) {
            const reply = try get(arena, self.port, "/health");
            if (std.mem.indexOf(u8, reply, "HTTP/1.1 200") != null) return;
            _ = posix.poll(&[_]posix.pollfd{}, 10) catch 0;
        }
        return error.NotReady;
    }
};

fn get(arena: Allocator, port: u16, path: []const u8) ![]const u8 {
    const asked = try std.fmt.allocPrint(arena, "GET {s} HTTP/1.1\r\nhost: test\r\nconnection: close\r\n\r\n", .{path});
    const fd = try sent(port, asked);
    defer _ = posix.system.close(fd);
    return heard(arena, fd);
}

const trio = [_]Fleet.Member{
    .{ .name = "flash", .ids = "Qwen3.8-Flash-Next-FP4-Local,flash" },
    .{ .name = "ocr", .ids = "Unlimited-OCR" },
    .{ .name = "asr", .ids = "Qwen3-ASR-1.7B", .audio = true },
};

test "a body's model picks the copy that lists it, a missing model the default, an unknown one vLLM's 404" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fleet = try Fleet.start(gpa, &trio);
    defer fleet.end();
    try fleet.ready(arena);

    const ocr = try post(arena, fleet.port, "/v1/chat/completions", "{\"model\":\"Unlimited-OCR\",\"messages\":[{\"role\":\"user\",\"content\":\"read\"}]}", "");
    try testing.expect(std.mem.indexOf(u8, ocr, "\"from\":\"ocr\"") != null);
    // a chat naming the audio copy's model goes to it, as its transcriptions do (Qwen3-ASR's chat, spec 150)
    const asr = try post(arena, fleet.port, "/v1/chat/completions", "{\"model\":\"Qwen3-ASR-1.7B\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"AAAA\",\"format\":\"wav\"}}]}]}", "");
    try testing.expect(std.mem.indexOf(u8, asr, "\"from\":\"asr\"") != null);
    // an alias, and a route the router keys nothing about but the model still routes (tokenize)
    const alias = try post(arena, fleet.port, "/v1/completions", "{\"model\":\"flash\",\"prompt\":\"hi\"}", "");
    try testing.expect(std.mem.indexOf(u8, alias, "\"from\":\"flash\"") != null);
    const tokens = try post(arena, fleet.port, "/tokenize", "{\"model\":\"Unlimited-OCR\",\"prompt\":\"hi\"}", "");
    try testing.expect(std.mem.indexOf(u8, tokens, "\"from\":\"ocr\"") != null);
    // the same opening to another model is not the first one's conversation
    const same_opening = "\"messages\":[{\"role\":\"user\",\"content\":\"read\"}]";
    const other = try post(arena, fleet.port, "/v1/chat/completions", try std.fmt.allocPrint(arena, "{{\"model\":\"flash\",{s}}}", .{same_opening}), "");
    try testing.expect(std.mem.indexOf(u8, other, "\"from\":\"flash\"") != null);

    // no model, an empty one, a body that is not JSON: the default copy's model (vLLM's first name)
    const bare = try post(arena, fleet.port, "/v1/chat/completions", chatBody(arena, "s", "u"), "");
    try testing.expect(std.mem.indexOf(u8, bare, "\"from\":\"flash\"") != null);
    const empty = try post(arena, fleet.port, "/v1/chat/completions", "{\"model\":\"\",\"messages\":[]}", "");
    try testing.expect(std.mem.indexOf(u8, empty, "\"from\":\"flash\"") != null);

    // an unknown model: vLLM's NotFoundError, and no copy is asked
    const before = fleet.fakes[0].count.load(.acquire) + fleet.fakes[1].count.load(.acquire) + fleet.fakes[2].count.load(.acquire);
    const unknown = try post(arena, fleet.port, "/v1/chat/completions", "{\"model\":\"nope\",\"messages\":[]}", "");
    try testing.expect(std.mem.indexOf(u8, unknown, "HTTP/1.1 404") != null);
    const body = bodyOf(unknown);
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{});
    const err = parsed.object.get("error").?.object;
    try testing.expectEqualStrings("The model `nope` does not exist.", err.get("message").?.string);
    try testing.expectEqualStrings("NotFoundError", err.get("type").?.string);
    try testing.expectEqualStrings("model", err.get("param").?.string);
    try testing.expectEqual(@as(i64, 404), err.get("code").?.integer);
    const after = fleet.fakes[0].count.load(.acquire) + fleet.fakes[1].count.load(.acquire) + fleet.fakes[2].count.load(.acquire);
    try testing.expectEqual(before, after);
}

test "every path under /v1/audio/ goes to the audio copy, its multipart body unread" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const fleet = try Fleet.start(gpa, &trio);
    defer fleet.end();
    try fleet.ready(arena);

    const multipart = "--b\r\ncontent-disposition: form-data; name=\"model\"\r\n\r\nUnlimited-OCR\r\n--b\r\ncontent-disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\n\x00\x01RIFF\xff\r\n--b--\r\n";
    const asked = try std.fmt.allocPrint(arena, "POST /v1/audio/transcriptions HTTP/1.1\r\nhost: test\r\ncontent-type: multipart/form-data; boundary=b\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n{s}", .{ multipart.len, multipart });
    const fd = try sent(fleet.port, asked);
    defer _ = posix.system.close(fd);
    const reply = try heard(arena, fd);
    // the form's own `model` names another copy's: the path decides, and the bytes arrive as sent
    try testing.expect(std.mem.indexOf(u8, reply, "\"from\":\"asr\"") != null);
    fleet.fakes[2].waitFor(1);
    try testing.expectEqualStrings(multipart, fleet.fakes[2].requests.items[0].body);
    try testing.expectEqual(@as(usize, 0), fleet.fakes[1].count.load(.acquire));

    const other = try post(arena, fleet.port, "/v1/audio/translations", "{}", "");
    try testing.expect(std.mem.indexOf(u8, other, "\"from\":\"asr\"") != null);

    // no audio copy configured: the router says so itself
    const bare = try Fleet.start(gpa, trio[0..2]);
    defer bare.end();
    try bare.ready(arena);
    const none = try post(arena, bare.port, "/v1/audio/transcriptions", "{}", "");
    try testing.expect(std.mem.indexOf(u8, none, "HTTP/1.1 404") != null);
}

test "/v1/models merges every copy's list once, and /health is 200 only while every model has a copy up" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // two replicas of Flash Next, OCR and ASR
    const members = [_]Fleet.Member{
        .{ .name = "flash0", .ids = "Qwen3.8-Flash-Next-FP4-Local" },
        .{ .name = "flash1", .ids = "Qwen3.8-Flash-Next-FP4-Local" },
        .{ .name = "ocr", .ids = "Unlimited-OCR" },
        .{ .name = "asr", .ids = "Qwen3-ASR-1.7B", .audio = true },
    };
    const fleet = try Fleet.start(gpa, &members);
    defer fleet.end();
    try fleet.ready(arena);

    const listed = bodyOf(try get(arena, fleet.port, "/v1/models"));
    const list = (try std.json.parseFromSliceLeaky(std.json.Value, arena, listed, .{})).object;
    try testing.expectEqualStrings("list", list.get("object").?.string);
    const data = list.get("data").?.array.items;
    try testing.expectEqual(@as(usize, 3), data.len);
    try testing.expectEqualStrings("Qwen3.8-Flash-Next-FP4-Local", data[0].object.get("id").?.string);
    try testing.expectEqualStrings("Unlimited-OCR", data[1].object.get("id").?.string);
    try testing.expectEqualStrings("Qwen3-ASR-1.7B", data[2].object.get("id").?.string);
    const one = try get(arena, fleet.port, "/v1/models/Unlimited-OCR");
    try testing.expect(std.mem.indexOf(u8, one, "\"id\":\"Unlimited-OCR\"") != null);
    try testing.expect(std.mem.indexOf(u8, try get(arena, fleet.port, "/v1/models/nope"), "HTTP/1.1 404") != null);

    // one replica of a model lost: the model still has a copy, so the router stays 200
    fleet.fakes[0].die();
    _ = posix.poll(&[_]posix.pollfd{}, 400) catch 0; // many polls: the copy is left by now
    try testing.expect(std.mem.indexOf(u8, try get(arena, fleet.port, "/health"), "HTTP/1.1 200") != null);
    // the only OCR copy lost: its model has none, the router is 503 until the copy answers again
    fleet.fakes[2].die();
    try waitHealth(arena, fleet.port, false, 5000);
    try testing.expect(std.mem.indexOf(u8, try get(arena, fleet.port, "/health"), "\"ok\":false") != null);
    try fleet.fakes[2].restart();
    try fleet.fakes[0].restart();
    try waitHealth(arena, fleet.port, true, 8000);
}

fn waitHealth(arena: Allocator, port: u16, want_ok: bool, limit_ms: usize) !void {
    var waited: usize = 0;
    while (waited < limit_ms / 10) : (waited += 1) {
        const reply = try get(arena, port, "/health");
        const ok = std.mem.indexOf(u8, reply, "HTTP/1.1 200") != null;
        if (ok == want_ok) return;
        _ = posix.poll(&[_]posix.pollfd{}, 10) catch 0;
    }
    return error.HealthNeverChanged;
}

test "each copy holds its own --parallel in flight: a full OCR queues while Flash Next takes more" {
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const members = [_]Fleet.Member{
        .{ .name = "flash", .ids = "flash" },
        .{ .name = "ocr", .ids = "Unlimited-OCR", .parallel = 1 },
    };
    const fleet = try Fleet.start(gpa, &members);
    defer fleet.end();
    try fleet.ready(arena);

    const Asker = struct {
        fn run(port: u16, body: []const u8, out: *std.atomic.Value(bool)) void {
            var a: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
            defer a.deinit();
            const reply = post(a.allocator(), port, "/v1/chat/completions", body, "") catch return;
            out.store(std.mem.indexOf(u8, reply, "HTTP/1.1 200") != null, .release);
        }
    };
    const flash_body = "{\"model\":\"flash\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]}";
    const ocr_body = "{\"model\":\"Unlimited-OCR\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]}";
    fleet.fakes[0].hold.store(true, .release);
    fleet.fakes[1].hold.store(true, .release);
    var ok: [4]std.atomic.Value(bool) = .{ .init(false), .init(false), .init(false), .init(false) };
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*thread, at| thread.* = try std.Thread.spawn(.{}, Asker.run, .{ fleet.port, if (at < 2) flash_body else ocr_body, &ok[at] });
    // Flash Next's limit is the config's 4: both its requests are in flight (the fake answers one at a time,
    // the router counts both); OCR's is 1: its second waits at the router, its fake never sees it
    const flash_two = "{\"name\":\"flash\",\"state\":\"closed\",\"healthy\":true,\"in_flight\":2}";
    const ocr_one = "{\"name\":\"ocr\",\"state\":\"closed\",\"healthy\":true,\"in_flight\":1}";
    var buf: [512]u8 = undefined;
    var waited: usize = 0;
    while (waited < 5000) : (waited += 1) {
        const said = fleet.router.healthJson(&buf);
        if (std.mem.indexOf(u8, said, flash_two) != null and std.mem.indexOf(u8, said, ocr_one) != null) break;
        _ = posix.poll(&[_]posix.pollfd{}, 1) catch 0;
    }
    _ = posix.poll(&[_]posix.pollfd{}, 200) catch 0; // OCR's second had its time to go and did not
    try testing.expect(std.mem.indexOf(u8, fleet.router.healthJson(&buf), ocr_one) != null);
    try testing.expect(std.mem.indexOf(u8, fleet.router.healthJson(&buf), flash_two) != null);
    try testing.expectEqual(@as(usize, 1), fleet.fakes[1].count.load(.acquire));
    fleet.fakes[0].hold.store(false, .release);
    fleet.fakes[1].hold.store(false, .release);
    for (threads) |thread| thread.join();
    for (&ok) |*each| try testing.expect(each.load(.acquire));
    try testing.expectEqual(@as(usize, 2), fleet.fakes[1].count.load(.acquire));
}

//! The two-copies router (spec 088): one address over `zspark serve` copies, each copy whole on its own
//! machine. A request stays on the copy that holds its conversation, by the key of its opening — the
//! header `x-zspark-routing-key` when given, else the hash of the body's opening (the system, the tools
//! and the first user message of chat completions and Messages; the instructions, tools and first user
//! input of Responses) — and a Responses conversation by the table its `previous_response_id` is kept in.
//! A request whose key and id are unknown goes to the least in-flight copy. A copy that fails is left
//! after `failure_limit` consecutive failures, for `open_ms` or until a `/health` probe (the half-open
//! probe, one a poll) answers, whichever comes later; while it is left everything goes to the survivor.
//! In-flight per copy is bounded at its own `serve`'s `--parallel`; past it a request waits in arrival
//! order. A reply's bytes pass through as the copy sent them, SSE and its framing included.
//!
//! Threading: a thread a client connection, a poll thread a copy, one mutex over the tables, breakers and
//! queues, one condition broadcast on every change. Routing keys and response ids live in fixed rings,
//! oldest turned over first; a restart of the router loses them (a conversation is then routed as new,
//! which re-prefills on the copy it meets).
//!
//! Models (spec 139): the copies need not be one model. Each copy's `GET /v1/models` is read when the
//! router starts, and again when a copy comes back, and its ids (the `--name` and aliases) are kept. A JSON
//! request (chat completions, completions, Responses, Messages, tokenize) goes by its body's `model` to
//! the copies that list it, then by the rules above among them; a missing `model` goes to the default
//! copy's model, as vLLM serves its first name; an unknown one gets vLLM's 404 body. Any `/v1/audio/`
//! path goes to the copies named audio, its body unread (multipart stays bytes). `GET /v1/models` is the
//! copies' lists merged; `/health` is 200 only when every model has a copy up. Copies that list no model
//! (an older `serve`, a fake) are one model, and everything is routed as before.

const std = @import("std");
const http = @import("http.zig");
const relay = @import("relay.zig");
const router_key = @import("router_key.zig");
const net = std.Io.net;
const posix = std.posix;
const Allocator = std.mem.Allocator;

/// The one header a client may pin a copy with (SGLang's `sticky`, zspark's name).
pub const routing_header = "x-zspark-routing-key";

pub const Config = struct {
    io: std.Io,
    /// The copies' own `--parallel`: the most requests in flight at one copy, where its spec gives none.
    parallel: usize = 4,
    poll_ms: u64 = 2000,
    /// How long a copy is left after its failures, at the least; the probe only answers after it.
    open_ms: u64 = 30_000,
    failure_limit: usize = 3,
    connect_timeout_ms: i32 = 2000,
    health_timeout_ms: i32 = 1000,
    /// The most a request body may be; the client gets 413 past it.
    max_body: usize = 64 << 20,
    head_limit: usize = 32 * 1024,
    /// Routing keys and response ids kept, oldest turned over first.
    table_entries: usize = 4096,
    /// The copy whose model a request without `model` is given (vLLM serves its first name).
    default_copy: usize = 0,
};

pub const CopySpec = struct {
    name: []const u8,
    address: net.IpAddress,
    /// Serves `/v1/audio/*`: those paths go to the audio copies by path, their body unread.
    audio: bool = false,
    /// This copy's own `--parallel` where it is not the config's (spec 139: OCR and ASR serve one at a time).
    parallel: ?usize = null,
};

/// One copy's guarded state and its queue of waiting requests, in arrival order.
const Copy = struct {
    name: []const u8,
    address: net.IpAddress,
    in_flight: usize = 0,
    /// The most requests in flight here: its `serve`'s `--parallel`.
    parallel: usize,
    failures: usize = 0,
    breaker: Breaker = .closed,
    opened_at: i96 = 0,
    healthy: bool = true,
    queue: std.ArrayList(*Waiter) = .empty,
    poll_thread: ?std.Thread = null,
    audio: bool = false,
    /// `/v1/models` was read: `ids` and `cards` are what it listed (none when the copy lists no model).
    discovered: bool = false,
    ids: [][]u8 = &.{},
    /// Each id's model card, the JSON the copy sent, for the merged list.
    cards: [][]u8 = &.{},

    fn lists(copy: *const Copy, model: []const u8) bool {
        for (copy.ids) |id| if (std.mem.eql(u8, id, model)) return true;
        return false;
    }

    fn freeModels(copy: *Copy, gpa: Allocator) void {
        for (copy.ids) |id| gpa.free(id);
        for (copy.cards) |card| gpa.free(card);
        if (copy.ids.len > 0) gpa.free(copy.ids);
        if (copy.cards.len > 0) gpa.free(copy.cards);
        copy.ids = &.{};
        copy.cards = &.{};
    }
};

/// The copies a request may go to, a bit a copy.
const Mask = struct {
    bits: [4]u64 = .{ 0, 0, 0, 0 },

    fn set(m: *Mask, at: usize) void {
        m.bits[at / 64] |= @as(u64, 1) << @intCast(at % 64);
    }
    fn has(m: Mask, at: usize) bool {
        return m.bits[at / 64] >> @intCast(at % 64) & 1 == 1;
    }
    fn any(m: Mask) bool {
        return (m.bits[0] | m.bits[1] | m.bits[2] | m.bits[3]) != 0;
    }
    fn all(n: usize) Mask {
        var m: Mask = .{};
        for (0..n) |at| m.set(at);
        return m;
    }
};

/// Why the router answers a request itself, before any copy.
const Refusal = enum { none, unknown_model, models_unknown, no_audio };

const Breaker = enum { closed, open };

/// A request's place in a copy's queue while it waits for a slot: the node's address is its identity.
const Waiter = struct { held: u8 = 0 };

/// How a request found its copy, as the reply's log line says it.
pub const How = enum { header, id, key, miss };

pub const Decision = struct {
    /// The copy to forward to; null when every copy is left and the client gets the 502.
    copy: ?usize,
    key: ?u64,
    how: How,
    /// Capture the reply's response id (a Responses request the client may continue from).
    arm_capture: bool,
    /// The copies it may go to.
    allow: Mask = .{},
    refusal: Refusal = .none,
    /// The body's `model`, for the 404.
    model: ?[]const u8 = null,
};

const KeyEntry = struct { key: u64 = 0, copy: u8 = 0, valid: bool = false };
const IdEntry = struct { id: [64]u8 = undefined, len: u8 = 0, copy: u8 = 0, valid: bool = false };

pub const Router = struct {
    gpa: Allocator,
    config: Config,
    copies: []Copy,
    mutex: std.Io.Mutex = .init,
    changed: std.Io.Condition = .init,
    log_mutex: std.Io.Mutex = .init,
    stop: std.atomic.Value(bool) = .init(false),
    open_connections: std.atomic.Value(usize) = .init(0),
    keys: []KeyEntry,
    key_at: usize = 0,
    ids: []IdEntry,
    id_at: usize = 0,

    pub fn create(gpa: Allocator, config: Config, specs: []const CopySpec) !*Router {
        if (specs.len == 0) return error.NoCopyGiven;
        if (specs.len > 255) return error.TooManyCopies;
        const self = try gpa.create(Router);
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .config = config,
            .copies = try gpa.alloc(Copy, specs.len),
            .keys = try gpa.alloc(KeyEntry, config.table_entries),
            .ids = try gpa.alloc(IdEntry, config.table_entries),
        };
        for (self.copies, specs) |*copy, spec| {
            copy.* = .{ .name = try gpa.dupe(u8, spec.name), .address = spec.address, .audio = spec.audio, .parallel = spec.parallel orelse config.parallel };
        }
        @memset(self.keys, .{});
        @memset(self.ids, .{});
        for (self.copies, 0..) |*copy, at| {
            copy.poll_thread = std.Thread.spawn(.{}, pollLoop, .{ self, at }) catch null;
        }
        return self;
    }

    pub fn deinit(self: *Router) void {
        self.stop.store(true, .release);
        for (self.copies) |*copy| if (copy.poll_thread) |thread| thread.join();
        // a connection's thread counts itself out after its socket is closed
        while (self.open_connections.load(.acquire) != 0) sleep(self.config.io, 2);
        for (self.copies) |*copy| {
            copy.queue.deinit(self.gpa);
            self.gpa.free(copy.name);
            copy.freeModels(self.gpa);
        }
        self.gpa.free(self.copies);
        self.gpa.free(self.keys);
        self.gpa.free(self.ids);
        self.gpa.destroy(self);
    }

    /// The accept loop, on the caller's thread, until `stop` is set.
    pub fn serve(self: *Router, listener: posix.socket_t) void {
        while (!self.stop.load(.acquire)) {
            var fds = [_]posix.pollfd{.{ .fd = listener, .events = posix.POLL.IN, .revents = 0 }};
            const ready = posix.poll(&fds, 200) catch break;
            if (ready == 0) continue;
            var storage align(4) = std.mem.zeroes([64]u8);
            var len: posix.socklen_t = 64;
            const rc = posix.system.accept(listener, @ptrCast(&storage), &len);
            if (posix.errno(rc) != .SUCCESS) continue;
            const fd: posix.socket_t = @intCast(rc);
            _ = self.open_connections.fetchAdd(1, .acq_rel);
            const thread = std.Thread.spawn(.{}, handleConn, .{ self, fd }) catch {
                _ = self.open_connections.fetchSub(1, .acq_rel);
                http.close(fd);
                continue;
            };
            thread.detach();
        }
    }

    /// The router's own health, for the script that waits for the stack and for whatever watches the address.
    pub fn healthJson(self: *Router, buf: []u8) []const u8 {
        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);
        var w: std.Io.Writer = .fixed(buf);
        w.print("{{\"copies\":[", .{}) catch {};
        for (self.copies, 0..) |*copy, at| {
            if (at > 0) w.print(",", .{}) catch {};
            w.print("{{\"name\":\"{s}\",\"state\":\"{s}\",\"healthy\":{},\"in_flight\":{d}}}", .{
                copy.name, @tagName(copy.breaker), copy.healthy, copy.in_flight,
            }) catch {};
        }
        w.print("],\"ok\":{}}}", .{self.allUpLocked()}) catch {};
        return w.buffered();
    }

    /// Every model has a copy up: a copy counts when its breaker is closed, its probe answers and its
    /// models are read. Copies that list no model are one model; a copy not yet read is its own.
    fn allUpLocked(self: *Router) bool {
        for (self.copies, 0..) |*copy, at| {
            if (!copy.discovered) {
                if (!self.upLocked(copy)) return false;
                continue;
            }
            // judged once, at the first copy of its model
            const lead = self.leadOfLocked(at);
            if (lead != at) continue;
            var up = false;
            for (self.copies[at..]) |*other| {
                if (other.discovered and sameModel(copy, other) and self.upLocked(other)) up = true;
            }
            if (!up) return false;
        }
        return true;
    }

    fn upLocked(self: *Router, copy: *const Copy) bool {
        _ = self;
        return copy.discovered and copy.breaker == .closed and copy.healthy;
    }

    fn sameModel(a: *const Copy, b: *const Copy) bool {
        if (a.ids.len == 0 or b.ids.len == 0) return a.ids.len == b.ids.len;
        return std.mem.eql(u8, a.ids[0], b.ids[0]);
    }

    /// The first copy of this copy's model.
    fn leadOfLocked(self: *Router, at: usize) usize {
        for (self.copies[0..at], 0..) |*other, index| {
            if (other.discovered and sameModel(&self.copies[at], other)) return index;
        }
        return at;
    }

    fn anyModelsLocked(self: *Router) bool {
        for (self.copies) |*copy| if (copy.ids.len > 0) return true;
        return false;
    }

    /// The merged `/v1/models` body: each id once, in the copies' order.
    fn modelsJsonLocked(self: *Router, out: *std.ArrayList(u8)) !void {
        try out.appendSlice(self.gpa, "{\"object\":\"list\",\"data\":[");
        var first = true;
        for (self.copies, 0..) |*copy, at| {
            for (copy.ids, copy.cards) |id, card| {
                if (self.seenBeforeLocked(at, id)) continue;
                if (!first) try out.append(self.gpa, ',');
                first = false;
                try out.appendSlice(self.gpa, card);
            }
        }
        try out.appendSlice(self.gpa, "]}");
    }

    fn seenBeforeLocked(self: *Router, at: usize, id: []const u8) bool {
        for (self.copies[0..at]) |*other| if (other.lists(id)) return true;
        return false;
    }

    /// The copy-set for a request that names no model: the default copy's model, audio copies apart.
    fn defaultMaskLocked(self: *Router) Mask {
        var mask: Mask = .{};
        const d = if (self.config.default_copy < self.copies.len) self.config.default_copy else 0;
        const model = &self.copies[d];
        for (self.copies, 0..) |*copy, at| {
            if (model.ids.len > 0) {
                if (sameModel(model, copy) and copy.ids.len > 0) mask.set(at);
            } else if (!copy.audio) mask.set(at);
        }
        if (!mask.any()) return Mask.all(self.copies.len);
        return mask;
    }

    // ---- one client connection ---------------------------------------------------------------

    fn handleConn(self: *Router, fd: posix.socket_t) void {
        defer {
            http.close(fd);
            _ = self.open_connections.fetchSub(1, .acq_rel);
        }
        var arena_state: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const head_buf = self.gpa.alloc(u8, self.config.head_limit) catch return;
        defer self.gpa.free(head_buf);

        const head = http.readHead(fd, head_buf) catch return;
        if (std.mem.eql(u8, head.method, "GET")) {
            if (isHealth(head.target)) {
                self.replyHealth(fd) catch {};
                return;
            }
            if (modelsTarget(head.target)) |id| {
                self.replyModels(arena, fd, id) catch {};
                return;
            }
        }
        if (head.header("transfer-encoding") != null) {
            self.replySimple(fd, 400, "a chunked request body is not taken") catch {};
            return;
        }
        if (head.header("expect")) |expect| {
            if (containsIgnoreCase(expect, "100-continue")) {
                http.writeAll(fd, "HTTP/1.1 100 Continue\r\n\r\n") catch return;
            }
        }
        const length = http.contentLength(head) catch {
            self.replySimple(fd, 400, "Content-Length is not a number") catch {};
            return;
        };
        if (length > self.config.max_body) {
            self.replySimple(fd, 413, "the request body is past the router's limit") catch {};
            return;
        }
        const body = arena.alloc(u8, length) catch {
            self.replySimple(fd, 500, "out of memory") catch {};
            return;
        };
        http.readBody(fd, head.extra, body) catch return; // the client left mid-body

        const decision = self.decide(arena, head, body);
        switch (decision.refusal) {
            .none => {},
            .unknown_model => {
                self.replyNotFound(arena, fd, decision.model orelse "") catch {};
                return;
            },
            .models_unknown => {
                self.replySimple(fd, 503, "a backend's models are not known yet") catch {};
                return;
            },
            .no_audio => {
                self.replySimple(fd, 404, "no audio backend is configured") catch {};
                return;
            },
        }
        const chosen = decision.copy orelse {
            self.replySimple(fd, 502, "no copy is up") catch {};
            return;
        };

        var outcome = self.forward(arena, fd, head, body, chosen, decision);
        if (outcome == .connect_failed) {
            if (self.reRoute(chosen, decision.key, decision.allow)) |next| {
                outcome = self.forward(arena, fd, head, body, next, decision);
            }
        }
        switch (outcome) {
            .replied, .client_gone => {},
            .connect_failed, .write_failed, .read_failed, .no_copy => self.replySimple(fd, 502, "no copy took the request") catch {},
        }
    }

    fn isHealth(target: []const u8) bool {
        const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
        return std.mem.eql(u8, path, "/health") or std.mem.eql(u8, path, "/healthz");
    }

    fn replyHealth(self: *Router, fd: posix.socket_t) !void {
        var body_buf: [4096]u8 = undefined;
        const body = self.healthJson(&body_buf);
        // 200 only when every model has a copy up (the body's `ok`, the last field)
        const ok = std.mem.endsWith(u8, body, "\"ok\":true}");
        var head_buf: [256]u8 = undefined;
        const head = try std.fmt.bufPrint(&head_buf, "HTTP/1.1 {s}\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n", .{ if (ok) "200 OK" else "503 Service Unavailable", body.len });
        try http.writeAll(fd, head);
        try http.writeAll(fd, body);
    }

    /// The path of `/v1/models` is "" and of `/v1/models/ID` the id; null for any other target.
    fn modelsTarget(target: []const u8) ?[]const u8 {
        const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
        const prefix = "/v1/models";
        if (!std.mem.startsWith(u8, path, prefix)) return null;
        const rest = path[prefix.len..];
        if (rest.len == 0) return "";
        if (rest[0] == '/') return rest[1..];
        return null;
    }

    /// The merged list, or one model's card.
    fn replyModels(self: *Router, arena: Allocator, fd: posix.socket_t, id: []const u8) !void {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(self.gpa);
        var found = true;
        {
            self.mutex.lockUncancelable(self.config.io);
            defer self.mutex.unlock(self.config.io);
            if (id.len == 0) {
                try self.modelsJsonLocked(&out);
            } else {
                found = false;
                for (self.copies) |*copy| {
                    for (copy.ids, copy.cards) |have, card| {
                        if (!found and std.mem.eql(u8, have, id)) {
                            try out.appendSlice(self.gpa, card);
                            found = true;
                        }
                    }
                }
            }
        }
        if (!found) return self.replyNotFound(arena, fd, id);
        var head_buf: [256]u8 = undefined;
        const head = try std.fmt.bufPrint(&head_buf, "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n", .{out.items.len});
        try http.writeAll(fd, head);
        try http.writeAll(fd, out.items);
    }

    /// vLLM's 404 for a model it does not serve (`check_model`): NotFoundError, param "model", code 404.
    fn replyNotFound(self: *Router, arena: Allocator, fd: posix.socket_t, model: []const u8) !void {
        const message = try std.fmt.allocPrint(arena, "The model `{s}` does not exist.", .{model});
        const body = try std.json.Stringify.valueAlloc(arena, .{
            .@"error" = .{ .message = message, .type = "NotFoundError", .param = "model", .code = 404 },
        }, .{});
        const head = try std.fmt.allocPrint(arena, "HTTP/1.1 404 Not Found\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n", .{body.len});
        try http.writeAll(fd, head);
        try http.writeAll(fd, body);
        self.log("replied 404 itself: unknown model", .{});
    }

    fn replySimple(self: *Router, fd: posix.socket_t, code: u16, message: []const u8) !void {
        var buf: [1024]u8 = undefined;
        const body = try std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\"}}}}", .{message});
        const head = try std.fmt.bufPrint(buf[body.len..], "HTTP/1.1 {d} {s}\r\ncontent-type: application/json\r\ncontent-length: {d}\r\nconnection: close\r\n\r\n", .{ code, reasonOf(code), body.len });
        try http.writeAll(fd, head);
        try http.writeAll(fd, body);
        self.log("replied {d} itself: {s}", .{ code, message });
    }

    // ---- the choice of copy -------------------------------------------------------------------

    /// The tables read and, on a miss, written; the body is parsed before the mutex is taken.
    fn decide(self: *Router, arena: Allocator, head: http.Head, body: []const u8) Decision {
        const api = apiOf(head.target);
        const is_post = std.mem.eql(u8, head.method, "POST");

        // The header's key stands for the opening's.
        var from_header = false;
        var key: ?u64 = null;
        if (head.header(routing_header)) |given| {
            key = std.hash.Wyhash.hash(0, given);
            from_header = true;
        }
        const is_audio = std.mem.startsWith(u8, head.target, "/v1/audio/");
        var parsed: ?std.json.Value = null;
        if (is_post and !is_audio and (api != .none or tokenizeTarget(head.target)) and body.len > 0) {
            parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch null;
        }
        const model: ?[]const u8 = if (parsed) |value| blk: {
            const said = stringField(value, "model") orelse break :blk null;
            break :blk if (said.len == 0) null else said;
        } else null;

        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);

        // Which copies: audio by path, a named model by its list, else the default copy's model.
        var allow = Mask.all(self.copies.len);
        var refusal: Refusal = .none;
        if (is_audio) {
            allow = .{};
            for (self.copies, 0..) |*copy, at| if (copy.audio) allow.set(at);
            if (!allow.any()) refusal = .no_audio;
        } else if (self.anyModelsLocked()) {
            if (model) |m| {
                allow = .{};
                for (self.copies, 0..) |*copy, at| if (copy.lists(m)) allow.set(at);
                if (!allow.any()) {
                    refusal = .unknown_model;
                    for (self.copies) |*copy| if (!copy.discovered) {
                        refusal = .models_unknown;
                    };
                }
            } else {
                allow = self.defaultMaskLocked();
            }
        } else {
            // no model is listed anywhere: all copies are one model, the audio ones aside
            var rest: Mask = .{};
            for (self.copies, 0..) |*copy, at| if (!copy.audio) rest.set(at);
            if (rest.any()) allow = rest;
        }
        if (refusal != .none) return .{ .copy = null, .key = null, .how = .miss, .arm_capture = false, .allow = allow, .refusal = refusal, .model = model };

        // A Responses conversation continues by its id, not by its opening.
        if (parsed) |value| if (api == .responses) {
            if (stringField(value, "previous_response_id")) |id| {
                if (self.idCopyLocked(id)) |copy| if (allow.has(copy) and self.copies[copy].breaker == .closed)
                    return .{ .copy = copy, .key = null, .how = .id, .arm_capture = true, .allow = allow, .model = model };
            }
        };

        // The opening's key, when the header gave none.
        if (key == null and is_post and api != .none) {
            if (parsed) |value| key = openingKey(arena, api, value);
            // one prompt to two models is two conversations
            if (key) |k| if (model) |m| {
                key = k ^ std.hash.Wyhash.hash(0, m);
            };
        }
        if (key) |k| {
            if (self.keyCopyLocked(k)) |copy| if (allow.has(copy) and self.copies[copy].breaker == .closed)
                return .{ .copy = copy, .key = key, .how = if (from_header) .header else .key, .arm_capture = api == .responses and is_post, .allow = allow, .model = model };
        }
        const miss = self.leastBusyLocked(allow) orelse return .{ .copy = null, .key = key, .how = .miss, .arm_capture = false, .allow = allow, .model = model };
        if (key) |k| self.putKeyLocked(k, miss);
        // A DELETE or a GET of a response goes where the response lives, if the table still says.
        if (std.mem.eql(u8, head.method, "DELETE") or std.mem.eql(u8, head.method, "GET")) {
            if (responseIdOfTarget(head.target)) |id| {
                if (self.idCopyLocked(id)) |copy| if (allow.has(copy))
                    return .{ .copy = copy, .key = key, .how = .id, .arm_capture = false, .allow = allow, .model = model };
            }
        }
        return .{ .copy = miss, .key = key, .how = .miss, .arm_capture = api == .responses and is_post, .allow = allow, .model = model };
    }

    /// Where a request goes when nothing is known of it: the least in-flight copy that is up, ties to the
    /// copy with the fewer keys tracked, then to the lower index.
    fn leastBusyLocked(self: *Router, allow: Mask) ?usize {
        var best: ?usize = null;
        var best_load: usize = std.math.maxInt(usize);
        var best_keys: usize = std.math.maxInt(usize);
        for (self.copies, 0..) |*copy, at| {
            if (copy.breaker != .closed or !allow.has(at)) continue;
            const keys = self.keysLocked(at);
            if (copy.in_flight < best_load or (copy.in_flight == best_load and keys < best_keys)) {
                best = at;
                best_load = copy.in_flight;
                best_keys = keys;
            }
        }
        return best;
    }

    /// The copy a failed connect's request retries on instead, its key moved with it.
    fn reRoute(self: *Router, excluding: usize, key: ?u64, allow: Mask) ?usize {
        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);
        var best: ?usize = null;
        var best_load: usize = std.math.maxInt(usize);
        var best_keys: usize = std.math.maxInt(usize);
        for (self.copies, 0..) |*copy, at| {
            if (at == excluding or copy.breaker != .closed or !allow.has(at)) continue;
            const keys = self.keysLocked(at);
            if (copy.in_flight < best_load or (copy.in_flight == best_load and keys < best_keys)) {
                best = at;
                best_load = copy.in_flight;
                best_keys = keys;
            }
        }
        if (best) |at| if (key) |k| self.putKeyLocked(k, at);
        return best;
    }

    fn keyCopyLocked(self: *Router, key: u64) ?usize {
        var hit: ?usize = null;
        for (self.keys) |entry| {
            if (entry.valid and entry.key == key) hit = entry.copy;
        }
        if (hit) |at| if (at >= self.copies.len) return null;
        return hit;
    }

    fn putKeyLocked(self: *Router, key: u64, copy: usize) void {
        // A key the table already holds is rewritten where it lies: a re-routed session's newer copy
        // must win, and a second entry would let the ring's older record outvote it once the write
        // index has wrapped past the one and not the other.
        for (self.keys) |*entry| {
            if (entry.valid and entry.key == key and entry.copy != copy) {
                entry.copy = @intCast(copy);
                return;
            }
            if (entry.valid and entry.key == key) return; // already here, already this copy
        }
        self.keys[self.key_at] = .{ .key = key, .copy = @intCast(copy), .valid = true };
        self.key_at = (self.key_at + 1) % self.keys.len;
    }

    fn idCopyLocked(self: *Router, id: []const u8) ?usize {
        if (id.len > self.ids[0].id.len) return null;
        var hit: ?usize = null;
        for (self.ids) |entry| {
            if (entry.valid and entry.len == id.len and std.mem.eql(u8, entry.id[0..entry.len], id)) hit = entry.copy;
        }
        if (hit) |at| if (at >= self.copies.len) return null;
        return hit;
    }

    fn putIdLocked(self: *Router, id: []const u8, copy: usize) void {
        if (id.len > self.ids[0].id.len) return;
        // A response id the table holds is rewritten where it lies, for the keys' reason: one entry
        // an id, and the ring's older records cannot outvote a newer home.
        for (self.ids) |*entry| {
            if (entry.valid and entry.len == id.len and std.mem.eql(u8, entry.id[0..entry.len], id)) {
                entry.copy = @intCast(copy);
                return;
            }
        }
        self.ids[self.id_at] = .{ .copy = @intCast(copy), .valid = true, .len = @intCast(id.len) };
        @memcpy(self.ids[self.id_at].id[0..id.len], id);
        self.id_at = (self.id_at + 1) % self.ids.len;
    }

    fn keysLocked(self: *Router, at: usize) usize {
        var count: usize = 0;
        for (self.keys) |entry| {
            if (entry.valid and entry.copy == at) count += 1;
        }
        return count;
    }

    // ---- a slot, the forward, the pump ---------------------------------------------------------

    /// Waits for a slot at `first`, in arrival order; re-chooses when that copy is left while it waits.
    fn takeSlot(self: *Router, first: usize, allow: Mask) ?usize {
        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);
        var want = first;
        var waiter = Waiter{};
        var queued = false;
        while (true) {
            if (self.stop.load(.acquire)) return null;
            if (self.copies[want].breaker == .open) {
                if (queued) {
                    self.removeFromQueue(want, &waiter);
                    queued = false;
                }
                want = self.leastBusyLocked(allow) orelse return null;
                continue;
            }
            const copy = &self.copies[want];
            const head_of_queue = if (queued)
                (copy.queue.items.len > 0 and copy.queue.items[0] == &waiter)
            else
                copy.queue.items.len == 0;
            if (copy.in_flight < copy.parallel and head_of_queue) {
                if (queued) self.removeFromQueue(want, &waiter);
                copy.in_flight += 1;
                return want;
            }
            if (!queued) {
                copy.queue.append(self.gpa, &waiter) catch return null;
                queued = true;
            }
            self.changed.waitUncancelable(self.config.io, &self.mutex);
        }
    }

    fn removeFromQueue(self: *Router, at: usize, waiter: *const Waiter) void {
        for (self.copies[at].queue.items, 0..) |item, index| {
            if (item == waiter) {
                _ = self.copies[at].queue.orderedRemove(index);
                return;
            }
        }
    }

    fn releaseSlot(self: *Router, at: usize) void {
        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);
        self.copies[at].in_flight -= 1;
        self.changed.broadcast(self.config.io);
    }

    const Outcome = enum { replied, connect_failed, write_failed, read_failed, client_gone, no_copy };

    fn forward(self: *Router, arena: Allocator, client: posix.socket_t, head: http.Head, body: []const u8, first: usize, decision: Decision) Outcome {
        const at = self.takeSlot(first, decision.allow) orelse return .no_copy;
        const started = self.nowNs();
        const result = relay.forward(arena, client, self.copies[at].address, self.config.connect_timeout_ms, head, body, decision.arm_capture);
        defer if (result != .reply) self.releaseSlot(at);
        const reply = switch (result) {
            .connect_failed => {
                self.recordFailure(at, "the connect failed");
                return .connect_failed;
            },
            .request_head_failed => return .write_failed,
            .write_failed => {
                self.recordFailure(at, "the request did not go out");
                return .write_failed;
            },
            .read_failed => |why| {
                self.recordFailure(at, why);
                return .read_failed;
            },
            .client_head_failed => return .client_gone,
            .reply => |reply| reply,
        };
        if (reply.end == .complete and reply.capture.id_len > 0) {
            self.mutex.lockUncancelable(self.config.io);
            defer self.mutex.unlock(self.config.io);
            self.putIdLocked(reply.capture.id[0..reply.capture.id_len], at);
        }
        self.finish(at, reply.status < 500 and reply.end != .copy_mid_reply);
        self.logReply(at, decision, reply.status, reply.bytes, started, reply.note());
        return switch (reply.end) {
            .client_at_head, .client_mid_reply => .client_gone,
            .complete, .copy_mid_reply => .replied,
        };
    }

    /// The slot given back and the copy's state told the reply's verdict; the queue's head wakes.
    fn finish(self: *Router, at: usize, ok: bool) void {
        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);
        self.copies[at].in_flight -= 1;
        if (ok) {
            self.copies[at].failures = 0;
        } else {
            self.countFailureLocked(at, "a reply of 5xx");
        }
        self.changed.broadcast(self.config.io);
    }

    fn recordFailure(self: *Router, at: usize, why: []const u8) void {
        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);
        self.log("copy {s} failed: {s}", .{ self.copies[at].name, why });
        self.countFailureLocked(at, why);
        self.changed.broadcast(self.config.io);
    }

    fn countFailureLocked(self: *Router, at: usize, why: []const u8) void {
        const copy = &self.copies[at];
        copy.failures += 1;
        if (copy.failures >= self.config.failure_limit and copy.breaker == .closed) {
            copy.breaker = .open;
            copy.opened_at = self.nowNs();
            self.log("copy {s} is left for {d} ms after {d} failures ({s})", .{ copy.name, self.config.open_ms, copy.failures, why });
        }
    }

    // ---- the health poll -----------------------------------------------------------------------

    fn pollLoop(self: *Router, at: usize) void {
        self.discover(at); // the first read, at once: a copy already up is known before a request comes
        while (!self.stop.load(.acquire)) {
            sleep(self.config.io, self.config.poll_ms);
            if (self.stop.load(.acquire)) return;
            self.mutex.lockUncancelable(self.config.io);
            const was_open = self.copies[at].breaker == .open;
            const due = !was_open or self.nowNs() - self.copies[at].opened_at >= openNs(self.config.open_ms);
            self.mutex.unlock(self.config.io);
            if (was_open and !due) continue;
            const ok = self.probe(at);
            // a copy not yet read, or back from being left, lists its models (it may be another's)
            if (ok) {
                self.mutex.lockUncancelable(self.config.io);
                const unread = !self.copies[at].discovered;
                self.mutex.unlock(self.config.io);
                if (unread or was_open) self.discover(at);
            }
            self.mutex.lockUncancelable(self.config.io);
            defer self.mutex.unlock(self.config.io);
            const copy = &self.copies[at];
            if (was_open) {
                if (ok) {
                    copy.breaker = .closed;
                    copy.failures = 0;
                    copy.healthy = true;
                    self.log("copy {s} is taken back (its health probe answered)", .{copy.name});
                    self.changed.broadcast(self.config.io);
                } else {
                    copy.opened_at = self.nowNs(); // left for another open_ms
                }
            } else if (ok) {
                copy.failures = 0; // a probe answered: its failures are over
                if (!copy.healthy) {
                    copy.healthy = true;
                    self.log("copy {s} answers /health again", .{copy.name});
                }
            } else {
                // a probe's failure is a failure like a request's: three in a row leave the copy
                copy.healthy = false;
                self.countFailureLocked(at, "no answer from /health");
                self.changed.broadcast(self.config.io);
            }
        }
    }

    /// Reads the copy's `/v1/models` and keeps its ids and cards; a copy that does not answer stays unread.
    fn discover(self: *Router, at: usize) void {
        var arena_state: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const ids_and_cards = self.fetchModels(arena, at) orelse return;
        const ids = self.gpa.alloc([]u8, ids_and_cards.ids.len) catch return;
        const cards = self.gpa.alloc([]u8, ids_and_cards.ids.len) catch {
            self.gpa.free(ids);
            return;
        };
        var made: usize = 0;
        for (ids_and_cards.ids, ids_and_cards.cards, 0..) |id, card, index| {
            ids[index] = self.gpa.dupe(u8, id) catch break;
            cards[index] = self.gpa.dupe(u8, card) catch {
                self.gpa.free(ids[index]);
                break;
            };
            made += 1;
        }
        self.mutex.lockUncancelable(self.config.io);
        defer self.mutex.unlock(self.config.io);
        const copy = &self.copies[at];
        copy.freeModels(self.gpa);
        if (made == ids.len and made > 0) {
            copy.ids = ids;
            copy.cards = cards;
        } else {
            for (ids[0..made]) |id| self.gpa.free(id);
            for (cards[0..made]) |card| self.gpa.free(card);
            self.gpa.free(ids);
            self.gpa.free(cards);
            if (made != ids_and_cards.ids.len) return; // out of memory: read again next poll
        }
        copy.discovered = true;
        self.log("copy {s} serves {d} model name(s){s}{s}", .{ copy.name, copy.ids.len, if (copy.ids.len > 0) ", the first " else "", if (copy.ids.len > 0) copy.ids[0] else "" });
        self.changed.broadcast(self.config.io);
    }

    const Listed = struct { ids: []const []const u8, cards: []const []const u8 };

    fn fetchModels(self: *Router, arena: Allocator, at: usize) ?Listed {
        const fd = http.connect(self.copies[at].address, self.config.connect_timeout_ms) catch return null;
        defer http.close(fd);
        // the copy's own address as the host: a `serve` bound to the loopback refuses any other name (403)
        var said: [64]u8 = undefined;
        var asked_buf: [160]u8 = undefined;
        const asked = std.fmt.bufPrint(&asked_buf, "GET /v1/models HTTP/1.1\r\nhost: {s}\r\nconnection: close\r\n\r\n", .{http.addressText(&said, self.copies[at].address)}) catch return null;
        http.writeAll(fd, asked) catch return null;
        const ms: i32 = @max(self.config.health_timeout_ms, 2000);
        const timeout: posix.timeval = .{ .sec = @intCast(@divTrunc(ms, 1000)), .usec = @intCast(@as(u32, @intCast(@rem(ms, 1000))) * 1000) };
        posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch return null;
        const buf = arena.alloc(u8, 32 * 1024) catch return null;
        const reply = http.readHead(fd, buf) catch return null;
        const code = reply.status() catch return null;
        if (code < 200 or code >= 300) return null;
        const limit: usize = 1 << 20;
        var body: std.ArrayList(u8) = .empty;
        body.appendSlice(arena, reply.extra) catch return null;
        const want = http.contentLength(reply) catch return null;
        if (want > limit) return null;
        // read to the length when it is given, else to the close (the request said `connection: close`)
        var chunk: [4096]u8 = undefined;
        while (want == 0 or body.items.len < want) {
            const got = http.readSome(fd, &chunk) catch break;
            if (got == 0) break;
            if (body.items.len + got > limit) return null;
            body.appendSlice(arena, chunk[0..got]) catch return null;
        }
        const root = std.json.parseFromSliceLeaky(std.json.Value, arena, body.items, .{}) catch return null;
        if (root != .object) return null;
        const data = root.object.get("data") orelse return .{ .ids = &.{}, .cards = &.{} };
        if (data != .array) return .{ .ids = &.{}, .cards = &.{} };
        var ids: std.ArrayList([]const u8) = .empty;
        var cards: std.ArrayList([]const u8) = .empty;
        for (data.array.items) |entry| {
            const id = stringField(entry, "id") orelse continue;
            const card = std.json.Stringify.valueAlloc(arena, entry, .{}) catch return null;
            ids.append(arena, id) catch return null;
            cards.append(arena, card) catch return null;
        }
        return .{ .ids = ids.items, .cards = cards.items };
    }

    fn probe(self: *Router, at: usize) bool {
        const fd = http.connect(self.copies[at].address, self.config.connect_timeout_ms) catch return false;
        defer http.close(fd);
        var said: [64]u8 = undefined;
        var asked_buf: [160]u8 = undefined;
        const asked = std.fmt.bufPrint(&asked_buf, "GET /health HTTP/1.1\r\nhost: {s}\r\nconnection: close\r\n\r\n", .{http.addressText(&said, self.copies[at].address)}) catch return false;
        http.writeAll(fd, asked) catch return false;
        // the timeout split across the timeval's fields: `usec` holds under a million of them
        const ms = self.config.health_timeout_ms;
        const timeout: posix.timeval = .{ .sec = @intCast(@divTrunc(ms, 1000)), .usec = @intCast(@as(u32, @intCast(@rem(ms, 1000))) * 1000) };
        posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch return false;
        var buf: [2048]u8 = undefined;
        const reply = http.readHead(fd, &buf) catch return false;
        const code = reply.status() catch return false;
        return code >= 200 and code < 300;
    }

    // ---- the rest ------------------------------------------------------------------------------

    fn nowNs(self: *Router) i96 {
        return std.Io.Clock.awake.now(self.config.io).toNanoseconds();
    }

    /// The hook the recipe is held by: a reply, its copy and the key that routed it.
    fn logReply(self: *Router, at: usize, decision: Decision, status: u16, bytes: usize, started: i96, note: ?[]const u8) void {
        const ms = @as(f64, @floatFromInt(self.nowNs() - started)) / std.time.ns_per_ms;
        var key_buf: [32]u8 = undefined;
        const key_said = if (decision.key) |k| std.fmt.bufPrint(&key_buf, "{x:0>16}", .{k}) catch "?" else "none";
        if (note) |said| {
            self.log("reply copy={s} how={s} key={s} status={d} bytes={d} ms={d:.0} ({s})", .{ self.copies[at].name, @tagName(decision.how), key_said, status, bytes, ms, said });
        } else {
            self.log("reply copy={s} how={s} key={s} status={d} bytes={d} ms={d:.0}", .{ self.copies[at].name, @tagName(decision.how), key_said, status, bytes, ms });
        }
    }

    fn log(self: *Router, comptime fmt: []const u8, args: anytype) void {
        var buf: [2048]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        w.print("[router] " ++ fmt ++ "\n", args) catch {};
        self.log_mutex.lockUncancelable(self.config.io);
        defer self.log_mutex.unlock(self.config.io);
        writeOut(w.buffered());
    }
};

fn openNs(ms: u64) i96 {
    return @as(i96, ms) * std.time.ns_per_ms;
}

fn writeOut(bytes: []const u8) void {
    // stderr, not stdout: the vendored host logs there, and a test's stdout is the build runner's
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = posix.system.write(2, bytes[off..].ptr, bytes.len - off);
        if (posix.errno(rc) != .SUCCESS or rc <= 0) return;
        off += @intCast(rc);
    }
}

fn sleep(io: std.Io, ms: u64) void {
    std.Io.sleep(io, .fromMilliseconds(@intCast(ms)), .awake) catch {};
}

fn reasonOf(code: u16) []const u8 {
    return switch (code) {
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        else => "Error",
    };
}

/// vLLM's tokenizer routes take a `model` like the generation ones.
fn tokenizeTarget(target: []const u8) bool {
    const path = if (std.mem.indexOfScalar(u8, target, '?')) |q| target[0..q] else target;
    return std.mem.endsWith(u8, path, "/tokenize") or std.mem.endsWith(u8, path, "/detokenize");
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var at: usize = 0;
    while (at + needle.len <= haystack.len) : (at += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[at .. at + needle.len], needle)) return true;
    }
    return false;
}

// ---- the routing key: a request's opening, hashed --------------------------------------------- //

// In `router_key.zig`.
pub const Api = router_key.Api;
pub const apiOf = router_key.apiOf;
pub const openingKey = router_key.openingKey;
const stringField = router_key.stringField;
const responseIdOfTarget = router_key.responseIdOfTarget;

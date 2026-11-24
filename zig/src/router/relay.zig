//! Forward one HTTP request and stream the backend's reply to the client.

const std = @import("std");
const http = @import("http.zig");
const Capture = @import("response_id.zig").Capture;
const Allocator = std.mem.Allocator;
const Socket = std.posix.socket_t;

pub const Result = union(enum) {
    connect_failed,
    request_head_failed,
    write_failed,
    read_failed: []const u8,
    client_head_failed,
    reply: Reply,
};

pub const Reply = struct {
    status: u16,
    bytes: usize = 0,
    capture: Capture = .{},
    end: End = .complete,

    pub const End = enum { complete, client_at_head, client_mid_reply, copy_mid_reply };

    pub fn note(self: Reply) ?[]const u8 {
        return switch (self.end) {
            .complete => null,
            .client_at_head => "the client left at the head",
            .client_mid_reply => "the client left mid-reply",
            .copy_mid_reply => "the copy ended mid-reply",
        };
    }

    fn send(self: *Reply, client: Socket, bytes: []const u8, capture: bool) !void {
        if (capture) self.capture.scan(bytes);
        self.bytes += bytes.len;
        try http.writeAll(client, bytes);
    }
};

pub fn forward(arena: Allocator, client: Socket, address: std.Io.net.IpAddress, timeout_ms: i32, head: http.Head, body: []const u8, capture_id: bool) Result {
    const fd = http.connect(address, timeout_ms) catch return .connect_failed;
    defer http.close(fd);

    var addr_buf: [64]u8 = undefined;
    const request_head = upstreamHead(arena, head, http.addressText(&addr_buf, address), body.len) catch return .request_head_failed;
    http.writeAll(fd, request_head) catch return .write_failed;
    http.writeAll(fd, body) catch return .write_failed;

    var buf: [32 * 1024]u8 = undefined;
    const head_from_copy = http.readHead(fd, &buf) catch return .{ .read_failed = "no reply head came" };
    const status = head_from_copy.status() catch return .{ .read_failed = "the reply head is not HTTP/1.1" };
    const to_client = clientHead(arena, head_from_copy) catch return .client_head_failed;
    var reply: Reply = .{ .status = status };
    const capture = capture_id and status < 300;
    http.writeAll(client, to_client) catch {
        reply.end = .client_at_head;
        return .{ .reply = reply };
    };
    // The head reader may also have read the first body bytes. Relay them before reusing buf.
    if (head_from_copy.extra.len > 0) reply.send(client, head_from_copy.extra, capture) catch {
        reply.end = .client_at_head;
        return .{ .reply = reply };
    };
    while (true) {
        const got = http.readSome(fd, &buf) catch {
            reply.end = .copy_mid_reply;
            break;
        };
        if (got == 0) break;
        reply.send(client, buf[0..got], capture) catch {
            reply.end = .client_mid_reply;
            break;
        };
    }
    return .{ .reply = reply };
}

/// The request as the copy is sent it: the client's fields but the hop's, and told to close.
fn upstreamHead(arena: Allocator, head: http.Head, address: []const u8, body_len: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(arena, head.method);
    try out.append(arena, ' ');
    try out.appendSlice(arena, head.target);
    try out.appendSlice(arena, " HTTP/1.1\r\nhost: ");
    try out.appendSlice(arena, address);
    try out.appendSlice(arena, "\r\n");
    var rest = head.fields;
    while (rest.len > 0) {
        const end = std.mem.indexOf(u8, rest, "\r\n") orelse rest.len;
        const line = rest[0..end];
        rest = if (end >= rest.len) rest[end..] else rest[end + 2 ..];
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon];
        if (named(name, "host") or named(name, "connection") or named(name, "keep-alive") or
            named(name, "content-length") or named(name, "expect") or named(name, "transfer-encoding")) continue;
        try out.appendSlice(arena, line);
        try out.appendSlice(arena, "\r\n");
    }
    var num_buf: [32]u8 = undefined;
    try out.appendSlice(arena, "content-length: ");
    try out.appendSlice(arena, std.fmt.bufPrint(&num_buf, "{d}", .{body_len}) catch return error.OutOfMemory);
    try out.appendSlice(arena, "\r\nconnection: close\r\n\r\n");
    return out.items;
}

/// The reply head for the client keeps the backend's status and end-to-end fields.
fn clientHead(arena: Allocator, reply: http.Head) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    const line_end = std.mem.indexOf(u8, reply.raw, "\r\n") orelse reply.raw.len;
    try out.appendSlice(arena, reply.raw[0..line_end]);
    try out.appendSlice(arena, "\r\nconnection: close\r\n");
    var rest = reply.fields;
    while (rest.len > 0) {
        const end = std.mem.indexOf(u8, rest, "\r\n") orelse rest.len;
        const line = rest[0..end];
        rest = if (end >= rest.len) rest[end..] else rest[end + 2 ..];
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        if (named(line[0..colon], "connection") or named(line[0..colon], "keep-alive")) continue;
        try out.appendSlice(arena, line);
        try out.appendSlice(arena, "\r\n");
    }
    try out.appendSlice(arena, "\r\n");
    return out.items;
}

fn named(field: []const u8, name: []const u8) bool {
    return std.ascii.eqlIgnoreCase(field, name);
}

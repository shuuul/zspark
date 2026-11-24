//! The HTTP/1.1 pieces the router needs (spec 088): a request head read and inspected, an upstream
//! connection taken with a connect timeout, and raw byte pumps. Nothing here reads a body; the router
//! hands body bytes on as they came, reply framing included, so a stream is the copy's bytes and nothing
//! else. Addresses are IPv4 literals; the recipes name them so.

const std = @import("std");
const posix = std.posix;

pub const Error = error{
    /// The peer closed before the head was whole or a body's bytes were complete.
    Closed,
    /// The head passed the buffer it was given, or a body its Content-Length's bound.
    TooLong,
    /// The head is not an HTTP/1.x request or reply.
    BadHead,
    /// Content-Length is there and not a number.
    BadLength,
};

pub const Head = struct {
    /// The whole head, status or request line included, inside the caller's buffer.
    raw: []const u8,
    method: []const u8,
    target: []const u8,
    /// The fields alone, one `Name: value` line a `\r\n`.
    fields: []const u8,
    /// Body bytes the head's reads took in beyond the head, before the socket again.
    extra: []const u8,

    /// A field's value, the first one named, without its leading spaces; null when the head has none.
    pub fn header(h: Head, name: []const u8) ?[]const u8 {
        var rest = h.fields;
        while (rest.len > 0) {
            const end = std.mem.indexOf(u8, rest, "\r\n") orelse rest.len;
            const line = rest[0..end];
            rest = if (end >= rest.len) rest[end..] else rest[end + 2 ..];
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            if (!std.ascii.eqlIgnoreCase(line[0..colon], name)) continue;
            var value = line[colon + 1 ..];
            while (value.len > 0 and value[0] == ' ') value = value[1..];
            return value;
        }
        return null;
    }

    /// The reply's or request's status code, the second word of the first line.
    pub fn status(h: Head) Error!u16 {
        const line_end = std.mem.indexOf(u8, h.raw, "\r\n") orelse return error.BadHead;
        var parts = std.mem.tokenizeScalar(u8, h.raw[0..line_end], ' ');
        _ = parts.next() orelse return error.BadHead;
        const code = parts.next() orelse return error.BadHead;
        return std.fmt.parseInt(u16, code, 10) catch error.BadHead;
    }
};

/// Reads a head, to `\r\n\r\n`, into `buf`; the returned slices point into it. A reply's status line
/// (`HTTP/1.1 200 OK`) and a request's request line (`POST /x HTTP/1.1`) both parse; `method` is the
/// first word, `target` the second, whichever of the two is the version.
pub fn readHead(fd: posix.socket_t, buf: []u8) Error!Head {
    var got: usize = 0;
    const at = while (true) {
        const found = std.mem.indexOf(u8, buf[0..got], "\r\n\r\n") orelse {
            if (got == buf.len) return error.TooLong;
            const read = posix.read(fd, buf[got..]) catch return error.Closed;
            if (read == 0) return error.Closed;
            got += read;
            continue;
        };
        break found;
    };
    const raw = buf[0 .. at + 4];
    const line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return error.BadHead;
    var parts = std.mem.tokenizeScalar(u8, raw[0..line_end], ' ');
    const first = parts.next() orelse return error.BadHead;
    const second = parts.next() orelse return error.BadHead;
    if (std.mem.startsWith(u8, first, "HTTP/1.")) {
        // a status line: the version leads and the code follows
        return .{
            .raw = raw,
            .method = first,
            .target = second,
            .fields = raw[line_end + 2 .. at],
            .extra = buf[at + 4 .. got],
        };
    }
    const third = parts.next() orelse return error.BadHead;
    if (!std.mem.startsWith(u8, third, "HTTP/1.")) return error.BadHead;
    return .{
        .raw = raw,
        .method = first,
        .target = second,
        .fields = raw[line_end + 2 .. at],
        .extra = buf[at + 4 .. got],
    };
}

/// The head's Content-Length; 0 when it has none.
pub fn contentLength(h: Head) Error!usize {
    const said = h.header("content-length") orelse return 0;
    return std.fmt.parseInt(usize, said, 10) catch error.BadLength;
}

/// Reads a body of `body.len` bytes into it, from what the head's reads took in and then the socket.
pub fn readBody(fd: posix.socket_t, head_extra: []const u8, body: []u8) Error!void {
    const take = @min(head_extra.len, body.len);
    @memcpy(body[0..take], head_extra[0..take]);
    var at: usize = take;
    while (at < body.len) {
        const read = posix.read(fd, body[at..]) catch return error.Closed;
        if (read == 0) return error.Closed;
        at += read;
    }
}

pub fn writeAll(fd: posix.socket_t, bytes: []const u8) Error!void {
    var at: usize = 0;
    while (at < bytes.len) {
        const wrote = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        if (posix.errno(wrote) != .SUCCESS) return error.Closed;
        if (wrote == 0) return error.Closed;
        at += @intCast(wrote);
    }
}

/// One read's bytes; 0 when the peer closed.
pub fn readSome(fd: posix.socket_t, buf: []u8) Error!usize {
    return posix.read(fd, buf) catch return error.Closed;
}

/// The listening socket: SO_REUSEADDR, backlog 128, as the vendored host's is.
pub fn listen(address: std.Io.net.IpAddress) !posix.socket_t {
    var storage align(4) = std.mem.zeroes([64]u8);
    const len = toPosix(address, &storage);
    const family: u32 = if (address == .ip4) posix.AF.INET else posix.AF.INET6;
    const rc = posix.system.socket(family, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    errdefer _ = posix.system.close(fd);
    const one: c_int = 1;
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
    switch (posix.errno(posix.system.bind(fd, @ptrCast(&storage), len))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        else => return error.BindFailed,
    }
    if (posix.errno(posix.system.listen(fd, 128)) != .SUCCESS) return error.ListenFailed;
    return fd;
}

/// A connected socket to `address`, giving up after `timeout_ms`; blocking afterwards.
pub fn connect(address: std.Io.net.IpAddress, timeout_ms: i32) !posix.socket_t {
    var storage align(4) = std.mem.zeroes([64]u8);
    const len = toPosix(address, &storage);
    const family: u32 = if (address == .ip4) posix.AF.INET else posix.AF.INET6;
    const rc = posix.system.socket(family, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    errdefer _ = posix.system.close(fd);
    // a socket just made has no status flags worth keeping, so O_NONBLOCK is set and cleared alone
    setNonblocking(fd, true);
    defer setNonblocking(fd, false);
    switch (posix.errno(posix.system.connect(fd, @ptrCast(&storage), len))) {
        .SUCCESS => return fd,
        .INPROGRESS => {},
        else => return error.ConnectFailed,
    }
    var fds = [_]posix.pollfd{.{ .fd = fd, .events = posix.POLL.OUT, .revents = 0 }};
    const ready = posix.poll(&fds, timeout_ms) catch return error.ConnectFailed;
    if (ready == 0) return error.ConnectTimedOut;
    var err: c_int = 0;
    var err_len: posix.socklen_t = @sizeOf(c_int);
    if (posix.errno(posix.system.getsockopt(fd, posix.SOL.SOCKET, posix.SO.ERROR, @ptrCast(&err), &err_len)) != .SUCCESS) return error.ConnectFailed;
    if (err != 0) return error.ConnectFailed;
    return fd;
}

fn setNonblocking(fd: posix.socket_t, nonblocking: bool) void {
    const flags: posix.O = .{ .ACCMODE = .RDWR, .NONBLOCK = nonblocking };
    _ = posix.system.fcntl(fd, posix.F.SETFL, @as(u32, @bitCast(flags)));
}

pub fn close(fd: posix.socket_t) void {
    _ = posix.system.shutdown(fd, posix.SHUT.RDWR);
    _ = posix.system.close(fd);
}

/// The address as a `host` header's value would want it written.
pub fn addressText(buf: []u8, address: std.Io.net.IpAddress) []const u8 {
    switch (address) {
        .ip4 => |a| return std.fmt.bufPrint(buf, "{d}.{d}.{d}.{d}:{d}", .{ a.bytes[0], a.bytes[1], a.bytes[2], a.bytes[3], a.port }) catch "",
        .ip6 => |a| return std.fmt.bufPrint(buf, "[{x}]:{d}", .{ a.bytes, a.port }) catch "",
    }
}

/// `std.Io.net`'s address in a sockaddr, as the vendored listener's `Threaded.addressToPosix` does it.
fn toPosix(address: std.Io.net.IpAddress, storage: *[64]u8) posix.socklen_t {
    switch (address) {
        .ip4 => |a| {
            const in: *posix.sockaddr.in = @ptrCast(@alignCast(storage));
            in.* = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, a.port), .addr = @bitCast(a.bytes) };
            return @sizeOf(posix.sockaddr.in);
        },
        .ip6 => |a| {
            const in6: *posix.sockaddr.in6 = @ptrCast(@alignCast(storage));
            in6.* = .{ .family = posix.AF.INET6, .port = std.mem.nativeToBig(u16, a.port), .flowinfo = 0, .addr = a.bytes, .scope_id = 0 };
            return @sizeOf(posix.sockaddr.in6);
        },
    }
}

//! The control channel between two ranks (spec 045 WS-02): one TCP connection over the link's addresses, rank 1
//! listening and rank 0 connecting. Host bytes cross here and never through the GPU: the handshake (what both
//! ranks must hold alike, and NCCL's unique id), a request, a barrier, a failure. A peer that is gone is an
//! error at the next frame (`PeerGone`), so a rank ends instead of waiting inside NCCL. Host only.
//!
//! A frame is its payload's length (u32, little endian), its kind (u8) and the payload. A hello is `magic`,
//! the protocol, the sender's rank and the world, then the build, the checkpoint's headers and the options
//! each as a SHA-256 and a label a person reads in a refusal, then the 128 bytes of NCCL's id (zero from a
//! rank that takes it). A request's bytes are its sender's (WS-04 says what they hold); a barrier is a
//! counter, sent and expected back equal; a failure is text.

const std = @import("std");
const Io = std.Io;
const net = std.Io.net;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const magic = "zsrk";
pub const protocol: u32 = 1;
pub const world = 2;
/// NCCL's `ncclUniqueId`, as bytes.
pub const Id = [128]u8;
/// The most a frame carries: a prompt's ids and a grammar's masks are far below it, a wrong length above.
pub const max_payload = 64 << 20;
const header_bytes = 5;
const max_label = 160;

pub const Error = error{
    /// The connection closed, between frames or under a write.
    PeerGone,
    /// It closed inside a frame.
    TruncatedFrame,
    FrameTooLarge,
    UnknownFrame,
    /// A frame of a kind the step does not take.
    UnexpectedFrame,
    /// The peer sent a failure: `Channel.said` has its text.
    PeerFailed,
    /// A barrier's counter is not ours.
    OutOfStep,
    NotARank,
    ProtocolsDiffer,
    RanksCollide,
    BuildsDiffer,
    CheckpointsDiffer,
    OptionsDiffer,
    OutOfMemory,
};

pub const Kind = enum(u8) { hello = 1, request = 2, barrier = 3, failure = 4 };

pub const Digest = [Sha256.digest_length]u8;

/// One digest of `parts` in order, each with its length, so two lists that concatenate alike differ.
pub fn digest(parts: []const []const u8) Digest {
    var hash: Sha256 = .init(.{});
    for (parts) |part| {
        var length: [8]u8 = undefined;
        std.mem.writeInt(u64, &length, part.len, .little);
        hash.update(&length);
        hash.update(part);
    }
    var out: Digest = undefined;
    hash.final(&out);
    return out;
}

/// What is compared, and what a refusal shows of it.
pub const Named = struct {
    digest: Digest,
    label: []const u8 = "",

    /// `text` is both: its digest and, cut to a line, its label.
    pub fn of(text: []const u8) Named {
        return .{ .digest = digest(&.{text}), .label = text[0..@min(text.len, max_label)] };
    }
};

/// What both ranks must hold alike before anything runs.
pub const Identity = struct {
    /// `zspark version`'s commit, and whether the tree differed from it.
    build: Named,
    /// The checkpoint's headers: every tensor's name, type and shape.
    checkpoint: Named,
    /// The options a rank was started with that change what it computes.
    options: Named,
};

const facets = [_]struct { field: []const u8, what: []const u8, err: Error }{
    .{ .field = "build", .what = "the builds", .err = error.BuildsDiffer },
    .{ .field = "checkpoint", .what = "the checkpoints' headers", .err = error.CheckpointsDiffer },
    .{ .field = "options", .what = "the options", .err = error.OptionsDiffer },
};

pub const Message = union(enum) {
    /// Valid until the channel's next frame.
    request: []const u8,
    barrier: u64,
};

pub const Channel = struct {
    reader: *Io.Reader,
    /// The connection's socket, where the channel is over one (`Link.channel`): a frame is waited for awake
    /// (`spin_ns`) before the read blocks.
    fd: ?std.posix.fd_t = null,
    writer: *Io.Writer,
    gpa: std.mem.Allocator,
    /// The last frame's payload.
    payload: std.ArrayList(u8) = .empty,
    /// The last refusal, ours or the peer's.
    said_buffer: [512]u8 = undefined,
    said_len: usize = 0,

    pub fn init(gpa: std.mem.Allocator, reader: *Io.Reader, writer: *Io.Writer) Channel {
        return .{ .reader = reader, .writer = writer, .gpa = gpa };
    }

    pub fn deinit(self: *Channel) void {
        self.payload.deinit(self.gpa);
        self.* = undefined;
    }

    /// The text of the last refusal: what this rank refused, or the failure the peer sent.
    pub fn said(self: *const Channel) []const u8 {
        return self.said_buffer[0..self.said_len];
    }

    fn say(self: *Channel, comptime format: []const u8, args: anytype) void {
        const text = std.fmt.bufPrint(&self.said_buffer, format, args) catch self.said_buffer[0..];
        self.said_len = text.len;
    }

    /// One frame, sent whole before this returns.
    pub fn send(self: *Channel, kind: Kind, bytes: []const u8) Error!void {
        if (bytes.len > max_payload) return error.FrameTooLarge;
        var header: [header_bytes]u8 = undefined;
        std.mem.writeInt(u32, header[0..4], @intCast(bytes.len), .little);
        header[4] = @backingInt(kind);
        self.writer.writeAll(&header) catch return error.PeerGone;
        self.writer.writeAll(bytes) catch return error.PeerGone;
        self.writer.flush() catch return error.PeerGone;
    }

    const Frame = struct { kind: Kind, bytes: []const u8 };

    /// How long a rank polls its socket for the next frame before its read blocks (spec 114 WS-02): a core that
    /// sleeps in the read wakes from the CPU's deep idle state when the frame comes, some 0.5 ms a frame on the
    /// GB10's (`ping` between the Sparks 0.91 ms on average, 0.36 with both machines' cores kept busy), and a
    /// rank that follows waits for a frame at every operation of a round. Past a round's length (a reply's
    /// end, a wait for a request) the read blocks and the core rests.
    pub var spin_ns: u64 = 200 * std.time.ns_per_ms;

    fn awaitFrame(self: *Channel) void {
        const fd = self.fd orelse return;
        if (self.reader.bufferedLen() > 0) return;
        const start = now();
        var fds = [1]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        while (now() - start < spin_ns) {
            fds[0].revents = 0;
            const ready = std.posix.poll(&fds, 0) catch return;
            if (ready > 0) return;
            std.atomic.spinLoopHint();
        }
    }

    fn now() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    /// The next frame, whatever its kind; its bytes are valid until the one after.
    pub fn receive(self: *Channel) Error!Frame {
        self.awaitFrame();
        var header: [header_bytes]u8 = undefined;
        const got = self.reader.readSliceShort(&header) catch return error.PeerGone;
        if (got == 0) return error.PeerGone;
        if (got < header.len) return error.TruncatedFrame;
        const length = std.mem.readInt(u32, header[0..4], .little);
        if (length > max_payload) return error.FrameTooLarge;
        const kind = std.enums.fromInt(Kind, header[4]) orelse return error.UnknownFrame;
        try self.payload.resize(self.gpa, length);
        const read = self.reader.readSliceShort(self.payload.items) catch return error.TruncatedFrame;
        if (read < length) return error.TruncatedFrame;
        return .{ .kind = kind, .bytes = self.payload.items };
    }

    /// The next frame that is not a failure; a failure is `PeerFailed`, its text in `said`.
    fn take(self: *Channel) Error!Frame {
        const frame = try self.receive();
        if (frame.kind != .failure) return frame;
        self.say("{s}", .{frame.bytes[0..@min(frame.bytes.len, self.said_buffer.len)]});
        return error.PeerFailed;
    }

    /// A request to the peer: opaque here.
    pub fn request(self: *Channel, bytes: []const u8) Error!void {
        return self.send(.request, bytes);
    }

    /// What the peer sent next: a request, or a barrier it waits at (`pass` answers it).
    pub fn next(self: *Channel) Error!Message {
        const frame = try self.take();
        return switch (frame.kind) {
            .request => .{ .request = frame.bytes },
            .barrier => if (frame.bytes.len == 8) .{ .barrier = std.mem.readInt(u64, frame.bytes[0..8], .little) } else error.UnexpectedFrame,
            else => error.UnexpectedFrame,
        };
    }

    /// The follower's half of `barrier`: answers `counter`, a peer's `barrier` call completing.
    pub fn pass(self: *Channel, counter: u64) Error!void {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, counter, .little);
        return self.send(.barrier, &bytes);
    }

    /// Both ranks call this at the same step with the same counter, and neither returns before the other
    /// has reached it. A peer at another step is `OutOfStep`, and is told.
    pub fn barrier(self: *Channel, counter: u64) Error!void {
        try self.pass(counter);
        const theirs = switch (try self.next()) {
            .barrier => |theirs| theirs,
            .request => return error.UnexpectedFrame,
        };
        if (theirs == counter) return;
        self.say("out of step: barrier {d} here, {d} on the other rank", .{ counter, theirs });
        self.fail(self.said());
        return error.OutOfStep;
    }

    /// Tells the peer this rank fails and why; the peer's next step is `PeerFailed`. What cannot be sent is
    /// not an error: the peer then meets a closed connection.
    pub fn fail(self: *Channel, text: []const u8) void {
        self.send(.failure, text[0..@min(text.len, self.said_buffer.len)]) catch {};
    }

    /// A rank's exit status when the channel ended its step: the peer's failure, or the channel's error.
    pub fn gone(channel: *Channel, rank: usize, err: Error) u8 {
        if (err == error.PeerFailed) {
            std.log.err("rank {d}: the other rank failed: {s}", .{ rank, channel.said() });
        } else std.log.err("rank {d}: the control channel: {t}{s}{s}", .{ rank, err, if (channel.said().len > 0) ": " else "", channel.said() });
        return 1;
    }

    /// A rank's exit status when its own step failed: logged, and the peer told.
    pub fn failWith(channel: *Channel, rank: usize, err: anyerror) u8 {
        var buffer: [160]u8 = undefined;
        const text = std.fmt.bufPrint(&buffer, "rank {d}: {t}", .{ rank, err }) catch "rank failed";
        std.log.err("{s}", .{text});
        channel.fail(text);
        return 1;
    }

    fn hello(self: *Channel, mine: Identity, rank: usize, id: *const Id) Error!void {
        var buffer: [16 + facets.len * (Sha256.digest_length + 1 + max_label) + @sizeOf(Id)]u8 = undefined;
        var at: usize = 0;
        @memcpy(buffer[0..magic.len], magic);
        std.mem.writeInt(u32, buffer[4..8], protocol, .little);
        buffer[8] = @intCast(rank);
        buffer[9] = world;
        at = 10;
        inline for (facets) |facet| {
            const one: Named = @field(mine, facet.field);
            const label = one.label[0..@min(one.label.len, max_label)];
            @memcpy(buffer[at..][0..one.digest.len], &one.digest);
            buffer[at + one.digest.len] = @intCast(label.len);
            @memcpy(buffer[at + one.digest.len + 1 ..][0..label.len], label);
            at += one.digest.len + 1 + label.len;
        }
        @memcpy(buffer[at..][0..id.len], id);
        return self.send(.hello, buffer[0 .. at + id.len]);
    }

    /// The peer's hello against `mine`: what differs is refused by name, the peer told, and `said` holds it.
    fn greeted(self: *Channel, mine: Identity, rank: usize, id: ?*Id) Error!void {
        const frame = try self.take();
        if (frame.kind != .hello) return error.UnexpectedFrame;
        const bytes = frame.bytes;
        if (bytes.len < 10 or !std.mem.eql(u8, bytes[0..magic.len], magic)) return error.NotARank;
        const theirs = std.mem.readInt(u32, bytes[4..8], .little);
        if (theirs != protocol) {
            self.say("the channel's protocols differ: {d} on rank {d}, {d} on the other", .{ protocol, rank, theirs });
            self.fail(self.said());
            return error.ProtocolsDiffer;
        }
        if (bytes[8] == rank or bytes[8] >= world or bytes[9] != world) {
            self.say("both ranks are rank {d} of {d}", .{ rank, world });
            self.fail(self.said());
            return error.RanksCollide;
        }
        var at: usize = 10;
        var refused: ?Error = null;
        inline for (facets) |facet| {
            const one: Named = @field(mine, facet.field);
            if (bytes.len < at + one.digest.len + 1) return error.NotARank;
            const length = bytes[at + one.digest.len];
            if (bytes.len < at + one.digest.len + 1 + length) return error.NotARank;
            const label = bytes[at + one.digest.len + 1 ..][0..length];
            if (refused == null and !std.mem.eql(u8, bytes[at..][0..one.digest.len], &one.digest)) {
                self.say(facet.what ++ " differ: '{s}' on rank {d}, '{s}' on rank {d}", .{ one.label[0..@min(one.label.len, max_label)], rank, label, bytes[8] });
                refused = facet.err;
            }
            at += one.digest.len + 1 + length;
        }
        if (refused) |err| {
            self.fail(self.said());
            return err;
        }
        if (bytes.len != at + @sizeOf(Id)) return error.NotARank;
        if (id) |out| @memcpy(out, bytes[at..][0..@sizeOf(Id)]);
    }

    /// Rank 0's first step: its hello, with the id every rank joins NCCL with.
    pub fn offer(self: *Channel, mine: Identity, id: *const Id) Error!void {
        return self.hello(mine, 0, id);
    }

    /// Rank 1's whole handshake: rank 0's hello checked against `mine`, `id` taken from it, its own sent back.
    pub fn answer(self: *Channel, mine: Identity, id: *Id) Error!void {
        try self.greeted(mine, 1, id);
        return self.hello(mine, 1, &@as(Id, @splat(0)));
    }

    /// Rank 0's last step: rank 1's hello, or its refusal.
    pub fn conclude(self: *Channel, mine: Identity) Error!void {
        return self.greeted(mine, 0, null);
    }

    /// The handshake as `rank`: rank 0 gives `id`, rank 1 receives it. Both hold the same build, checkpoint
    /// and options after it, or both fail with what differs in `said`.
    pub fn join(self: *Channel, rank: usize, mine: Identity, id: *Id) Error!void {
        if (rank != 0) return self.answer(mine, id);
        try self.offer(mine, id);
        return self.conclude(mine);
    }
};

/// The connection a channel runs over. Rank 1 listens at an address and takes one connection; rank 0 connects
/// to it, trying again while rank 1 is still starting.
pub const Link = struct {
    io: Io,
    stream: net.Stream,
    reader: net.Stream.Reader,
    writer: net.Stream.Writer,
    read_buffer: [4096]u8,
    write_buffer: [4096]u8,

    /// How long rank 0 tries: rank 1 is started first, and may still be opening its device. Rank 1 listens
    /// twice as long and then leaves with `error.NoPeer`: a rank 0 that gave up while rank 1 waited for its
    /// machine's lock would leave it holding that lock for nobody.
    pub const Patience = struct { tries: usize = 120, milliseconds: i64 = 500 };

    /// In place: the reader and the writer keep this address.
    pub fn open(self: *Link, io: Io, rank: usize, at: net.IpAddress, patience: Patience) !void {
        if (rank == 0) {
            var tries: usize = 0;
            const stream = while (true) : (tries += 1) {
                break at.connect(io, .{ .mode = .stream }) catch |err| {
                    if (tries + 1 >= patience.tries) return err;
                    try io.sleep(.fromMilliseconds(patience.milliseconds), .awake);
                    continue;
                };
            };
            return self.over(io, stream);
        }
        var server = try at.listen(io, .{ .reuse_address = true });
        defer server.deinit(io);
        var fds = [1]std.posix.pollfd{.{ .fd = server.socket.handle, .events = std.posix.POLL.IN, .revents = 0 }};
        const wait: i64 = 2 * @as(i64, @intCast(patience.tries)) * patience.milliseconds;
        if (try std.posix.poll(&fds, @intCast(wait)) == 0) return error.NoPeer;
        return self.over(io, try server.accept(io));
    }

    /// Over a connection already made.
    pub fn over(self: *Link, io: Io, stream: net.Stream) void {
        self.io = io;
        self.stream = stream;
        // a frame is small and the peer answers none of a lead's: none waits for the last one's acknowledgement
        std.posix.setsockopt(stream.socket.handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &std.mem.toBytes(@as(c_int, 1))) catch {};
        self.reader = stream.reader(io, &self.read_buffer);
        self.writer = stream.writer(io, &self.write_buffer);
    }

    pub fn channel(self: *Link, gpa: std.mem.Allocator) Channel {
        var made: Channel = .init(gpa, &self.reader.interface, &self.writer.interface);
        made.fd = self.stream.socket.handle;
        return made;
    }

    pub fn close(self: *Link) void {
        self.stream.close(self.io);
        self.* = undefined;
    }
};

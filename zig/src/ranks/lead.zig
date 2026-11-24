//! Rank 0 leads, rank 1 follows (spec 045 WS-04, TensorFold's `families/glm/mirror.zig`): rank 0 alone chooses
//! (the sampling, a constraint, when a reply ends, a client that hung up), and before each operation on the
//! device it tells rank 1 which, through the control channel; rank 1 runs the same operations in the same
//! order, so the gathers inside them meet. A model names its operations (`ranks_forward.zig`'s `Op` for GLM) and reads
//! them back (`mirror.zig`); here is the frame, and the watcher of the connection.
//!
//! An operation is a request frame of u32 words: its kind, its arguments, then a window's tokens. Where the
//! ranks say words to each other over NCCL (`ranks.Said`), a `hot` lead sends its frames there instead, a
//! word before each saying what follows (`Heard`).

const std = @import("std");
const channel_file = @import("channel.zig");

pub const Error = error{PeerGone};

pub const Lead = struct {
    channel: *channel_file.Channel,
    /// A `hot` frame's first word (`Heard`), then the frame.
    words: [1 + widest]u32 = undefined,
    /// Operations told.
    told: usize = 0,
    /// The follower runs these operations of its own (Flash Next's reply rounds, spec 095): `tell` sends
    /// nothing while set, and what the follower cannot derive reaches it another way.
    muted: bool = false,
    /// Words said to every rank in the device's stream (`ranks.Said`, over NCCL) where the ranks have them.
    /// While `hot` the frames go there (spec 144 WS-10): one wakes a follower waiting in NCCL in some 20 us,
    /// where the channel's took 110 to 120 one way (WS-07's traces). Set only where both ranks know the
    /// follower listens there (`qwen4/pool/hall_ranks.zig`), and never across an idle wait: the follower's
    /// wait is a kernel on its device.
    over: ?Over = null,
    hot: bool = false,

    /// What `hot` frames go through: `say` gives every rank `words`, at most `width` of them, in the stream's
    /// order, without waiting for the device.
    pub const Over = struct {
        context: *anyopaque,
        width: usize,
        say: *const fn (context: *anyopaque, words: []const u32) anyerror!void,
    };

    /// A frame's words at most: a kind, a few arguments and a prompt chunk's tokens. GLM's rounds are cut to
    /// it (`glm5/pool/hall.zig`); Flash Next's frames are `widest`.
    pub const most = 8 + 1024;
    /// The widest frame: a round of the pool on two ranks (`qwen4/pool/hall_ranks.zig`), a few words a
    /// stream, every row's token and as many of the next chunk's (asked ahead, spec 149 WS-02), a round of
    /// 4,096 rows (WS-06) beside 64 streams' words; and Flash Next's pair (`qwen4/session/pair*.zig`,
    /// `mirror.zig`): a window of 4,096 rows told a pass, a prompt's frame, the rows its last window chose
    /// beside a reply's first arguments (spec 111).
    pub const widest = 8 + 2 * 64 + 2 * 4096;

    /// Before the operation runs here. Whatever the channel fails with, the peer is gone for the caller.
    pub fn tell(self: *Lead, kind: u32, arguments: []const u32, tokens: []const i32) Error!void {
        if (self.muted) return;
        const n = 1 + arguments.len + tokens.len;
        std.debug.assert(n <= widest);
        const frame = self.words[1..][0..n];
        frame[0] = kind;
        @memcpy(frame[1..][0..arguments.len], arguments);
        for (frame[1 + arguments.len ..][0..tokens.len], tokens) |*word, token| word.* = @bitCast(token);
        if (self.hot) if (self.over) |over| {
            // a frame wider than the words goes on the channel, the words saying so first
            const fits = n < over.width;
            self.words[0] = if (fits) @intCast(n) else Heard.channel_word;
            over.say(over.context, self.words[0..if (fits) 1 + n else 1]) catch |err| {
                std.log.err("the ranks' words, telling operation {d}: {t}", .{ kind, err });
                return error.PeerGone;
            };
            if (fits) {
                self.told += 1;
                return;
            }
        };
        self.channel.request(std.mem.sliceAsBytes(frame)) catch |err| {
            std.log.err("the control channel, telling operation {d}: {t}", .{ kind, err });
            return error.PeerGone;
        };
        self.told += 1;
    }

    /// From here frames go `over`, where the ranks have it.
    pub fn heat(self: *Lead) void {
        self.hot = self.over != null;
    }

    /// The frames go on the channel again, the follower told so (`Heard.cooled`); nothing where not `hot`.
    pub fn cool(self: *Lead) Error!void {
        if (!self.hot) return;
        self.hot = false;
        const over = self.over.?;
        const words = [1]u32{Heard.cooled_word};
        over.say(over.context, &words) catch |err| {
            std.log.err("the ranks' words, back to the channel: {t}", .{err});
            return error.PeerGone;
        };
    }
};

/// What the follower hears of a `hot` frame: the first word says what the rest is (spec 144 WS-10).
pub const Heard = union(enum) {
    /// The operation, valid until the next words are heard.
    told: Told,
    /// The operation is the channel's next frame.
    channel,
    /// The leader went back to the channel (`Lead.cool`): every frame from here is the channel's.
    cooled,

    /// The first word of what sends the frame on the channel, and of what cools.
    pub const channel_word: u32 = std.math.maxInt(u32);
    pub const cooled_word: u32 = 0;

    /// `words`, all that was heard; null where they are no frame.
    pub fn read(words: []const u32) ?Heard {
        if (words.len < 1) return null;
        const n = words[0];
        if (n == cooled_word) return .cooled;
        if (n == channel_word) return .channel;
        if (n >= words.len) return null;
        return .{ .told = .{ .kind = words[1], .rest = words[2 .. 1 + n] } };
    }
};

/// An operation as the follower reads it: valid until the channel's next frame.
pub const Told = struct {
    kind: u32,
    /// The arguments and then the tokens: the kind says how many are which.
    rest: []align(1) const u32,

    pub fn read(bytes: []const u8) ?Told {
        if (bytes.len < 4 or bytes.len % 4 != 0) return null;
        const words: []align(1) const u32 = @ptrCast(bytes);
        return .{ .kind = words[0], .rest = words[1..] };
    }

    /// The tokens after `arguments` arguments, into `out`.
    pub fn tokens(self: Told, arguments: usize, out: []i32) ?[]const i32 {
        if (self.rest.len < arguments or self.rest.len - arguments > out.len) return null;
        const from = self.rest[arguments..];
        for (out[0..from.len], from) |*token, word| token.* = @bitCast(word);
        return out[0..from.len];
    }
};

/// Watches the connection while this rank computes: a rank inside an all-gather waits in NCCL for a peer that
/// may be gone, and nothing there returns. The watcher's thread polls the socket and ends the process when
/// the peer hung up, with the peer's last words where it sent any (rank 0 reads nothing else after the
/// handshake, so a frame that arrives there is rank 1's failure). `stop` before an orderly close.
pub const Watcher = struct {
    handle: std.posix.fd_t,
    rank: usize,
    /// Rank 0's: the watcher reads a failure's text from it. Null on rank 1, whose own thread reads.
    channel: ?*channel_file.Channel,
    closing: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// In place: the thread keeps this address.
    pub fn start(self: *Watcher) !void {
        self.thread = try std.Thread.spawn(.{}, watch, .{self});
    }

    /// The peer may close from here on; returns once the thread has ended.
    pub fn stop(self: *Watcher) void {
        self.closing.store(true, .release);
        if (self.thread) |thread| thread.join();
        self.thread = null;
    }

    fn watch(self: *Watcher) void {
        // POLLRDHUP (Linux; `std.posix.POLL` does not name it): the peer closed, or shut down its writing
        const hung: i16 = 0x2000 | std.posix.POLL.HUP | std.posix.POLL.ERR;
        const reads: i16 = if (self.channel != null) std.posix.POLL.IN else 0;
        while (!self.closing.load(.acquire)) {
            var fds = [1]std.posix.pollfd{.{ .fd = self.handle, .events = hung | reads, .revents = 0 }};
            const ready = std.posix.poll(&fds, 100) catch return;
            if (ready == 0) continue;
            if (self.closing.load(.acquire)) return;
            if (self.channel) |channel| {
                if (fds[0].revents & std.posix.POLL.IN != 0) {
                    if (channel.next()) |_| {
                        std.log.err("rank {d}: the other rank sent a frame where it only follows", .{self.rank});
                    } else |err| if (err == error.PeerFailed) {
                        std.log.err("rank {d}: the other rank failed: {s}", .{ self.rank, channel.said() });
                    } else std.log.err("rank {d}: the other rank is gone ({t})", .{ self.rank, err });
                    std.process.exit(1);
                }
            }
            std.log.err("rank {d}: the other rank is gone", .{self.rank});
            std.process.exit(1);
        }
    }
};

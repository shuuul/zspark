//! Two processes become two ranks (spec 045): the control channel first (`channel.zig`: rank 1 listens, rank 0
//! connects; what both must hold alike is compared, and NCCL's unique id crosses), then NCCL's communicator.
//! What `ranks-check` starts with, and a served pair after it.

const std = @import("std");
const cli = @import("../cli.zig");
const channel_file = @import("channel.zig");
const nccl = @import("nccl.zig");

pub const world = channel_file.world;

/// `rank=R at=ADDR:PORT`, the words every two-rank command takes: rank 1 listens there, rank 0 connects.
pub const Place = struct {
    rank: ?usize = null,
    at: ?std.Io.net.IpAddress = null,

    /// Whether `word` was one of the two; a value that cannot be read leaves its field null.
    pub fn take(self: *Place, word: []const u8) bool {
        if (cli.value(word, "rank=")) |text| {
            self.rank = std.fmt.parseInt(usize, text, 10) catch null;
        } else if (cli.value(word, "at=")) |text| {
            const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return true;
            self.at = std.Io.net.IpAddress.parse(text[0..colon], std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return true) catch null;
        } else return false;
        return true;
    }

    pub fn complete(self: Place) bool {
        return self.rank != null and self.rank.? < world and self.at != null;
    }
};

/// `words` without `rank=`, `at=`, `--rank` and `--at`. A flag without its value is `error.InvalidArgument`.
pub fn rest(arena: std.mem.Allocator, words: []const []const u8, place: *Place) ![]const []const u8 {
    var kept: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < words.len) : (i += 1) {
        if (place.take(words[i])) continue;
        if (std.mem.eql(u8, words[i], "--rank") or std.mem.eql(u8, words[i], "--at")) {
            const named = if (std.mem.eql(u8, words[i], "--rank")) "rank" else "at";
            i += 1;
            if (i == words.len) return error.InvalidArgument;
            _ = place.take(try std.fmt.allocPrint(arena, "{s}={s}", .{ named, words[i] }));
            continue;
        }
        try kept.append(arena, words[i]);
    }
    return kept.items;
}

/// The checkpoint's headers as both ranks must hold them: every tensor's name, type and shape, by name.
/// `checkpoint` is a `*const safetensors.Checkpoint`, which this layer does not import.
pub fn headers(arena: std.mem.Allocator, checkpoint: anytype) !channel_file.Named {
    const names = try arena.dupe([]const u8, checkpoint.tensors.keys());
    std.mem.sort([]const u8, names, {}, struct {
        fn before(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.before);
    var parts: std.ArrayList([]const u8) = .empty;
    for (names) |name| {
        const tensor = checkpoint.tensors.get(name).?;
        try parts.append(arena, name);
        try parts.append(arena, @tagName(tensor.dtype));
        try parts.append(arena, std.mem.sliceAsBytes(tensor.shape));
    }
    return .{ .digest = channel_file.digest(parts.items), .label = try std.fmt.allocPrint(arena, "{d} tensors' headers", .{names.len}) };
}

/// Where rank 1 holds no checkpoint (`serve --weights-from-peer`), what it needs of rank 0's before they join:
/// `blobs` (its config, its files' headers) over a connection of their own, one port above `place`'s, where rank
/// 1 listens as it does for the join. Rank 0 sends (`hand`), rank 1 receives (`handed`).
pub fn hand(io: std.Io, gpa: std.mem.Allocator, place: Place, blobs: []const []const u8) !void {
    var link: channel_file.Link = undefined;
    try link.open(io, 0, handing(place), .{});
    defer link.close();
    var channel = link.channel(gpa);
    defer channel.deinit();
    var count: [8]u8 = undefined;
    std.mem.writeInt(u64, &count, blobs.len, .little);
    try channel.request(&count);
    for (blobs) |blob| {
        std.mem.writeInt(u64, &count, blob.len, .little);
        try channel.request(&count);
        var at: usize = 0;
        while (at < blob.len) : (at += channel_file.max_payload) try channel.request(blob[at..][0..@min(channel_file.max_payload, blob.len - at)]);
    }
    // rank 1 has them all before the connection closes
    _ = try channel.next();
}

/// Rank 1's side of `hand`: the blobs, in `arena`.
pub fn handed(io: std.Io, gpa: std.mem.Allocator, arena: std.mem.Allocator, place: Place) ![]const []const u8 {
    var link: channel_file.Link = undefined;
    try link.open(io, 1, handing(place), .{});
    defer link.close();
    var channel = link.channel(gpa);
    defer channel.deinit();
    const blobs = try arena.alloc([]const u8, try handedCount(&channel));
    for (blobs) |*blob| {
        const bytes = try arena.alloc(u8, try handedCount(&channel));
        var at: usize = 0;
        while (at < bytes.len) {
            const part = switch (try channel.next()) {
                .request => |part| part,
                .barrier => return error.UnexpectedFrame,
            };
            if (part.len > bytes.len - at) return error.UnexpectedFrame;
            @memcpy(bytes[at..][0..part.len], part);
            at += part.len;
        }
        blob.* = bytes;
    }
    try channel.request("taken");
    return blobs;
}

fn handedCount(channel: *channel_file.Channel) !usize {
    const bytes = switch (try channel.next()) {
        .request => |bytes| bytes,
        .barrier => return error.UnexpectedFrame,
    };
    if (bytes.len != 8) return error.UnexpectedFrame;
    return std.math.cast(usize, std.mem.readInt(u64, bytes[0..8], .little)) orelse error.UnexpectedFrame;
}

fn handing(place: Place) std.Io.net.IpAddress {
    var at = place.at.?;
    at.setPort(at.getPort() +% 1);
    return at;
}

/// This binary as `zspark version` names it: both ranks run the same commit, of a tree as clean.
pub fn build() channel_file.Named {
    return .of(cli.stamp.commit ++ if (cli.stamp.dirty) " dirty" else "");
}

pub const Joined = struct {
    link: channel_file.Link,
    channel: channel_file.Channel,
    comm: nccl.Comm,

    /// In place. `checkpoint` and `options` are what the command holds the ranks alike in, beside the build
    /// and NCCL's version; a difference ends both ranks with its name logged on each.
    pub fn open(self: *Joined, io: std.Io, gpa: std.mem.Allocator, place: Place, checkpoint: channel_file.Named, options: []const u8) !void {
        const rank = place.rank.?;
        // the whole text is compared (`Named.of`), its label alone is cut
        var text_buffer: [1024]u8 = undefined;
        const text = std.fmt.bufPrint(&text_buffer, "NCCL {d}, {s}", .{ try nccl.version(), options }) catch &text_buffer;
        try self.link.open(io, rank, place.at.?, .{});
        errdefer self.link.close();
        self.channel = self.link.channel(gpa);
        errdefer self.channel.deinit();
        var id: nccl.UniqueId = if (rank == 0) try nccl.uniqueId() else undefined;
        self.channel.join(rank, .{ .build = build(), .checkpoint = checkpoint, .options = .of(text) }, &id.internal) catch |err| {
            if (self.channel.said().len > 0) std.log.err("rank {d}: {s}", .{ rank, self.channel.said() });
            return err;
        };
        self.comm = try .init(world, id, rank);
    }

    pub fn close(self: *Joined) void {
        self.comm.deinit();
        self.channel.deinit();
        self.link.close();
        self.* = undefined;
    }
};

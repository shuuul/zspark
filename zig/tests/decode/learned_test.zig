//! `learned.zig`: a file read back, what is never read, and the cap.

const std = @import("std");
const testing = std.testing;
const learned = @import("../../src/decode/learned.zig");
const Store = learned.Store;

const fx = struct {
    fn put(store: *Store, tokens: []const i32, parts: []const []const u8) !void {
        try store.putHost(tokens, parts, &.{});
    }

    fn held(store: *Store, tokens: []const i32) !bool {
        var image = try store.find(testing.allocator, tokens) orelse return false;
        image.close(testing.allocator);
        return true;
    }
};

test "a file written is read back part by part, and by no other tokens, sizes or identity" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 0xabc, 1 << 20);
    defer store.close();
    var tokens: [3000]i32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(i * 7 % 1000);
    try testing.expect(!store.has(&tokens));
    try testing.expect(try store.find(testing.allocator, &tokens) == null);
    try fx.put(&store, &tokens, &.{ "first part", "", "the third" });
    try testing.expect(store.has(&tokens));
    try testing.expectEqual(1, store.written);

    var image = (try store.find(testing.allocator, &tokens)).?;
    try testing.expectEqual(3, image.parts.len);
    for ([_]u64{ 10, 0, 9 }, image.parts) |bytes, part| try testing.expectEqual(bytes, part.bytes);
    var first: [10]u8 = undefined;
    var third: [9]u8 = undefined;
    // a part of another size is refused before a byte of it is read
    try testing.expectError(error.Mismatch, image.part(third[0..]));
    try image.part(&first);
    try image.part(&.{});
    try image.part(&third);
    try testing.expectEqualStrings("first part", &first);
    try testing.expectEqualStrings("the third", &third);
    try testing.expectError(error.Mismatch, image.part(&.{}));
    image.close(testing.allocator);

    // a shorter prefix and one that differs in its last token are other files
    try testing.expect(!try fx.held(&store, tokens[0..2999]));
    tokens[2999] += 1;
    try testing.expect(!try fx.held(&store, &tokens));
    tokens[2999] -= 1;

    // another identity reads nothing of this one, and its directory is its own
    var other: Store = try .open(testing.allocator, io, tmp.dir, 0xabd, 1 << 20);
    defer other.close();
    try testing.expect(!try fx.held(&other, &tokens));
    try testing.expect(try fx.held(&store, &tokens));
}

test "a file that is not what its name says is deleted and not read" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 7, 1 << 20);
    defer store.close();
    const tokens = [_]i32{ 5, 6, 7, 8 };

    const Break = enum { cut, longer, identity, tokens, magic };
    for (std.enums.values(Break)) |how| {
        try fx.put(&store, &tokens, &.{ "abcdefgh", "ij" });
        const file_name = learned.nameOf(learned.keyOf(&tokens));
        const file = try store.dir.openFile(io, &file_name, .{ .mode = .read_write });
        const length = try file.length(io);
        switch (how) {
            .cut => try file.setLength(io, length - 1),
            .longer => try file.writePositionalAll(io, "x", length),
            .identity => try file.writePositionalAll(io, &.{9}, 8),
            .tokens => try file.writePositionalAll(io, &.{9}, 24 + 4),
            .magic => try file.writePositionalAll(io, "y", 0),
        }
        file.close(io);
        try testing.expect(store.has(&tokens));
        try testing.expect(try store.find(testing.allocator, &tokens) == null);
        try testing.expect(!store.has(&tokens));
    }
}

test "past the cap other identities' files go first, then the least recently read; a write cut short is cleared" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const body: [1000]u8 = @splat(1);
    const a = [_]i32{1};
    const b = [_]i32{2};
    const c = [_]i32{3};
    const d = [_]i32{4};
    {
        var old: Store = try .open(testing.allocator, io, tmp.dir, 1, 1 << 20);
        defer old.close();
        try fx.put(&old, &a, &.{&body});
        try fx.put(&old, &b, &.{&body});
        // a write that never finished
        try old.dir.writeFile(io, .{ .sub_path = &learned.partOf(learned.keyOf(&c)), .data = &body });
    }
    // room for three files: the newer identity's two and one of the older's
    var store: Store = try .open(testing.allocator, io, tmp.dir, 2, 3 * 1100);
    defer store.close();
    try fx.put(&store, &a, &.{&body});
    try fx.put(&store, &b, &.{&body});
    {
        var old: Store = try .open(testing.allocator, io, tmp.dir, 1, 1 << 20);
        defer old.close();
        try testing.expect(!old.has(&a));
        try testing.expect(old.has(&b));
        try testing.expect(!try fx.held(&old, &c));
        var parts: usize = 0;
        var files = old.dir.iterate();
        while (try files.next(io)) |entry| parts += @intFromBool(std.mem.endsWith(u8, entry.name, ".part"));
        try testing.expectEqual(0, parts);
    }
    // a third pushes the older identity out whole, its directory with it
    try fx.put(&store, &c, &.{&body});
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "0000000000000001", .{}));
    try testing.expect(store.has(&a) and store.has(&b) and store.has(&c));
    // `a` is read, so `b` is the one a fourth pushes out
    try testing.expect(try fx.held(&store, &a));
    try fx.put(&store, &d, &.{&body});
    try testing.expect(store.has(&a) and !store.has(&b) and store.has(&c) and store.has(&d));
    try testing.expect(!store.fits(4000));
    try testing.expect(store.fits(1000));
}

test "the longest file a prompt begins with is found without a read, by a store opened later too, and not once it is gone" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var prompt: [400]i32 = undefined;
    for (&prompt, 0..) |*token, i| token.* = @intCast(i * 13 % 997);
    var other = prompt;
    other[150] += 1;
    {
        var store: Store = try .open(testing.allocator, io, tmp.dir, 3, 1 << 20);
        defer store.close();
        try testing.expectEqual(null, store.longest(&prompt, prompt.len));
        try fx.put(&store, prompt[0..100], &.{"a"});
        try fx.put(&store, prompt[0..300], &.{"b"});
        try fx.put(&store, other[0..200], &.{"c"});
        try testing.expectEqual(300, store.longest(&prompt, prompt.len));
        // one shorter than a length, and no longer than the prompt
        try testing.expectEqual(100, store.longest(&prompt, 300));
        try testing.expectEqual(null, store.longest(&prompt, 100));
        try testing.expectEqual(100, store.longest(prompt[0..299], 400));
        try testing.expectEqual(200, store.longest(&other, other.len));
        try testing.expectEqual(0, store.read);
    }
    var store: Store = try .open(testing.allocator, io, tmp.dir, 3, 1 << 20);
    defer store.close();
    try testing.expectEqual(3, store.held.items.len);
    try testing.expectEqual(300, store.longest(&prompt, prompt.len));
    store.discard(prompt[0..300]);
    try testing.expect(!store.has(prompt[0..300]));
    try testing.expectEqual(100, store.longest(&prompt, prompt.len));
    // written again, it is found again, once
    try fx.put(&store, prompt[0..300], &.{"b"});
    try fx.put(&store, prompt[0..300], &.{"b"});
    try testing.expectEqual(3, store.held.items.len);
    try testing.expectEqual(300, store.longest(&prompt, prompt.len));
    // another identity's are not this one's
    var elsewhere: Store = try .open(testing.allocator, io, tmp.dir, 4, 1 << 20);
    defer elsewhere.close();
    try testing.expectEqual(null, elsewhere.longest(&prompt, prompt.len));
}

test "a part is read in pieces that end where it ends, and a file pushed out by the cap is no longer found" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 5, 1 << 20);
    defer store.close();
    const tokens = [_]i32{ 1, 2, 3 };
    try fx.put(&store, &tokens, &.{ "abcdefgh", "ij" });
    var image = (try store.find(testing.allocator, &tokens)).?;
    defer image.close(testing.allocator);
    var a: [3]u8 = undefined;
    var b: [5]u8 = undefined;
    var c: [2]u8 = undefined;
    try image.some(&a);
    // a whole part is not read from inside one, nor a piece past its end
    try testing.expectError(error.Mismatch, image.part(&b));
    var past: [6]u8 = undefined;
    try testing.expectError(error.Mismatch, image.some(&past));
    try image.some(&b);
    try image.part(&c);
    try testing.expectEqualStrings("abc", &a);
    try testing.expectEqualStrings("defgh", &b);
    try testing.expectEqualStrings("ij", &c);

    // room for two files: the third pushes the oldest out, and it is found no more
    var small_dir = testing.tmpDir(.{ .iterate = true });
    defer small_dir.cleanup();
    var small: Store = try .open(testing.allocator, io, small_dir.dir, 5, 2 * 1100);
    defer small.close();
    const body: [1000]u8 = @splat(2);
    const long = [_]i32{ 7, 8, 9, 10 };
    try fx.put(&small, long[0..1], &.{&body});
    try fx.put(&small, long[0..2], &.{&body});
    try testing.expectEqual(1, small.longest(&long, 2));
    try fx.put(&small, long[0..3], &.{&body});
    try testing.expectEqual(3, small.longest(&long, 4));
    try testing.expectEqual(2, small.longest(&long, 3));
    try testing.expectEqual(null, small.longest(&long, 2));
    try testing.expectEqual(2, small.held.items.len);
}

/// A state of rows by block (spec 101): its own bytes, a cache's rows (3 bytes a position) and pooled keys
/// (2 bytes every 4), each row its tokens' alone, as the row rule makes a model's.
const Rowed = struct {
    own: [50]u8,
    rows: []u8,
    pooled: []u8,

    const per = [_]u32{ 0, 1, 4 };

    fn init(tokens: []const i32) !Rowed {
        const at = tokens.len - 1;
        var self: Rowed = .{ .own = @splat(@truncate(tokens.len)), .rows = try testing.allocator.alloc(u8, at * 3), .pooled = try testing.allocator.alloc(u8, at / 4 * 2) };
        for (0..at) |i| for (0..3) |b| {
            self.rows[i * 3 + b] = @truncate(@as(u32, @bitCast(tokens[i] *% 7 +% tokens[i + 1])) +% @as(u32, @intCast(b)));
        };
        for (0..at / 4) |i| for (0..2) |b| {
            self.pooled[i * 2 + b] = @truncate(@as(u32, @bitCast(tokens[4 * i + 3])) +% @as(u32, @intCast(b * 5)));
        };
        return self;
    }

    fn deinit(self: *Rowed) void {
        testing.allocator.free(self.rows);
        testing.allocator.free(self.pooled);
    }

    fn parts(self: *const Rowed) [3][]const u8 {
        return .{ &self.own, self.rows, self.pooled };
    }

    fn put(store: *Store, tokens: []const i32) !void {
        var one: Rowed = try .init(tokens);
        defer one.deinit();
        const all = one.parts();
        try store.putHost(tokens, &all, &per);
    }

    /// The state for `tokens` is held and reads back as written.
    fn expectHeld(store: *Store, tokens: []const i32) !void {
        var one: Rowed = try .init(tokens);
        defer one.deinit();
        var image = try store.find(testing.allocator, tokens) orelse return error.TestExpectedState;
        defer image.close(testing.allocator);
        for (one.parts(), image.parts, per) |want, part, rows| {
            try testing.expectEqual(rows, part.per);
            const got = try testing.allocator.alloc(u8, want.len);
            defer testing.allocator.free(got);
            try image.part(got);
            try testing.expectEqualSlices(u8, want, got);
        }
    }
};

fn filesEnding(io: std.Io, dir: std.Io.Dir, ending: []const u8) !usize {
    var n: usize = 0;
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| n += @intFromBool(std.mem.endsWith(u8, entry.name, ending));
    return n;
}

fn bytesUnder(io: std.Io, dir: std.Io.Dir) !u64 {
    var n: u64 = 0;
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| n += (try dir.statFile(io, entry.name, .{})).size;
    return n;
}

fn tokensOf(buffer: []i32, seed: i32) []i32 {
    for (buffer, 0..) |*token, i| token.* = seed +% @as(i32, @intCast(i * 31 % 1009));
    return buffer;
}

test "by block: a state's rows lie in its whole blocks and the rest in its file, and read back as written, by a later store too" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var ids: [700]i32 = undefined;
    const tokens = tokensOf(&ids, 1);
    {
        var store: Store = try .open(testing.allocator, io, tmp.dir, 11, 1 << 20);
        defer store.close();
        try Rowed.put(&store, tokens);
        try Rowed.expectHeld(&store, tokens);
        // position 699: blocks [0, 256) and [256, 512); the rows of 512 to 698 in the state's file
        try testing.expectEqual(2, try filesEnding(io, store.dir, ".blk"));
        try testing.expectEqual(1, try filesEnding(io, store.dir, ".st"));
        const own = (699 - 512) * 3 + (174 - 128) * 2 + 50;
        const st = try store.dir.statFile(io, &learned.nameOf(learned.keyOf(tokens)), .{});
        try testing.expectEqual(learned.prefixBytes(700, 3) + own, st.size);
        // a prefix of the state is no state, though its blocks are
        try testing.expect(try store.find(testing.allocator, tokens[0..600]) == null);
    }
    var store: Store = try .open(testing.allocator, io, tmp.dir, 11, 1 << 20);
    defer store.close();
    try testing.expectEqual(1, store.held.items.len);
    try Rowed.expectHeld(&store, tokens);
    // at 256 a state has its first block whole, the one above's; below, none
    try Rowed.put(&store, tokens[0..257]);
    try Rowed.expectHeld(&store, tokens[0..257]);
    try testing.expectEqual(2, try filesEnding(io, store.dir, ".blk"));
    try Rowed.put(&store, tokens[0..256]);
    try Rowed.expectHeld(&store, tokens[0..256]);
    try testing.expectEqual(2, try filesEnding(io, store.dir, ".blk"));
    const st = try store.dir.statFile(io, &learned.nameOf(learned.keyOf(tokens[0..256])), .{});
    try testing.expectEqual(learned.prefixBytes(256, 3) + 255 * 3 + 63 * 2 + 50, st.size);
}

test "by block: two states over one prefix share its blocks on disk, a block going with the last state that names it" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 12, 1 << 22);
    defer store.close();
    var first_ids: [700]i32 = undefined;
    var second_ids: [900]i32 = undefined;
    const first = tokensOf(&first_ids, 1);
    const second = tokensOf(&second_ids, 1);
    // they part at 600: blocks 0 and 1 (through token 512, its lookahead) are the same
    for (second[600..]) |*token| token.* +%= 5;
    try Rowed.put(&store, first);
    const alone = try bytesUnder(io, store.dir);
    try Rowed.put(&store, second);
    try testing.expectEqual(3, try filesEnding(io, store.dir, ".blk"));
    try Rowed.expectHeld(&store, first);
    try Rowed.expectHeld(&store, second);
    // the second added its third block and its file, not its first two
    const both = try bytesUnder(io, store.dir);
    const block = @sizeOf(@import("../../src/decode/learned_blocks.zig").BlockHeader) + 256 * 3 + 64 * 2;
    const second_file = learned.prefixBytes(900, 3) + (899 - 768) * 3 + (224 - 192) * 2 + 50;
    try testing.expectEqual(alone + block + second_file, both);

    // a third whose token 512 differs shares block 0 alone
    var third_ids: [700]i32 = undefined;
    const third = tokensOf(&third_ids, 1);
    third[512] +%= 1;
    try Rowed.put(&store, third);
    try testing.expectEqual(4, try filesEnding(io, store.dir, ".blk"));
    try Rowed.expectHeld(&store, third);

    store.discard(first);
    try testing.expectEqual(4, try filesEnding(io, store.dir, ".blk"));
    try Rowed.expectHeld(&store, second);
    store.discard(third);
    try testing.expectEqual(3, try filesEnding(io, store.dir, ".blk"));
    store.discard(second);
    try testing.expectEqual(0, try filesEnding(io, store.dir, ".blk"));
    try testing.expectEqual(0, try bytesUnder(io, store.dir));
}

test "by block: a block missing, cut, longer or another's is a miss and no error: the state goes, and its prompt runs" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var store: Store = try .open(testing.allocator, io, tmp.dir, 13, 1 << 22);
    defer store.close();
    var ids: [600]i32 = undefined;
    const tokens = tokensOf(&ids, 3);
    const Break = enum { missing, cut, longer, name, identity };
    for (std.enums.values(Break)) |how| {
        try Rowed.put(&store, tokens);
        // either of its two blocks
        var name: [36]u8 = undefined;
        var entries = store.dir.iterate();
        while (try entries.next(io)) |entry| if (std.mem.endsWith(u8, entry.name, ".blk")) @memcpy(&name, entry.name);
        const file: []const u8 = &name;
        switch (how) {
            .missing => try store.dir.deleteFile(io, file),
            else => {
                const open = try store.dir.openFile(io, file, .{ .mode = .read_write });
                defer open.close(io);
                const length = try open.length(io);
                switch (how) {
                    .cut => try open.setLength(io, length - 1),
                    .longer => try open.writePositionalAll(io, "x", length),
                    .name => try open.writePositionalAll(io, &.{0xff}, 16 + 3),
                    .identity => try open.writePositionalAll(io, &.{9}, 8),
                    .missing => unreachable,
                }
            },
        }
        try testing.expectEqual(600, store.longest(tokens, 601));
        try testing.expect(try store.find(testing.allocator, tokens) == null);
        try testing.expect(!store.has(tokens));
        try testing.expectEqual(null, store.longest(tokens, 601));
        // the blocks no state names go with it, the broken one too
        try testing.expectEqual(0, try filesEnding(io, store.dir, ".blk"));
    }
    // a store opened over a state missing a block drops it
    try Rowed.put(&store, tokens);
    var entries = store.dir.iterate();
    while (try entries.next(io)) |entry| if (std.mem.endsWith(u8, entry.name, ".blk")) {
        try store.dir.deleteFile(io, entry.name);
        break;
    };
    var later: Store = try .open(testing.allocator, io, tmp.dir, 13, 1 << 22);
    defer later.close();
    try testing.expectEqual(0, later.held.items.len);
    try testing.expectEqual(0, try filesEnding(io, later.dir, ".blk"));
    try testing.expectEqual(0, try filesEnding(io, later.dir, ".st"));
}

test "by block: past the cap the least recently used state goes, its blocks only where no state left names them" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var a_ids: [700]i32 = undefined;
    var b_ids: [700]i32 = undefined;
    var c_ids: [700]i32 = undefined;
    const a = tokensOf(&a_ids, 1);
    const b = tokensOf(&b_ids, 1);
    const c = tokensOf(&c_ids, 2);
    for (b[600..]) |*token| token.* +%= 5;
    // a and b share their two blocks; c is all its own
    var probe: Store = try .open(testing.allocator, io, tmp.dir, 14, 1 << 22);
    try Rowed.put(&probe, a);
    const one = try bytesUnder(io, probe.dir);
    try Rowed.put(&probe, b);
    const two = try bytesUnder(io, probe.dir);
    for ([_][]const i32{ a, b }) |tokens| probe.discard(tokens);
    probe.close();
    // room for a, b and c's file but not c's blocks too: a, the oldest, goes and the shared blocks stay
    var store: Store = try .open(testing.allocator, io, tmp.dir, 14, two + one - 1);
    defer store.close();
    try Rowed.put(&store, a);
    try Rowed.put(&store, b);
    try Rowed.put(&store, c);
    try testing.expect(!store.has(a) and store.has(b) and store.has(c));
    try testing.expectEqual(4, try filesEnding(io, store.dir, ".blk"));
    try Rowed.expectHeld(&store, b);
    try Rowed.expectHeld(&store, c);
}

/// The two kinds over one `Blocks`, as `learnt.open` makes them: the blocks under the learned root.
const Both = struct {
    shared: *learned.Blocks,
    learn: Store,
    spill: Store,

    fn open(io: std.Io, learn_root: std.Io.Dir, spill_root: std.Io.Dir, id: u64) !Both {
        const shared: *learned.Blocks = try .open(testing.allocator, io, learn_root, id, .{ .learned = true, .spilled = true, .spilled_root = spill_root });
        errdefer shared.close();
        var learn: Store = try .openKind(testing.allocator, io, learn_root, id, 1 << 22, .learned, shared);
        errdefer learn.close();
        const spill: Store = try .openKind(testing.allocator, io, spill_root, id, 1 << 22, .spilled, shared);
        try learn.sweep();
        return .{ .shared = shared, .learn = learn, .spill = spill };
    }

    /// One kind alone, its blocks under `root`, as a load with one flag opens it.
    fn alone(io: std.Io, root: std.Io.Dir, id: u64, kind: learned.Kind) !struct { shared: *learned.Blocks, store: Store } {
        const shared: *learned.Blocks = try .open(testing.allocator, io, root, id, .{ .learned = kind == .learned, .spilled = kind == .spilled });
        errdefer shared.close();
        return .{ .shared = shared, .store = try .openKind(testing.allocator, io, root, id, 1 << 22, kind, shared) };
    }

    fn close(self: *Both) void {
        self.learn.close();
        self.spill.close();
        self.shared.close();
    }
};

const shared_fx = struct {
    const block = @sizeOf(@import("../../src/decode/learned_blocks.zig").BlockHeader) + 256 * 3 + 64 * 2;

    /// A harness of 700 tokens and a conversation of 900 over its first 600: blocks 0 and 1 are both's.
    fn tokens(harness: *[700]i32, talk: *[900]i32) void {
        _ = tokensOf(harness, 1);
        _ = tokensOf(talk, 1);
        for (talk[600..]) |*token| token.* +%= 5;
    }
};

test "one block store: a block a learned and a spilt state both name is on disk once, under the learned root, each kind's file under its own" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var learn_root = try tmp.dir.createDirPathOpen(io, "learned", .{ .open_options = .{ .iterate = true } });
    defer learn_root.close(io);
    var spill_root = try tmp.dir.createDirPathOpen(io, "spilled", .{ .open_options = .{ .iterate = true } });
    defer spill_root.close(io);
    var harness: [700]i32 = undefined;
    var talk: [900]i32 = undefined;
    shared_fx.tokens(&harness, &talk);
    {
        var both: Both = try .open(io, learn_root, spill_root, 21);
        defer both.close();
        try Rowed.put(&both.learn, &harness);
        const alone = try bytesUnder(io, both.learn.dir);
        try Rowed.put(&both.spill, &talk);
        try Rowed.expectHeld(&both.learn, &harness);
        try Rowed.expectHeld(&both.spill, &talk);
        // the conversation wrote its third block beside the harness's two, and its own file under its root
        try testing.expectEqual(3, try filesEnding(io, both.learn.dir, ".blk"));
        try testing.expectEqual(1, try filesEnding(io, both.learn.dir, ".st"));
        try testing.expectEqual(0, try filesEnding(io, both.spill.dir, ".blk"));
        try testing.expectEqual(1, try filesEnding(io, both.spill.dir, ".sst"));
        const talk_file = learned.prefixBytes(900, 3) + (899 - 768) * 3 + (224 - 192) * 2 + 50;
        try testing.expectEqual(talk_file, try bytesUnder(io, both.spill.dir));
        // the learned root: the third block more, and the spilt root's path, for a load that opens the
        // learned kind alone
        try testing.expectEqual(alone + shared_fx.block, try bytesUnder(io, both.learn.dir));
        try testing.expectEqual(1, try filesEnding(io, both.learn.dir, learned.roots_file));
    }
    // read back by the next load, each kind from its own root
    var both: Both = try .open(io, learn_root, spill_root, 21);
    defer both.close();
    try testing.expectEqual(1, both.learn.held.items.len);
    try testing.expectEqual(1, both.spill.held.items.len);
    try Rowed.expectHeld(&both.learn, &harness);
    try Rowed.expectHeld(&both.spill, &talk);
    // a kind's files are not the other's: the same tokens under each kind are two states, one set of blocks
    try Rowed.put(&both.spill, &harness);
    try testing.expectEqual(3, try filesEnding(io, both.learn.dir, ".blk"));
    try testing.expectEqual(2, both.spill.held.items.len);
}

test "one block store: a cap's eviction of either kind leaves the blocks the other's states name, whose share then rises" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var learn_root = try tmp.dir.createDirPathOpen(io, "learned", .{ .open_options = .{ .iterate = true } });
    defer learn_root.close(io);
    var spill_root = try tmp.dir.createDirPathOpen(io, "spilled", .{ .open_options = .{ .iterate = true } });
    defer spill_root.close(io);
    var harness: [700]i32 = undefined;
    var talk: [900]i32 = undefined;
    shared_fx.tokens(&harness, &talk);
    var both: Both = try .open(io, learn_root, spill_root, 22);
    defer both.close();
    try Rowed.put(&both.learn, &harness);
    try Rowed.put(&both.spill, &talk);
    const harness_file = learned.prefixBytes(700, 3) + (699 - 512) * 3 + (174 - 128) * 2 + 50;
    const talk_file = learned.prefixBytes(900, 3) + (899 - 768) * 3 + (224 - 192) * 2 + 50;
    // each kind counts half of the two blocks both name
    try testing.expectEqual(harness_file + shared_fx.block, try both.learn.holding());
    try testing.expectEqual(talk_file + 2 * shared_fx.block, try both.spill.holding());

    both.learn.cap = 0;
    try both.learn.sweep();
    try testing.expectEqual(0, both.learn.held.items.len);
    try testing.expectEqual(3, try filesEnding(io, both.learn.dir, ".blk"));
    try Rowed.expectHeld(&both.spill, &talk);
    try testing.expectEqual(talk_file + 3 * shared_fx.block, try both.spill.holding());

    // the reverse: the spilt state goes, the harness's two blocks stay and its own third goes
    both.learn.cap = 1 << 22;
    try Rowed.put(&both.learn, &harness);
    both.spill.cap = 0;
    try both.spill.sweep();
    try testing.expectEqual(0, both.spill.held.items.len);
    try testing.expectEqual(2, try filesEnding(io, both.learn.dir, ".blk"));
    try testing.expectEqual(0, try filesEnding(io, both.spill.dir, ".sst"));
    try Rowed.expectHeld(&both.learn, &harness);
    try testing.expectEqual(harness_file + 2 * shared_fx.block, try both.learn.holding());
}

test "one block store: a load that opens one kind alone deletes no block the other's states name, under two roots or one" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var learn_root = try tmp.dir.createDirPathOpen(io, "learned", .{ .open_options = .{ .iterate = true } });
    defer learn_root.close(io);
    var spill_root = try tmp.dir.createDirPathOpen(io, "spilled", .{ .open_options = .{ .iterate = true } });
    defer spill_root.close(io);
    var harness: [700]i32 = undefined;
    var talk: [900]i32 = undefined;
    shared_fx.tokens(&harness, &talk);
    {
        var both: Both = try .open(io, learn_root, spill_root, 23);
        defer both.close();
        try Rowed.put(&both.learn, &harness);
        try Rowed.put(&both.spill, &talk);
    }
    {
        // `--learn` alone: the spilt states under the root the learned one lists keep their blocks, through
        // a discard and a cap of nothing; a store that only reads deletes nothing either
        var one = try Both.alone(io, learn_root, 23, .learned);
        defer one.shared.close();
        defer one.store.close();
        try Rowed.expectHeld(&one.store, &harness);
        try testing.expectEqual(3, try filesEnding(io, one.store.dir, ".blk"));
        one.store.discard(&harness);
        one.store.cap = 0;
        try one.store.sweep();
        try testing.expectEqual(3, try filesEnding(io, one.store.dir, ".blk"));
    }
    {
        var both: Both = try .open(io, learn_root, spill_root, 23);
        defer both.close();
        try Rowed.expectHeld(&both.spill, &talk);
        try testing.expectEqual(0, both.learn.held.items.len);
    }

    // one root for both: a kind alone reads the other's files there as names, and leaves them
    var root = try tmp.dir.createDirPathOpen(io, "one", .{ .open_options = .{ .iterate = true } });
    defer root.close(io);
    {
        var both: Both = try .open(io, root, root, 24);
        defer both.close();
        try Rowed.put(&both.learn, &harness);
        try Rowed.put(&both.spill, &talk);
        try testing.expectEqual(3, try filesEnding(io, both.learn.dir, ".blk"));
        try testing.expectEqual(0, try filesEnding(io, both.learn.dir, learned.roots_file));
    }
    {
        var one = try Both.alone(io, root, 24, .spilled);
        defer one.shared.close();
        defer one.store.close();
        one.store.cap = 0;
        try one.store.sweep();
        try testing.expectEqual(0, one.store.held.items.len);
        // the conversation's own third block went with it; the harness's two and its file stay
        try testing.expectEqual(2, try filesEnding(io, one.store.dir, ".blk"));
        try testing.expectEqual(1, try filesEnding(io, one.store.dir, ".st"));
    }
    var one = try Both.alone(io, root, 24, .learned);
    defer one.shared.close();
    defer one.store.close();
    try Rowed.expectHeld(&one.store, &harness);
}

test "one block store: what the form before wrote is a miss: a state of the old magic, and blocks a kind's other root holds" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var harness: [700]i32 = undefined;
    var talk: [900]i32 = undefined;
    shared_fx.tokens(&harness, &talk);
    {
        var store: Store = try .open(testing.allocator, io, tmp.dir, 25, 1 << 22);
        defer store.close();
        try Rowed.put(&store, &harness);
        const file = try store.dir.openFile(io, &learned.nameOf(learned.keyOf(&harness)), .{ .mode = .read_write });
        defer file.close(io);
        try file.writePositionalAll(io, "zslearn\x02", 0);
    }
    {
        var store: Store = try .open(testing.allocator, io, tmp.dir, 25, 1 << 22);
        defer store.close();
        try testing.expectEqual(0, store.held.items.len);
        try testing.expect(try store.find(testing.allocator, &harness) == null);
        try testing.expectEqual(0, try bytesUnder(io, store.dir));
    }

    // a spilt root that held its own blocks (`--spill-gib` alone), opened beside a learned root: its states
    // are not found where the blocks now lie, and they and their blocks go
    var learn_root = try tmp.dir.createDirPathOpen(io, "learned", .{ .open_options = .{ .iterate = true } });
    defer learn_root.close(io);
    var spill_root = try tmp.dir.createDirPathOpen(io, "spilled", .{ .open_options = .{ .iterate = true } });
    defer spill_root.close(io);
    {
        var one = try Both.alone(io, spill_root, 26, .spilled);
        defer one.shared.close();
        defer one.store.close();
        try Rowed.put(&one.store, &talk);
        try testing.expectEqual(3, try filesEnding(io, one.store.dir, ".blk"));
    }
    var both: Both = try .open(io, learn_root, spill_root, 26);
    defer both.close();
    try testing.expectEqual(0, both.spill.held.items.len);
    try testing.expect(try both.spill.find(testing.allocator, &talk) == null);
    try testing.expectEqual(0, try bytesUnder(io, both.spill.dir));
}

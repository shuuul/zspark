//! `contract.zig`'s tests.

const std = @import("std");
const testing = std.testing;
const contract = @import("../../src/decode/contract.zig");
const draft = @import("../../src/decode/draft.zig");

const whole = struct {
    pub const Model = struct { max_rows: usize };
    pub const State = struct { pos: usize };
    pub const Buffers = struct {
        rows: usize,
        pub fn scoredRows(self: @This()) usize {
            return self.rows;
        }
    };
    pub const Error = error{Failed};
    pub fn forward(_: Model, _: *State, _: *Buffers, _: []const i32) Error!void {}
    pub fn streams(_: Buffers, _: usize) void {}
};

const owed = [_]contract.Owed{
    .{ .name = "Model", .is = .{ .type = .{ .fields = &.{"max_rows"} } } },
    .{ .name = "State", .is = .{ .type = .{ .fields = &.{"pos"} } } },
    .{ .name = "Buffers", .is = .{ .type = .{ .fields = &.{"rows"}, .decls = &.{"scoredRows"} } } },
    .{ .name = "Error", .is = .{ .type = .{} } },
    .{ .name = "forward", .is = .{ .function = 4 } },
    .{ .name = "streams", .is = .{ .function = 2 } },
};

test "a namespace that declares all it owes lacks nothing, and one that does not is told each thing" {
    try testing.expectEqualStrings("", comptime contract.lacking(whole, &owed));

    const partial = struct {
        pub const Model = struct { rows: usize };
        pub const State = u32;
        pub const Buffers = struct { rows: usize };
        pub const Error = 3;
        pub fn forward(_: Model, _: *State) void {}
        pub const streams = "none";
    };
    try testing.expectEqualStrings(
        \\
        \\  Model: no field `max_rows`
        \\  State: no field `pos`
        \\  Buffers: no declaration `scoredRows`
        \\  Error: not a type
        \\  forward: a function of 2 parameters, not of 4
        \\  streams: not a function
    , comptime contract.lacking(partial, &owed));
    try testing.expectEqualStrings("\n  Model: not declared", comptime contract.lacking(struct {}, owed[0..1]));
}

test "what the decoder asks is what its doc comment lists" {
    // every name the comment gives as `- `NAME`` is owed, and nothing else
    const names = [_][]const u8{ "Model", "State", "Buffers", "Graphs", "Error", "forward", "forwardScoring", "commit", "logitsRow", "streams" };
    try testing.expectEqual(names.len, draft.owed.len);
    for (names, draft.owed) |name, one| try testing.expectEqualStrings(name, one.name);
}

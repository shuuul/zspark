//! What `structured_test.zig`, `served_test.zig` and the server's other fixtures share of `structured.zig`: a
//! compiler on the tiny tokenizer (`Bench`; spec 121 moved it out of `zig/src/server/structured.zig`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const json_schema = @import("../../src/text/json_schema.zig");
const structured = @import("../../src/server/structured.zig");
const tokenizer_file = @import("../../src/text/tokenizer.zig");

/// A compiler on the tiny tokenizer, for `structured.zig`'s tests and the server's.
pub const Bench = struct {
    tokenizer: tokenizer_file.Tokenizer,
    engine: json_schema.Engine,
    stops: [1]u32,

    pub fn init(self: *Bench, gpa: Allocator) !void {
        var diagnostic: tokenizer_file.Diagnostic = .{};
        self.tokenizer = try .parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &diagnostic);
        errdefer self.tokenizer.deinit(gpa);
        self.stops = .{self.tokenizer.addedId("<|end|>").?};
        self.engine = try .init(gpa, &self.tokenizer, self.tokenizer.count() + 29, &self.stops);
    }

    pub fn deinit(self: *Bench, gpa: Allocator) void {
        self.engine.deinit(gpa);
        self.tokenizer.deinit(gpa);
    }

    pub fn compiler(self: *Bench) structured.Compiler {
        return .{ .engine = self.engine.interface(), .tokenizer = &self.tokenizer, .stops = &self.stops };
    }
};

//! Capture caches shared by graph sites. Each site keeps its staging, keys, preparation and storage;
//! `Cache` owns the graphs and either owns its stream or borrows one that outlives it. `Graphs` adapts the
//! ranks' forwards, whose all-gathers are captured and replayed in step.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");

/// The cache owns every graph it captures. Borrowed streams must stay at the same address until deinit;
/// graphs must be destroyed before any storage their launches point into.
pub fn Cache(comptime Key: type) type {
    return struct {
        const Self = @This();

        source: union(enum) { owned: cuda.Stream, borrowed: *cuda.Stream },
        captured: std.AutoHashMapUnmanaged(Key, cuda.Graph) = .empty,
        gpa: Allocator,

        pub fn init(gpa: Allocator) cuda.Error!Self {
            return .{ .source = .{ .owned = try .init() }, .gpa = gpa };
        }

        pub fn initBorrowed(gpa: Allocator, on: *cuda.Stream) Self {
            return .{ .source = .{ .borrowed = on }, .gpa = gpa };
        }

        pub fn stream(self: *Self) *cuda.Stream {
            return switch (self.source) {
                .owned => |*on| on,
                .borrowed => |on| on,
            };
        }

        pub fn capturing(self: *const Self) u64 {
            return switch (self.source) {
                .owned => |on| on.capturing,
                .borrowed => |on| on.capturing,
            };
        }

        pub fn count(self: *const Self) usize {
            return self.captured.count();
        }

        pub fn contains(self: *const Self, key: Key) bool {
            return self.captured.contains(key);
        }

        pub fn deinit(self: *Self) void {
            var graphs = self.captured.valueIterator();
            while (graphs.next()) |graph| graph.deinit();
            self.captured.deinit(self.gpa);
            switch (self.source) {
                .owned => |*on| on.deinit(),
                .borrowed => {},
            }
            self.* = undefined;
        }

        /// Records `compute(args)` only for a missing key, without launching it. Any failed capture is
        /// ended and dropped, and leaves no entry; the stream's own capture accounting remains authoritative.
        pub fn capture(self: *Self, key: Key, comptime compute: anytype, args: anytype) (@typeInfo(@TypeOf(@call(.auto, compute, args))).error_union.error_set || cuda.Error || Allocator.Error)!*cuda.Graph {
            const entry = try self.captured.getOrPut(self.gpa, key);
            if (entry.found_existing) return entry.value_ptr;
            errdefer self.captured.removeByPtr(entry.key_ptr);
            const on = self.stream();
            try on.begin();
            @call(.auto, compute, args) catch |err| {
                if (on.end()) |graph| {
                    var unused = graph;
                    unused.deinit();
                } else |_| {}
                return err;
            };
            entry.value_ptr.* = try on.end();
            return entry.value_ptr;
        }

        pub fn run(self: *Self, key: Key, comptime compute: anytype, args: anytype) (@typeInfo(@TypeOf(@call(.auto, compute, args))).error_union.error_set || cuda.Error || Allocator.Error)!void {
            const graph = try self.capture(key, compute, args);
            return graph.launch(self.stream().*);
        }

        pub fn launch(self: *Self, key: Key) cuda.Error!void {
            return self.captured.get(key).?.launch(self.stream().*);
        }
    };
}

/// `W` supplies `Model`, `State`, `Buffers`, `Error`, and:
/// - `enter(Model, *State, *Buffers, tokens) Error!Key`: tells and stages before returning the key;
/// - `compute(Model, *State, *Buffers, rows, Key) Error!void`: launches only;
/// - `ahead(State, Buffers, rows) ?Key`: predicts a window's key, null where no graph can be held.
/// `capture` asks `compute` alone of them.
pub fn Graphs(comptime Key: type, comptime W: type) type {
    return struct {
        const Self = @This();

        cache: Cache(Key),

        pub fn init(gpa: Allocator) cuda.Error!Self {
            return .{ .cache = try .init(gpa) };
        }

        pub fn deinit(self: *Self) void {
            self.cache.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, m: W.Model, state: *W.State, b: *W.Buffers, tokens: []const i32) (W.Error || Allocator.Error)!void {
            const key = try W.enter(m, state, b, tokens);
            return self.cache.run(key, W.compute, .{ m, state, b, tokens.len, key });
        }

        pub fn capture(self: *Self, m: W.Model, state: *W.State, b: *W.Buffers, rows: usize, key: Key) (W.Error || Allocator.Error)!*cuda.Graph {
            return self.cache.capture(key, W.compute, .{ m, state, b, rows, key });
        }

        pub fn holds(self: *const Self, state: W.State, b: W.Buffers, rows: usize) bool {
            const key = W.ahead(state, b, rows) orelse return false;
            return self.cache.contains(key);
        }
    };
}

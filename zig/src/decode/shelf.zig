//! A sequence's states held to return to, and the buffers of those that went, lent to the next taken (spec
//! 112 WS-02): the bookkeeping both models' sessions share, so a state dropped leaves its device buffers to
//! the next one and a reply allocates none once states are held. Host only: what a state is, how it is
//! taken into buffers or restored, and which one goes (`marks.thinned`) are the model's.
//!
//! A state is a struct with `pos` and `deinit`, or an array of them, a rank's each, in rank order and at one
//! position (Flash Next's two ranks in one process).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Frees `one`, each rank's where it is an array.
fn free(comptime T: type, one: *T) void {
    switch (@typeInfo(T)) {
        .array => for (one) |*part| part.deinit(),
        else => one.deinit(),
    }
}

/// The position of `one`, rank 0's where it is an array.
fn posOf(comptime T: type, one: T) usize {
    return switch (@typeInfo(T)) {
        .array => one[0].pos,
        else => one.pos,
    };
}

/// Buffers of states that went, the latest given lent first.
pub fn Spares(comptime T: type) type {
    return struct {
        const Self = @This();

        list: std.ArrayList(T) = .empty,

        pub fn deinit(self: *Self, gpa: Allocator) void {
            self.trim(0);
            self.list.deinit(gpa);
        }

        /// `one` kept for its buffers; freed where the list cannot grow.
        pub fn give(self: *Self, gpa: Allocator, one: T) void {
            self.list.append(gpa, one) catch {
                var gone = one;
                free(T, &gone);
            };
        }

        /// Room for one more `give` that cannot fail to keep it.
        pub fn reserve(self: *Self, gpa: Allocator) Allocator.Error!void {
            try self.list.ensureUnusedCapacity(gpa, 1);
        }

        /// One to write over, or null where none waits.
        pub fn lend(self: *Self) ?T {
            return self.list.pop();
        }

        pub fn count(self: *const Self) usize {
            return self.list.items.len;
        }

        /// Frees all but the `most` given first.
        pub fn trim(self: *Self, most: usize) void {
            while (self.list.items.len > most) {
                var gone = self.list.pop().?;
                free(T, &gone);
            }
        }

        /// The bytes they hold, by `T.bytes`.
        pub fn bytes(self: *const Self) usize {
            var total: usize = 0;
            for (self.list.items) |one| total += one.bytes();
            return total;
        }
    };
}

/// The states of one sequence, ascending by position, and the buffers of those dropped (`spare`). `Spare` is
/// what a dropped state leaves: the state whole where it is `Kept`, else what `Kept.release(*Kept, Allocator,
/// *Spares(Spare))` gives it (GLM's marks leave each rank's KDA buffers and free the drafter's context).
pub fn Shelf(comptime Kept: type, comptime Spare: type) type {
    return struct {
        const Self = @This();

        gpa: Allocator,
        held: std.ArrayList(Kept) = .empty,
        spare: Spares(Spare) = .{},

        pub fn deinit(self: *Self) void {
            for (self.held.items) |*one| free(Kept, one);
            self.held.deinit(self.gpa);
            self.spare.deinit(self.gpa);
        }

        /// The index of the state at `pos`, or null where none is held there.
        pub fn find(self: *const Self, pos: usize) ?usize {
            for (self.held.items, 0..) |one, i| if (posOf(Kept, one) == pos) return i;
            return null;
        }

        /// Whether every state held stands below `pos`: one taken there keeps them ascending.
        pub fn above(self: *const Self, pos: usize) bool {
            const newest = self.held.getLastOrNull() orelse return true;
            return posOf(Kept, newest) < pos;
        }

        /// The held states' positions into `out`, which has room for them; the slice written.
        pub fn positions(self: *const Self, out: []usize) []usize {
            for (self.held.items, out[0..self.held.items.len]) |one, *pos| pos.* = posOf(Kept, one);
            return out[0..self.held.items.len];
        }

        /// The buffers of a state that went, for the caller to take one into.
        pub fn lend(self: *Self) ?Spare {
            return self.spare.lend();
        }

        /// The state at index `at` goes, its buffers to `spare`.
        pub fn drop(self: *Self, at: usize) void {
            var gone = self.held.orderedRemove(at);
            self.release(&gone);
        }

        /// The state at index `at` is `one` instead, the one it was going as `drop`'s.
        pub fn replace(self: *Self, at: usize, one: Kept) void {
            self.release(&self.held.items[at]);
            self.held.items[at] = one;
        }

        /// The states above `pos` go.
        pub fn dropAbove(self: *Self, pos: usize) void {
            var at: usize = 0;
            while (at < self.held.items.len) {
                if (posOf(Kept, self.held.items[at]) > pos) self.drop(at) else at += 1;
            }
        }

        /// Every state goes, the newest first.
        pub fn clear(self: *Self) void {
            while (self.held.pop()) |taken| {
                var gone = taken;
                self.release(&gone);
            }
        }

        fn release(self: *Self, one: *Kept) void {
            if (Kept == Spare) self.spare.give(self.gpa, one.*) else one.release(self.gpa, &self.spare);
        }
    };
}

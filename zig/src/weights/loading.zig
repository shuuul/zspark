//! What every model's load shares (the GB10's load pipeline, `.agents/skills/cuda-gb10/references/loading.md`):
//! where a load's time goes with `ZSPARK_LOAD_PHASES` set, the loading thread's figures beside the reader's
//! (`safetensors.Stage.clock`, `safetensors.Held.clock`) and the device's (`cuda.phases`).

const std = @import("std");
const cuda = @import("../cuda/cuda.zig");
const units = @import("../cuda/units.zig");
const safetensors = @import("safetensors.zig");

const seconds = units.seconds;
const gib = units.gib;

/// The loading thread's time waiting for a part's bytes, its own reads included.
pub var waited_ns: u64 = 0;

/// The host's own steps a loader times (each summed over a load, on the loading thread): csf scales decoded,
/// experts packed into the kernels' blocks.
pub const host = struct {
    pub var csf_ns: u64 = 0;
    pub var pack_ns: u64 = 0;
};

/// A clock's start where `ZSPARK_LOAD_PHASES` is set, else 0: `since` of it adds nothing.
pub fn now() u64 {
    return if (cuda.phases.on) cuda.monotonic() else 0;
}

/// Adds the time since `start` (`now`) to `into`.
pub fn since(into: *u64, start: u64) void {
    if (cuda.phases.on) into.* += cuda.monotonic() - start;
}

/// With `ZSPARK_LOAD_PHASES` set, a line to stderr of where a load of `total_ns` went. The reader's: the direct
/// reads (into `Stage`'s buffers, or over the mapping, `Held`), the buffers' growth and the mappings put over
/// the ranges and back. The loading thread's: waiting for a part's bytes, `cudaMalloc`, waiting for the device's
/// launches before a copy, the copies to the device, `cudaFree`, the host steps timed (`host`), and the rest,
/// its own work and what is not timed.
pub fn report(total_ns: u64) void {
    if (!cuda.phases.on) return;
    const stage = safetensors.Stage.clock;
    const held = safetensors.Held.clock;
    const p = cuda.phases;
    const bytes = stage.bytes + held.bytes;
    const read_ns = stage.read_ns + held.read_ns;
    // `Held`'s reads and mappings are the loading thread's own
    const timed = held.read_ns + held.map_ns + held.release_ns + waited_ns + p.malloc_ns + p.wait_ns + p.copy_ns + p.free_ns + host.csf_ns + host.pack_ns;
    std.debug.print("zspark-load total {d:.2} s; reader: read {d:.2} GiB, {d:.2} s ({d:.2} GiB/s), buffers {d:.2} s, mapped back {d:.2} s; " ++
        "loader: waited for reads {d:.2} s, cudaMalloc {d} in {d:.2} s, device wait {d:.2} s, copy {d:.2} GiB, {d:.2} s ({d:.2} GiB/s), cudaFree {d:.2} s, " ++
        "csf {d:.2} s, pack {d:.2} s, host {d:.2} s\n", .{
        seconds(total_ns),        gib(bytes),                 seconds(read_ns),                               gib(bytes) / @max(seconds(read_ns), 1e-9), seconds(stage.map_ns + held.map_ns),
        seconds(held.release_ns), seconds(waited_ns),         p.mallocs,                                      seconds(p.malloc_ns),                      seconds(p.wait_ns),
        gib(p.copied),            seconds(p.copy_ns),         gib(p.copied) / @max(seconds(p.copy_ns), 1e-9), seconds(p.free_ns),                        seconds(host.csf_ns),
        seconds(host.pack_ns),    seconds(total_ns -| timed),
    });
}

/// One of `Ahead`'s two buffers: a part's ranges as read, and where each tensor's staged bytes are.
pub const Slot = struct {
    stage: safetensors.Stage = .{},
    /// A loader's second buffer, where it reads ranges for another than itself (GLM's rank 0 under
    /// `--weights-from-peer`: rank 1's, sent from here, and found here where they are its own too).
    peer: safetensors.Stage = .{},
    /// A tensor's address in the mapping -> how far its staged bytes are from it.
    shifts: std.AutoHashMapUnmanaged(usize, isize) = .empty,
    /// The part it holds, `Ahead`'s index of it.
    part: ?usize = null,
    failed: ?anyerror = null,
    /// Recorded after the launches that read the part it held (the experts the device places from it), which
    /// it waits for before it is written over.
    used: ?cuda.Event = null,

    pub fn deinit(self: *Slot, gpa: std.mem.Allocator) void {
        self.stage.deinit(gpa);
        self.peer.deinit(gpa);
        self.shifts.deinit(gpa);
        if (self.used) |*event| event.deinit();
    }

    /// Where each of `list`, the ranges of `tensors` (`tensors[i]` the whole tensor `list[i]` is of), is staged:
    /// `moved` then finds a tensor's bytes by its address in the mapping. A loader reads of a staged tensor
    /// only the range it staged: the rest of the slice `moved` gives is not its bytes.
    pub fn shift(self: *Slot, gpa: std.mem.Allocator, list: []const []const u8, tensors: []const []const u8) std.mem.Allocator.Error!void {
        try self.shifts.ensureTotalCapacity(gpa, @intCast(list.len));
        for (list, tensors) |range, tensor| {
            const by = self.stage.shift(range) orelse self.peer.shift(range) orelse continue;
            self.shifts.putAssumeCapacity(@intFromPtr(tensor.ptr), by);
        }
    }

    /// `data`, a whole tensor's bytes in the mapping, where this slot staged them; as it is where it did not.
    /// One relocation of a whole tensor: a rank's strided source runs are `runs.Packed`, not this slice.
    pub fn moved(self: *const Slot, data: []const u8) []const u8 {
        const by = self.shifts.get(@intFromPtr(data.ptr)) orelse return data;
        const at: usize = @bitCast(@as(isize, @bitCast(@intFromPtr(data.ptr))) + by);
        return @as([*]const u8, @ptrFromInt(at))[0..data.len];
    }
};

/// A load's parts in order, each read while the one before is packed and uploaded: a thread reads the next part
/// into the slot the part before it was packed from, so that the disk is off the path of the host's packing and
/// the device's allocations (TensorFold reads two layers ahead, `cuda/direct_read.py`). Two parts are held at
/// most. `Loader` gives `fill(*const Loader, *Slot, part) anyerror!void`, which reads `part`'s ranges into the
/// slot (`Stage.read`, then `Slot.shift`), and `after(*const Loader, part) ?usize`, the part read next; and may
/// give `forget(*const Loader, *Slot)`, called on each slot before it is freed. The loader is passed to each
/// call, so that it may hold its `Ahead` by value.
pub fn Ahead(comptime Loader: type) type {
    return struct {
        const Self = @This();

        slots: [2]Slot = .{ .{}, .{} },
        /// The slot of the part being packed (or packed last); the reader fills the other.
        current: ?*Slot = null,
        reader: ?std.Thread = null,

        pub fn deinit(self: *Self, gpa: std.mem.Allocator, loader: *const Loader) void {
            self.join();
            // a load that failed may have left launches that read the slots
            cuda.synchronize() catch {};
            for (&self.slots) |*slot| {
                if (@hasDecl(Loader, "forget")) loader.forget(slot);
                slot.deinit(gpa);
            }
            self.* = undefined;
        }

        pub fn join(self: *Self) void {
            if (self.reader) |thread| thread.join();
            self.reader = null;
        }

        fn fill(slot: *Slot, loader: *const Loader, part: usize) void {
            slot.part = part;
            slot.failed = null;
            slot.shifts.clearRetainingCapacity();
            if (slot.used) |event| event.synchronize() catch {
                slot.failed = error.CudaFailure;
                return;
            };
            loader.fill(slot, part) catch |err| {
                slot.failed = err;
            };
        }

        /// `part`'s slot, read (or waited for), and the next part's read begun on the reader's thread. The slot's
        /// bytes stay until two parts on.
        pub fn begin(self: *Self, loader: *const Loader, part: usize) anyerror!*Slot {
            const waiting = cuda.monotonic();
            self.join();
            const next: *Slot = if (self.current == &self.slots[0]) &self.slots[1] else &self.slots[0];
            if (next.part != part) fill(next, loader, part);
            if (next.failed) |err| return err;
            self.current = next;
            waited_ns += cuda.monotonic() - waiting;
            if (loader.after(part)) |ahead| {
                const other: *Slot = if (next == &self.slots[0]) &self.slots[1] else &self.slots[0];
                // `prefetch` read it already
                if (other.part != ahead) {
                    other.part = null;
                    // where no thread is to be had, `begin` reads the next part itself
                    self.reader = std.Thread.spawn(.{}, fill, .{ other, loader, ahead }) catch null;
                }
            }
            return next;
        }

        /// Marks the launches that read the current part's slot; the reader waits for them before it fills the
        /// slot again.
        pub fn end(self: *Self) void {
            const slot = self.current orelse return;
            if (slot.used == null) slot.used = cuda.Event.init() catch null;
            if (slot.used) |event| event.record() catch {
                slot.used.?.deinit();
                slot.used = null;
            };
            // no event to wait on: the device is waited for here
            if (slot.used == null) cuda.synchronize() catch {};
        }

        /// Reads `first` and the part after it on a thread of its own before a load asks for them (while the
        /// ranks join).
        pub fn prefetch(self: *Self, loader: *const Loader, first: usize) void {
            std.debug.assert(self.current == null and self.reader == null);
            self.reader = std.Thread.spawn(.{}, readFirst, .{ self, loader, first }) catch null;
        }

        fn readFirst(self: *Self, loader: *const Loader, first: usize) void {
            fill(&self.slots[0], loader, first);
            if (self.slots[0].failed != null) return;
            if (loader.after(first)) |ahead| fill(&self.slots[1], loader, ahead);
        }
    };
}

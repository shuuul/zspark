//! A rank's share of a whole pack onto the device, a part at a time (`model.Model.loadEach`): a reader thread
//! reads the next part's tensors into one of two buffers (`safetensors.Stage`) while the loading thread packs and
//! uploads this one from the other, which the loaders find through `View`, and the host copies a share makes of
//! them (`share.take`: half of the columns, half of a trellis' second axis) live in an arena kept from one part
//! to the next. On the Spark the host's memory is the device's: a share's copies kept to the end would be 50 GiB
//! beside its 82 on the device.

const std = @import("std");
const units = @import("../../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const Config = @import("../layers/config.zig").Config;
const layout = @import("../layers/layout.zig");
const model = @import("../layers/model.zig");
const mtp = @import("../mtp/mtp.zig");
const experts = @import("../../../weights/experts.zig");
const loading = @import("../../../weights/loading.zig");
const nccl = @import("../../../ranks/nccl.zig");

pub const LoadError = model.LoadError;

pub const Loaded = struct {
    m: model.Model,
    mtp: mtp.Weights,

    pub fn deinit(self: *Loaded) void {
        self.mtp.deinit();
        self.m.deinit();
        self.* = undefined;
    }
};

/// What `rank` reads of `entries`, a part's tensors (`Each.sort`): its half of what is split by rows, the whole
/// of the rest; `tensors` the tensor each range is of.
fn ranges(gpa: Allocator, entries: []const Each.Entry, rank: usize, halved: bool, list: *std.ArrayList([]const u8), tensors: *std.ArrayList([]const u8)) Allocator.Error!void {
    for (entries) |entry| {
        const range = rangeOf(entry.name, entry.data, rank, halved) orelse continue;
        try list.append(gpa, range);
        try tensors.append(gpa, entry.data);
    }
}

fn sent(err: anyerror, part: model.Model.Part) LoadError {
    std.log.err("rank 1's share of {any} does not cross from rank 0: {t}", .{ part, err });
    return error.CudaFailure;
}

/// `part`'s ranges read into `slot` (`Each.fill`): rank 0's under `Each.wire` with rank 1's in its peer buffer, sent
/// from there; rank 1's under it received; otherwise read from the disk.
fn fillOr(slot: *loading.Slot, each: *const Each, part: model.Model.Part) LoadError!void {
    const gpa = each.gpa;
    var clock = loading.now();
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(gpa);
    var tensors: std.ArrayList([]const u8) = .empty;
    defer tensors.deinit(gpa);
    const entries = each.parts[Each.index(part)].items;
    // the MTP layer's experts are read on the host, down whole
    const halved = switch (part) {
        .top => false,
        .layer => |index| each.swapping and index < each.c.layers,
    };
    try ranges(gpa, entries, each.rank, halved, &list, &tensors);
    lap(&Sending.ranges_ns, &clock);
    const wire = each.wire orelse {
        try slot.stage.read(gpa, each.checkpoint, list.items);
        return slot.shift(gpa, list.items, tensors.items);
    };
    if (each.rank != 0) {
        // rank 0 reads them and sends them, laid out as here
        const bytes = try slot.stage.lay(gpa, each.checkpoint, list.items, true) orelse return error.OutOfMemory;
        wire.receive(wire.context, bytes) catch |err| return sent(err, part);
        return slot.shift(gpa, list.items, tensors.items);
    }
    // rank 1's ranges first, sent while this rank's are read; what this rank reads of the part that rank 1's
    // hold already (a tensor both read whole) is found there, not read again
    var theirs: std.ArrayList([]const u8) = .empty;
    defer theirs.deinit(gpa);
    var their_tensors: std.ArrayList([]const u8) = .empty;
    defer their_tensors.deinit(gpa);
    try ranges(gpa, entries, 1, halved, &theirs, &their_tensors);
    lap(&Sending.ranges_ns, &clock);
    // what was sent from the buffer is sent before it is written over, or freed
    wire.flush(wire.context, slot.peer.buffer.ptr) catch |err| return sent(err, part);
    lap(&Sending.flush_ns, &clock);
    const bytes = try slot.peer.lay(gpa, each.checkpoint, theirs.items, true) orelse return error.OutOfMemory;
    if (!slot.peer.fetch()) {
        std.log.err("rank 1's ranges of {any} are not read from the disk", .{part});
        return error.Unsupported;
    }
    lap(&Sending.theirs_ns, &clock);
    wire.send(wire.context, bytes) catch |err| return sent(err, part);
    lap(&Sending.send_ns, &clock);
    var rest: std.ArrayList([]const u8) = .empty;
    defer rest.deinit(gpa);
    for (list.items) |range| if (slot.peer.moved(range) == null) try rest.append(gpa, range);
    try slot.stage.read(gpa, each.checkpoint, rest.items);
    lap(&Sending.own_ns, &clock);
    try slot.shift(gpa, list.items, tensors.items);
    lap(&Sending.shift_ns, &clock);
}

fn lap(into: *u64, clock: *u64) void {
    loading.since(into, clock.*);
    clock.* = loading.now();
}

/// Rank 0's reader under `Each.wire`, summed over a load (`ZSPARK_LOAD_PHASES`): finding the parts' ranges,
/// waiting for a slot's last send, reading rank 1's ranges, beginning their send, reading its own, placing them.
pub const Sending = struct {
    pub var ranges_ns: u64 = 0;
    pub var flush_ns: u64 = 0;
    pub var theirs_ns: u64 = 0;
    pub var send_ns: u64 = 0;
    pub var own_ns: u64 = 0;
    pub var shift_ns: u64 = 0;

    pub fn report() void {
        if (!cuda.phases.on or theirs_ns == 0) return;
        const s = units.seconds;
        std.debug.print("zspark-load rank 0's reader: ranges {d:.2} s, flush {d:.2} s, rank 1's reads {d:.2} s, send {d:.2} s, its own reads {d:.2} s, shift {d:.2} s\n", .{ s(ranges_ns), s(flush_ns), s(theirs_ns), s(send_ns), s(own_ns), s(shift_ns) });
    }
};

/// The checkpoint as the loaders read it (anything with `get(name)`): a tensor of the part being packed at its
/// staged bytes, any other through the mapping. A loader reads of a tensor only the range `ranges` names for its
/// rank: the rest of a staged tensor's slice is not its bytes.
pub const View = struct {
    checkpoint: *const safetensors.Checkpoint,
    each: *const Each,

    /// A part's bytes stay until the launches recorded at its `Each.end` have run: the device may read them
    /// (`mlp.Moe.placesOn`).
    pub const staged = true;

    /// Where the ranks trade halves of down (`Each.swapping`), the trade.
    pub fn swap(self: View) ?experts.Swap {
        if (!self.each.swapping) return null;
        const comm = self.each.comm.?;
        return .{ .rank = comm.rank, .context = comm, .trade = nccl.Comm.trade };
    }

    pub fn get(self: View, name: []const u8) ?safetensors.Tensor {
        var tensor = self.checkpoint.get(name) orelse return null;
        const slot = self.each.ahead.current orelse return tensor;
        tensor.data = slot.moved(tensor.data);
        return tensor;
    }
};

/// What `rank` reads of the tensor `name` whose bytes are `data`, or null: none of it. `halved`: a routed expert's
/// down projection by its half of the outputs, as `experts.Swap` trades it, and not whole.
pub fn rangeOf(name: []const u8, data: []const u8, rank: usize, halved: bool) ?[]const u8 {
    // a ModelOpt linear's csf streams are read whole, its window cut as they decode (`modelopt.cut`)
    if (std.mem.indexOf(u8, name, ".weight_scale.csf32_") != null) return data;
    // no rank keeps an activation's scale, but an NVFP4 expert's is read as its rows are quantized
    if (std.mem.endsWith(u8, name, ".input_scale")) return data;
    if (halved and std.mem.indexOf(u8, name, ".mlp.experts.") != null and std.mem.endsWith(u8, name, ".down_proj.weight"))
        return data[rank * (data.len / layout.world) ..][0 .. data.len / layout.world];
    return switch (layout.rule(name) orelse return null) {
        .drop => null,
        .row, .vocab => data[rank * (data.len / layout.world) ..][0 .. data.len / layout.world],
        .whole, .col => data,
    };
}

/// The parts `model.Model.loadSpan` asks for in turn, each read while the one before is packed and uploaded: a
/// thread reads the next part into the slot the part before it was packed from, so that the disk is off the path
/// of the host's packing and the device's allocations (TensorFold reads two layers ahead,
/// `cuda/direct_read.py`).
pub const Each = struct {
    gpa: Allocator,
    checkpoint: *const safetensors.Checkpoint,
    c: *const Config,
    rank: usize,
    /// The layers asked for, then the MTP layer's where `with_mtp`.
    first: usize,
    count: usize,
    with_mtp: bool,
    arena: std.heap.ArenaAllocator,
    /// Each rank reads half of a routed layer's down projections and trades the other half's inputs with its
    /// peer over `comm` (`experts.Swap`), which is set before the first such layer is loaded.
    swapping: bool = false,
    comm: ?*const nccl.Comm = null,
    /// `serve --weights-from-peer`: rank 0 reads rank 1's ranges of each part beside its own and sends them over
    /// this, rank 1 receives them, laid out alike, and opens no file (`safetensors.File.fromHead`). Both walk the
    /// parts in the same order; set before the first is read (`over`).
    wire: ?Wire = null,
    /// The parts' two buffers and the reader thread; a part's index is `index`'s.
    ahead: loading.Ahead(Each) = .{},
    /// Each part's tensors by `index`, in the checkpoint's order (`sort`): found once, not by a walk of every
    /// tensor's name a part (0.5 s of rank 0's reader under `wire`).
    parts: []std.ArrayList(Entry) = &.{},

    pub const Entry = struct { name: []const u8, data: []const u8 };

    // an MXFP8 head's scales beside its codes (FP4-Local's)
    const top = [_][]const u8{ layout.prefix ++ "embed_tokens.weight", layout.prefix ++ "norm.weight", "lm_head.weight", "lm_head.weight_scale" };

    pub fn deinit(self: *Each) void {
        self.ahead.deinit(self.gpa, self);
        for (self.parts) |*one| one.deinit(self.gpa);
        self.gpa.free(self.parts);
        self.arena.deinit();
    }

    /// `parts`, once, before the first part is read: the top's tensors (`top`) and each layer's (under
    /// `model.layers.N.`), the MTP layer's among them.
    pub fn sort(self: *Each) Allocator.Error!void {
        if (self.parts.len > 0) return;
        const parts = try self.gpa.alloc(std.ArrayList(Entry), self.c.layers + 2);
        @memset(parts, .empty);
        errdefer {
            for (parts) |*one| one.deinit(self.gpa);
            self.gpa.free(parts);
        }
        const under = layout.prefix ++ "layers.";
        var it = self.checkpoint.tensors.iterator();
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            const at: usize = if (std.mem.startsWith(u8, name, under)) at: {
                const rest = name[under.len..];
                const dot = std.mem.indexOfScalar(u8, rest, '.') orelse continue;
                const layer = std.fmt.parseInt(usize, rest[0..dot], 10) catch continue;
                if (layer > self.c.layers) continue;
                break :at index(.{ .layer = layer });
            } else for (top) |one| {
                if (std.mem.eql(u8, name, one)) break 0;
            } else continue;
            try parts[at].append(self.gpa, .{ .name = name, .data = entry.value_ptr.data });
        }
        self.parts = parts;
    }

    /// Under `wire`, what was sent from a slot is sent before it is freed.
    pub fn forget(self: *const Each, slot: *loading.Slot) void {
        if (self.wire) |wire| wire.flush(wire.context, slot.peer.buffer.ptr) catch {};
    }

    /// `Ahead`'s index of `part`: the top first, then the layers by their number (the MTP layer's last).
    pub fn index(part: model.Model.Part) usize {
        return switch (part) {
            .top => 0,
            .layer => |layer| layer + 1,
        };
    }

    fn partOf(at: usize) model.Model.Part {
        return if (at == 0) .top else .{ .layer = at - 1 };
    }

    pub fn fill(self: *const Each, slot: *loading.Slot, at: usize) anyerror!void {
        return fillOr(slot, self, partOf(at));
    }

    pub fn after(self: *const Each, at: usize) ?usize {
        const layer = switch (partOf(at)) {
            .top => return index(.{ .layer = self.first }),
            .layer => |layer| layer,
        };
        if (layer + 1 < self.first + self.count) return index(.{ .layer = layer + 1 });
        if (layer + 1 == self.first + self.count and self.with_mtp) return index(.{ .layer = self.c.layers });
        return null;
    }

    pub fn view(self: *const Each) View {
        return .{ .checkpoint = self.checkpoint, .each = self };
    }

    pub fn begin(self: *Each, part: model.Model.Part) LoadError!Allocator {
        // where `prefetch` did not
        try self.sort();
        // its pages kept for the next part, not faulted in again
        _ = self.arena.reset(.retain_capacity);
        _ = self.ahead.begin(self, index(part)) catch |err| return @errorCast(err);
        return self.arena.allocator();
    }

    /// The parts of `count` layers from `first` (and the MTP layer's, `with_mtp`) for `rank`.
    pub fn init(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, rank: usize, first: usize, count: usize, with_mtp: bool) Each {
        return .{ .gpa = gpa, .checkpoint = checkpoint, .c = c, .rank = rank, .first = first, .count = count, .with_mtp = with_mtp, .arena = .init(gpa) };
    }

    /// The share comes over `wire` (rank 1) or rank 1's goes over it (rank 0); the buffers the bytes cross from
    /// and into are made known to the device, which sends and receives them.
    pub fn over(self: *Each, wire: Wire) void {
        std.debug.assert(self.ahead.current == null and self.ahead.reader == null);
        self.wire = wire;
        const pin: safetensors.Stage.Pin = .{ .register = cuda.hostRegister, .unregister = cuda.hostUnregister };
        for (&self.ahead.slots) |*slot| {
            if (self.rank == 0) slot.peer.pin = pin else slot.stage.pin = pin;
        }
    }

    /// Reads the first two parts on a thread of its own before a load asks for them: while the ranks join.
    pub fn prefetch(self: *Each) Allocator.Error!void {
        try self.sort();
        self.ahead.prefetch(self, index(.top));
    }

    /// Marks the launches that read `part`'s slot; the reader waits for them before it fills the slot again.
    pub fn end(self: *Each, _: model.Model.Part) void {
        self.ahead.end();
    }
};

/// How rank 1's share crosses from rank 0 (`Each.wire`): an opaque peer with three steps. NCCL's is `Line`, a
/// test's a queue.
pub const Wire = struct {
    context: *anyopaque,
    /// Begins to send `bytes` (rank 0), which stay as they are until `flush`.
    send: *const fn (*anyopaque, []const u8) anyerror!void,
    /// Returns once what was sent from the buffer at `at` is sent; at once where nothing was.
    flush: *const fn (*anyopaque, at: [*]const u8) anyerror!void,
    /// Fills `into` with the next bytes sent (rank 1), before it returns.
    receive: *const fn (*anyopaque, []u8) anyerror!void,
};

/// `Wire` over NCCL: a communicator split from the ranks' (`open`), so that the reader's transfers keep their
/// order beside the loader's trades of down on the ranks' own, enqueued on a stream of its own. The reader may
/// begin before the ranks have joined (`Each.prefetch`): a transfer waits for `open`, or fails after `leave`.
pub const Line = struct {
    comm: nccl.Comm = undefined,
    state: std.atomic.Value(State) = .init(.waiting),
    side: cuda.Side,
    /// The last sends: where each was from, and an event recorded after it.
    sends: [2]Sent,
    next: usize = 0,

    const State = enum(u8) { waiting, open, gone };
    const Sent = struct { at: usize = 0, event: cuda.Event };

    pub fn init() cuda.Error!Line {
        var side: cuda.Side = try .init();
        errdefer side.deinit();
        var first: cuda.Event = try .init();
        errdefer first.deinit();
        return .{ .side = side, .sends = .{ .{ .event = first }, .{ .event = try .init() } } };
    }

    pub fn deinit(self: *Line) void {
        if (self.state.load(.acquire) == .open) self.comm.deinit();
        for (&self.sends) |*one| one.event.deinit();
        self.side.deinit();
        self.* = undefined;
    }

    /// Once the ranks have joined over `ranks`, on both.
    pub fn open(self: *Line, ranks: *const nccl.Comm) nccl.Error!void {
        self.comm = try ranks.split();
        self.state.store(.open, .release);
    }

    /// No `open` comes: a transfer waiting for it fails.
    pub fn leave(self: *Line) void {
        _ = self.state.cmpxchgStrong(.waiting, .gone, .acq_rel, .acquire);
    }

    pub fn wire(self: *Line) Wire {
        return .{ .context = self, .send = send, .flush = flush, .receive = receive };
    }

    fn opened(self: *Line) error{PeerGone}!*const nccl.Comm {
        while (true) switch (self.state.load(.acquire)) {
            .open => return &self.comm,
            .gone => return error.PeerGone,
            .waiting => {
                const millisecond: std.os.linux.timespec = .{ .sec = 0, .nsec = std.time.ns_per_ms };
                _ = std.os.linux.nanosleep(&millisecond, null);
            },
        };
    }

    fn move(self: *Line, one: nccl.Comm.Move) anyerror!void {
        const comm = try self.opened();
        try comm.exchangeOn(self.side.handle, .u8, &.{one});
    }

    fn send(context: *anyopaque, bytes: []const u8) anyerror!void {
        const self: *Line = @ptrCast(@alignCast(context));
        // a buffer's send is waited for before the buffer is filled again: two parts on
        const slot = &self.sends[self.next];
        if (slot.at != 0) try slot.event.synchronize();
        slot.at = 0;
        try self.move(.{ .peer = 1, .send = bytes.ptr, .send_count = bytes.len, .recv = null, .recv_count = 0 });
        try slot.event.recordOn(self.side.handle);
        slot.at = @intFromPtr(bytes.ptr);
        self.next = 1 - self.next;
    }

    fn flush(context: *anyopaque, at: [*]const u8) anyerror!void {
        const self: *Line = @ptrCast(@alignCast(context));
        for (&self.sends) |*slot| if (slot.at == @intFromPtr(at)) {
            try slot.event.synchronize();
            slot.at = 0;
        };
    }

    fn receive(context: *anyopaque, into: []u8) anyerror!void {
        const self: *Line = @ptrCast(@alignCast(context));
        try self.move(.{ .peer = 0, .send = null, .send_count = 0, .recv = into.ptr, .recv_count = into.len });
        const slot = &self.sends[0];
        try slot.event.recordOn(self.side.handle);
        try slot.event.synchronize();
    }
};

pub fn load(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, layers: usize, rank: usize, rows: cuda.Scratch.Rows) LoadError!Loaded {
    const got = try loadSpan(gpa, checkpoint, c, 0, layers, rank, rows, true);
    return .{ .m = got.m, .mtp = got.mtp.? };
}

/// A span of the model's layers, with the MTP layer where asked.
pub const Span = struct { m: model.Model, mtp: ?mtp.Weights };

/// `load` of the `count` layers from `first` alone (`model.Model.loadSpan`), with the embedding, the norm and
/// the head, and the MTP layer `with_mtp`.
pub fn loadSpan(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, c: *const Config, first: usize, count: usize, rank: usize, rows: cuda.Scratch.Rows, with_mtp: bool) LoadError!Span {
    std.debug.assert(count >= 1 and first + count <= c.layers);
    var each: Each = .init(gpa, checkpoint, c, rank, first, count, with_mtp);
    // the reader may be reading the next part of one that failed
    defer each.deinit();
    return loadFrom(gpa, &each, rows);
}

/// `loadSpan` of `each`'s parts, which `each.prefetch` may have begun to read; the caller frees `each` after.
pub fn loadFrom(gpa: Allocator, each: *Each, rows: cuda.Scratch.Rows) LoadError!Span {
    const c = each.c;
    if (each.swapping and (each.comm == null or each.comm.?.world != layout.world or each.comm.?.rank != each.rank)) return error.Unsupported;
    var m = try model.Model.loadSpan(gpa, each.view(), c, each.first, each.count, each.rank, rows, each);
    errdefer m.deinit();
    if (!each.with_mtp) return .{ .m = m, .mtp = null };
    const arena = try each.begin(.{ .layer = c.layers });
    const weights = try mtp.Weights.load(arena, gpa, each.view(), c, each.rank, rows);
    each.end(.{ .layer = c.layers });
    return .{ .m = m, .mtp = weights };
}

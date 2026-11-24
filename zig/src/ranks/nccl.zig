//! NCCL through its C API (spec 045 WS-01): what two ranks need of it, declared here and linked only when the
//! build names it (`zig build -Dnccl=PREFIX`; `linked`). Without it every call is `error.NcclNotLinked`, so a
//! build for one device needs no NCCL. Written against NCCL 2.30.7's `nccl.h`.

const std = @import("std");
const cuda = @import("../cuda/cuda.zig");

/// Whether this build links NCCL.
pub const linked = @import("linked").nccl;

pub const Error = error{ NcclNotLinked, NcclFailure };

/// What rank 0 makes and every rank joins with: 128 bytes, sent as bytes.
pub const UniqueId = extern struct { internal: [128]u8 };

/// `ncclDataType_t`'s values for the types zspark gathers.
const DataType = enum(c_int) { u8 = 1, i32 = 2, u32 = 3, i64 = 4, f16 = 6, f32 = 7, bf16 = 9 };

extern fn ncclGetVersion(version: *c_int) c_int;
extern fn ncclGetUniqueId(id: *UniqueId) c_int;
extern fn ncclCommInitRank(comm: *?*anyopaque, ranks: c_int, id: UniqueId, rank: c_int) c_int;
extern fn ncclCommDestroy(comm: ?*anyopaque) c_int;
extern fn ncclCommSplit(comm: ?*anyopaque, color: c_int, key: c_int, newcomm: *?*anyopaque, config: ?*anyopaque) c_int;
extern fn ncclAllGather(send: ?*const anyopaque, recv: ?*anyopaque, count: usize, dtype: c_int, comm: ?*anyopaque, stream: ?*anyopaque) c_int;
extern fn ncclSend(send: ?*const anyopaque, count: usize, dtype: c_int, peer: c_int, comm: ?*anyopaque, stream: ?*anyopaque) c_int;
extern fn ncclRecv(recv: ?*anyopaque, count: usize, dtype: c_int, peer: c_int, comm: ?*anyopaque, stream: ?*anyopaque) c_int;
extern fn ncclGroupStart() c_int;
extern fn ncclGroupEnd() c_int;
extern fn ncclGetErrorString(code: c_int) [*:0]const u8;

fn check(code: c_int) Error!void {
    if (code == 0) return;
    std.log.err("NCCL: {s}", .{ncclGetErrorString(code)});
    return error.NcclFailure;
}

fn dataType(dtype: cuda.DType) DataType {
    return switch (dtype) {
        .u8, .e4m3 => .u8,
        .i32 => .i32,
        .u32 => .u32,
        .i64 => .i64,
        .f16, .u16 => .f16,
        .f32 => .f32,
        .bf16 => .bf16,
    };
}

/// The library's version as it encodes it: 23007 is 2.30.7.
pub fn version() Error!u32 {
    if (!linked) return error.NcclNotLinked;
    var out: c_int = 0;
    try check(ncclGetVersion(&out));
    return @intCast(out);
}

/// A new communicator's id: rank 0 makes it and gives it to the others.
pub fn uniqueId() Error!UniqueId {
    if (!linked) return error.NcclNotLinked;
    var id: UniqueId = undefined;
    try check(ncclGetUniqueId(&id));
    return id;
}

pub const Comm = struct {
    handle: *anyopaque,
    world: usize,
    rank: usize,

    /// Joins the `world` ranks of `id` as `rank`; returns when every rank has.
    pub fn init(world: usize, id: UniqueId, rank: usize) Error!Comm {
        if (!linked) return error.NcclNotLinked;
        var handle: ?*anyopaque = null;
        try check(ncclCommInitRank(&handle, @intCast(world), id, @intCast(rank)));
        return .{ .handle = handle.?, .world = world, .rank = rank };
    }

    /// A communicator of the same ranks beside this one: what is enqueued on it keeps its own order, whatever
    /// another thread enqueues on this one meanwhile. Every rank calls it, and it returns when every rank has.
    pub fn split(self: Comm) Error!Comm {
        if (!linked) return error.NcclNotLinked;
        var handle: ?*anyopaque = null;
        try check(ncclCommSplit(self.handle, 0, @intCast(self.rank), &handle, null));
        return .{ .handle = handle orelse return error.NcclFailure, .world = self.world, .rank = self.rank };
    }

    pub fn deinit(self: *Comm) void {
        if (linked) _ = ncclCommDestroy(self.handle);
        self.* = undefined;
    }

    /// Every rank's `send` into `recv`, rank 0's first: `recv` holds `world` times `send`'s values. Enqueued
    /// on this thread's stream (the one a capture records, else the default), as a kernel's launch is.
    pub fn allGather(self: Comm, send: cuda.View, recv: cuda.View) Error!void {
        if (!linked) return error.NcclNotLinked;
        std.debug.assert(recv.dtype == send.dtype and recv.count() == self.world * send.count());
        try check(ncclAllGather(send.ptr, recv.ptr, send.count(), @backingInt(dataType(send.dtype)), self.handle, cuda.currentStream()));
    }

    /// One rank's sends and receives of rows (spec 106 WS-02): `count` values of `dtype` from `send` to each
    /// `Move`'s peer and into its `recv`, all in one NCCL group, enqueued as `allGather` is. A move of no value
    /// is left out.
    pub fn exchange(self: Comm, dtype: cuda.DType, moves: []const Move) Error!void {
        return self.exchangeOn(cuda.currentStream(), dtype, moves);
    }

    /// `exchange` enqueued on `stream` (a `cuda.Side`'s).
    pub fn exchangeOn(self: Comm, stream: ?*anyopaque, dtype: cuda.DType, moves: []const Move) Error!void {
        if (!linked) return error.NcclNotLinked;
        const code = @backingInt(dataType(dtype));
        try check(ncclGroupStart());
        for (moves) |one| {
            std.debug.assert(one.peer != self.rank and one.peer < self.world);
            const peer: c_int = @intCast(one.peer);
            if (one.send_count > 0) check(ncclSend(one.send, one.send_count, code, peer, self.handle, stream)) catch |err| {
                _ = ncclGroupEnd();
                return err;
            };
            if (one.recv_count > 0) check(ncclRecv(one.recv, one.recv_count, code, peer, self.handle, stream)) catch |err| {
                _ = ncclGroupEnd();
                return err;
            };
        }
        try check(ncclGroupEnd());
    }

    /// A load's trade of halves with the other of two ranks (`weights/experts.zig`'s `Swap.trade`, a model's
    /// down projections at load): `send` to it and its `recv` from it, bytes as long, in the stream's order.
    /// `context` is the `*const Comm`.
    pub fn trade(context: *const anyopaque, send: cuda.View, recv: cuda.View) cuda.Error!void {
        const comm: *const Comm = @ptrCast(@alignCast(context));
        comm.exchange(.u8, &.{.{ .peer = 1 - comm.rank, .send = send.ptr, .send_count = send.count(), .recv = recv.ptr, .recv_count = recv.count() }}) catch |err| {
            std.log.err("the ranks' halves do not cross: {t}", .{err});
            return error.CudaFailure;
        };
    }

    /// What `exchange` sends to one peer and receives from it.
    pub const Move = struct {
        peer: usize,
        send: ?*const anyopaque,
        send_count: usize,
        recv: ?*anyopaque,
        recv_count: usize,
    };
};

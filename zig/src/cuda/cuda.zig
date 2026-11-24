//! The CUDA runtime and the nvcc-built launchers in kernels/. `View`, `DType` and `DeviceInfo` mirror
//! zig/kernels/zs/abi.h and zig/kernels/zs/probe.h.

const std = @import("std");
const block_file = @import("block.zig");

pub const Error = error{CudaFailure};

pub const DType = enum(i32) {
    u8 = 0,
    i32 = 1,
    f32 = 2,
    bf16 = 3,
    f16 = 4,
    e4m3 = 5,
    i64 = 6,
    u16 = 7,
    u32 = 8,

    pub fn size(self: DType) usize {
        return switch (self) {
            .u8, .e4m3 => 1,
            .bf16, .f16, .u16 => 2,
            .i32, .f32, .u32 => 4,
            .i64 => 8,
        };
    }
};

/// A device tensor as the launchers read it. Only the rows of a 2-D view may be strided.
pub const View = extern struct {
    ptr: ?*anyopaque = null,
    size: [6]i64 = @splat(0),
    /// Elements from one row to the next; 0: contiguous.
    row_stride: i64 = 0,
    ndim: i32 = 0,
    dtype: DType = .u8,

    pub fn init(ptr: *anyopaque, dtype: DType, shape: []const usize) View {
        var view: View = .{ .ptr = ptr, .dtype = dtype, .ndim = @intCast(shape.len) };
        for (shape, 0..) |dim, i| view.size[i] = @intCast(dim);
        return view;
    }

    pub fn count(self: View) usize {
        var n: usize = 1;
        for (self.size[0..@intCast(self.ndim)]) |dim| n *= @intCast(dim);
        return n;
    }

    /// Rows [first, first + count) of the view, its row stride kept; past its rows it panics in every build, as
    /// `Buffer.span` does.
    pub fn span(self: View, first: usize, n: usize) View {
        const held: usize = @intCast(self.size[0]);
        if (first > held or n > held - first) std.debug.panic("cuda.View.span: rows {d} to {d} of a view of {d}", .{ first, first + n, held });
        var inner: usize = 1;
        for (self.size[1..@intCast(self.ndim)]) |dim| inner *= @intCast(dim);
        const row: usize = if (self.row_stride != 0) @intCast(self.row_stride) else inner;
        var view = self;
        view.ptr = @ptrCast(@as([*]u8, @ptrCast(self.ptr.?)) + first * row * self.dtype.size());
        view.size[0] = @intCast(n);
        return view;
    }
};

const success = 0;

extern fn cudaMalloc(ptr: *?*anyopaque, size: usize) c_int;
extern fn cudaFree(ptr: ?*anyopaque) c_int;
extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int) c_int;
extern fn cudaMemset(ptr: ?*anyopaque, value: c_int, count: usize) c_int;
extern fn cudaDeviceSynchronize() c_int;
extern fn cudaStreamSynchronize(stream: ?*anyopaque) c_int;
extern fn cudaGetErrorString(code: c_int) [*:0]const u8;

const host_to_device = 1;
const device_to_host = 2;
const device_to_device = 3;

fn check(code: c_int) Error!void {
    if (code == success) return;
    std.log.err("CUDA: {s}", .{cudaGetErrorString(code)});
    return error.CudaFailure;
}

/// Waits for every launch on the device and surfaces a kernel's asynchronous failure.
pub fn synchronize() Error!void {
    return check(cudaDeviceSynchronize());
}

/// Waits for the launches on this thread's stream (`currentStream`) alone: not for a transfer on a stream of its
/// own (`Side`), which may wait for this thread's next launches on the other rank.
pub fn synchronizeStream() Error!void {
    return check(cudaStreamSynchronize(current_stream));
}

extern fn cudaStreamCreate(stream: *?*anyopaque) c_int;
extern fn cudaStreamDestroy(stream: ?*anyopaque) c_int;
extern fn cudaStreamBeginCapture(stream: ?*anyopaque, mode: c_int) c_int;
extern fn cudaStreamEndCapture(stream: ?*anyopaque, graph: *?*anyopaque) c_int;
extern fn cudaGraphInstantiate(exec: *?*anyopaque, graph: ?*anyopaque, flags: c_ulonglong) c_int;
extern fn cudaGraphLaunch(exec: ?*anyopaque, stream: ?*anyopaque) c_int;
extern fn cudaGraphExecDestroy(exec: ?*anyopaque) c_int;
extern fn cudaGraphDestroy(graph: ?*anyopaque) c_int;

/// cudaStreamCaptureModeThreadLocal: only this thread's calls are held to a capture's rules.
const capture_thread_local = 1;

/// A stream of its own, which graphs are captured on and launched into. Everything else runs on the default
/// stream, which waits for this one and which this one waits for, so the two stay in program order.
pub const Stream = struct {
    handle: *anyopaque,
    /// Nanoseconds spent capturing: from each `begin` to the end of its `end`, the graph's instantiation included.
    capturing: u64 = 0,
    began: u64 = 0,

    pub fn init() Error!Stream {
        var handle: ?*anyopaque = null;
        try check(cudaStreamCreate(&handle));
        return .{ .handle = handle.? };
    }

    pub fn deinit(self: *Stream) void {
        _ = cudaStreamDestroy(self.handle);
        self.* = undefined;
    }

    /// From here to `capture`'s end this thread's launches are recorded, not run. A launcher that copies to or
    /// from the host, or waits, fails the capture.
    pub fn begin(self: *Stream) Error!void {
        self.began = monotonic();
        try check(cudaStreamBeginCapture(self.handle, capture_thread_local));
        zs_set_stream(self.handle);
        current_stream = self.handle;
    }

    /// The launches since `begin` as a graph. Call it after a failed launch too, to leave the capture.
    pub fn end(self: *Stream) Error!Graph {
        defer self.capturing += monotonic() - self.began;
        zs_set_stream(null);
        current_stream = null;
        var recorded: ?*anyopaque = null;
        try check(cudaStreamEndCapture(self.handle, &recorded));
        defer _ = cudaGraphDestroy(recorded);
        var exec: ?*anyopaque = null;
        try check(cudaGraphInstantiate(&exec, recorded, 0));
        return .{ .exec = exec.? };
    }
};

/// A point in the stream's order the host can wait for alone, where a synchronize waits for every launch.
pub const Event = struct {
    handle: *anyopaque,

    pub fn init() Error!Event {
        var handle: ?*anyopaque = null;
        // no timing: a record is then the cheapest marker
        try check(cudaEventCreateWithFlags(&handle, 0x2));
        return .{ .handle = handle.? };
    }

    pub fn deinit(self: *Event) void {
        _ = cudaEventDestroy(self.handle);
        self.* = undefined;
    }

    /// Marks the stream's order here: the launches before it.
    pub fn record(self: Event) Error!void {
        return check(cudaEventRecord(self.handle, current_stream));
    }

    /// Marks `stream`'s order here (a `Side`'s), where `record` marks this thread's.
    pub fn recordOn(self: Event, stream: *anyopaque) Error!void {
        return check(cudaEventRecord(self.handle, stream));
    }

    /// This thread's launches from here on wait for the launches before the last record, on whatever stream.
    pub fn join(self: Event) Error!void {
        return check(cudaStreamWaitEvent(current_stream, self.handle, 0));
    }

    /// Returns once the launches before the last `record` have run.
    pub fn synchronize(self: Event) Error!void {
        return check(cudaEventSynchronize(self.handle));
    }
};

/// A stream beside this thread's (spec 116 WS-03): non-blocking, so the legacy default stream neither waits for
/// it nor it for that one, and at the device's highest priority, so its kernels (an exchange's) take SMs as they
/// free. It keeps order with the rest only through events: `follow` before its work, an event recorded on it
/// after (`Event.recordOn`) and `Event.join` before its result is read. Inside a capture the same events fork
/// and join the graph.
pub const Side = struct {
    handle: *anyopaque,
    forked: Event,

    pub fn init() Error!Side {
        var low: c_int = 0;
        var high: c_int = 0;
        try check(cudaDeviceGetStreamPriorityRange(&low, &high));
        var handle: ?*anyopaque = null;
        // cudaStreamNonBlocking
        try check(cudaStreamCreateWithPriority(&handle, 0x1, high));
        errdefer _ = cudaStreamDestroy(handle);
        return .{ .handle = handle.?, .forked = try .init() };
    }

    pub fn deinit(self: *Side) void {
        self.forked.deinit();
        _ = cudaStreamDestroy(self.handle);
        self.* = undefined;
    }

    /// The side's next launches wait for this thread's launches so far.
    pub fn follow(self: Side) Error!void {
        try self.forked.record();
        return check(cudaStreamWaitEvent(self.handle, self.forked.handle, 0));
    }
};

extern fn cudaStreamWaitEvent(stream: ?*anyopaque, event: ?*anyopaque, flags: c_uint) c_int;
extern fn cudaStreamCreateWithPriority(stream: *?*anyopaque, flags: c_uint, priority: c_int) c_int;
extern fn cudaDeviceGetStreamPriorityRange(least: *c_int, greatest: *c_int) c_int;

/// Page-locked host memory: what an asynchronous copy from the device lands in (`Buffer.downloadLaunched`).
pub const HostBuffer = struct {
    bytes: []u8,

    pub fn init(size: usize) Error!HostBuffer {
        var ptr: ?*anyopaque = null;
        try check(cudaMallocHost(&ptr, size));
        const at: [*]u8 = @ptrCast(ptr.?);
        return .{ .bytes = at[0..size] };
    }

    pub fn deinit(self: *HostBuffer) void {
        _ = cudaFreeHost(self.bytes.ptr);
        self.* = undefined;
    }
};

/// What every synchronous copy from the device lands in first (spec 132): this thread's page-locked memory, at
/// most `most` bytes, copied on into the caller's by the CPU. On the Spark a `cudaMemcpy` from the device
/// straight into pageable memory (the host allocator's, freshly mapped) sometimes misses it, with no CUDA error:
/// some of its pages keep what they held, and the bytes land on other pages of the process (an allocator's
/// metadata, another buffer). The rank rounds' test met it in one run of six, spec 105's `ranks_hall` once.
/// The CPU's copy costs 10 us more for 296 KiB (22.8 against 12.5) and 2.5 to 3 times a direct copy's time from
/// 4 MiB up.
pub const staging = struct {
    pub const most: usize = 4 << 20;
    threadlocal var held: ?HostBuffer = null;

    fn take(len: usize) Error![]u8 {
        if (held) |one| if (one.bytes.len >= len) return one.bytes;
        if (held) |*one| one.deinit();
        held = null;
        held = try .init(@min(most, std.math.ceilPowerOfTwoAssert(usize, @max(len, 64 << 10))));
        return held.?.bytes;
    }
};

/// `dest.len` bytes of device memory from `source` into host memory, after pending launches finish, through
/// `staging`, a piece of at most `staging.most` bytes at a time.
fn copyOut(dest: []u8, source: *const anyopaque) Error!void {
    const from: [*]const u8 = @ptrCast(source);
    var done: usize = 0;
    while (done < dest.len) {
        const n = @min(dest.len - done, staging.most);
        const bounce = try staging.take(n);
        try check(cudaMemcpy(bounce.ptr, from + done, n, device_to_host));
        @memcpy(dest[done..][0..n], bounce[0..n]);
        done += n;
    }
}

/// Makes `bytes`, host memory of the process's own, known to the device as page-locked (`cudaHostRegister`):
/// what NCCL sends from and receives into beside device buffers. False where it cannot be.
pub fn hostRegister(bytes: []u8) bool {
    return cudaHostRegister(bytes.ptr, bytes.len, 0) == 0;
}

pub fn hostUnregister(bytes: []u8) void {
    _ = cudaHostUnregister(bytes.ptr);
}

extern fn cudaHostRegister(ptr: ?*anyopaque, size: usize, flags: c_uint) c_int;
extern fn cudaHostUnregister(ptr: ?*anyopaque) c_int;
extern fn cudaEventCreateWithFlags(event: *?*anyopaque, flags: c_uint) c_int;
extern fn cudaEventDestroy(event: ?*anyopaque) c_int;
extern fn cudaEventRecord(event: ?*anyopaque, stream: ?*anyopaque) c_int;
extern fn cudaEventSynchronize(event: ?*anyopaque) c_int;
extern fn cudaMallocHost(ptr: *?*anyopaque, size: usize) c_int;
extern fn cudaFreeHost(ptr: ?*anyopaque) c_int;

pub fn monotonic() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// Captured launches with their arguments as they were: the same kernels on the same buffers, in one call.
pub const Graph = struct {
    exec: *anyopaque,

    /// Graphs launched by this process, and the nanoseconds the host spent in the call: it returns once the
    /// graph's nodes are enqueued, about 0.1 us a node (0.5 under a node trace), and the device has nothing to run
    /// until then (specs 028, 089).
    pub var launches: u64 = 0;
    pub var launching: u64 = 0;
    /// Of those, each graph's first launch, which also uploads it (spec 063): the first `seen.len` graphs'.
    pub var first_launches: u64 = 0;
    pub var first_launching: u64 = 0;
    var seen: [1024]?*anyopaque = @splat(null);

    pub fn launch(self: Graph, stream: Stream) Error!void {
        const before = monotonic();
        defer {
            const took = monotonic() - before;
            launching += took;
            launches += 1;
            for (&seen) |*one| {
                if (one.* == self.exec) break;
                if (one.* == null) {
                    one.* = self.exec;
                    first_launches += 1;
                    first_launching += took;
                    break;
                }
            }
        }
        return check(cudaGraphLaunch(self.exec, stream.handle));
    }

    pub fn deinit(self: *Graph) void {
        _ = cudaGraphExecDestroy(self.exec);
        self.* = undefined;
    }
};

extern fn cudaMemGetInfo(free: *usize, total: *usize) c_int;

/// What the device says is free and what it has, in bytes. On the Spark the device's memory is the host's, so
/// `free` moves with the page cache and with every process, not with this program alone.
pub fn memory() Error!struct { free: usize, total: usize } {
    var free: usize = 0;
    var total: usize = 0;
    try check(cudaMemGetInfo(&free, &total));
    return .{ .free = free, .total = total };
}

/// The bytes the device can still give: the system's estimate where memory is unified (the device's own
/// count leaves out what the page cache would give back), else the device's.
pub fn available(io: std.Io) usize {
    const device = if (memory()) |held| held.free else |_| 0;
    return @max(device, systemAvailable(io) orelse 0);
}

/// `/proc/meminfo`'s `MemAvailable` in bytes, or null where it cannot be read. Read as a stream: the file
/// says its size is 0, and a read that asks the size first (`Dir.readFileAlloc`) gives nothing (spec 082: the
/// loads' checks had the device's count alone until then).
pub fn systemAvailable(io: std.Io) ?usize {
    var file = std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{}) catch return null;
    defer file.close(io);
    var text: [4096]u8 = undefined;
    var reader = file.readerStreaming(io, &.{});
    const len = reader.interface.readSliceShort(&text) catch return null;
    var lines = std.mem.tokenizeScalar(u8, text[0..len], '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var words = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const kib = std.fmt.parseInt(usize, words.next() orelse return null, 10) catch return null;
        return kib * 1024;
    }
    return null;
}

/// What a load leaves of `available` when it sizes what several replies hold (GLM's slots, Flash Next's
/// pool): the runtime's own, a round's rounding.
pub const reserve: usize = 2 << 30;

/// Where a load's time goes, with `ZSPARK_LOAD_PHASES` set (`main` reads it): the host's time in `cudaMalloc`,
/// in synchronous copies to the device, and waiting for launches before them (each copy first waits for the
/// thread's stream, as the copy does, so that a copy's time is the copy's). The reads' and the mappings' time is `safetensors.Held.clock`'s.
/// Written by whichever thread allocates, which is one at a time; off, nothing is timed and nothing waits.
pub const phases = struct {
    pub var on: bool = false;
    pub var malloc_ns: u64 = 0;
    pub var mallocs: u64 = 0;
    pub var free_ns: u64 = 0;
    pub var copy_ns: u64 = 0;
    pub var copied: u64 = 0;
    pub var wait_ns: u64 = 0;
    /// When `main` began, for `mark`.
    pub var started: u64 = 0;

    /// A line to stderr of the seconds since `main` began, at `what`.
    pub fn mark(what: []const u8) void {
        if (!on) return;
        std.debug.print("zspark-load at {d:.2} s: {s}\n", .{ @as(f64, @floatFromInt(monotonic() - started)) / 1e9, what });
    }

    fn malloc(ptr: *?*anyopaque, size: usize) c_int {
        if (!on) return cudaMalloc(ptr, size);
        const start = monotonic();
        defer malloc_ns += monotonic() - start;
        mallocs += 1;
        return cudaMalloc(ptr, size);
    }

    fn copy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize) c_int {
        if (!on) return cudaMemcpy(dst, src, count, host_to_device);
        const start = monotonic();
        // this thread's stream alone: a transfer on a stream of its own (`Side`, a load's `pack.Line`) runs on
        const waited = cudaStreamSynchronize(current_stream);
        const between = monotonic();
        wait_ns += between - start;
        if (waited != 0) return waited;
        defer copy_ns += monotonic() - between;
        copied += count;
        return cudaMemcpy(dst, src, count, host_to_device);
    }
};

/// The bytes asked of `cudaMalloc` through `Buffer` and not yet freed, and how many buffers hold them. Every
/// device allocation of the program is a `Buffer`; what the device holds beyond these is the runtime's (the
/// context, the kernels' code, the graphs) and the allocator's rounding of each buffer. Written by whichever
/// thread allocates, which is one at a time.
pub const account = struct {
    pub var bytes: usize = 0;
    pub var buffers: usize = 0;
    /// Buffers made since the program began, freed or not: two readings apart say whether a stretch of
    /// work allocated (a reply's rounds must not, spec 089).
    pub var made_ever: usize = 0;
    /// With `ZSPARK_ALLOCATIONS` set (`main` reads it), a line to stderr for each buffer made or freed: its
    /// address, bytes, dtype, shape and the return addresses above it, which `addr2line -i` names once the
    /// load address is taken from them (a `mark` line's `anchor` is `zs_last_error`'s). A diagnostic of where
    /// device memory goes (spec 030), read by `tools/reference/device_memory.py`.
    pub var trace: bool = false;

    /// Says what the buffers hold now and what the device has free, with `label` before it.
    pub fn mark(label: []const u8) void {
        const now = memory() catch return;
        std.debug.print("zspark-memory mark {s} buffers={d} bytes={d} device_free={d} device_total={d} anchor={x}\n", .{ label, buffers, bytes, now.free, now.total, @intFromPtr(&zs_last_error) });
    }

    noinline fn made(buffer: Buffer, size: usize) void {
        var addresses: [20]usize = undefined;
        const stack = std.debug.captureCurrentStackTrace(.{ .first_address = @returnAddress() }, &addresses);
        const view = buffer.view;
        std.debug.print("zspark-memory alloc {x} {d} {t} {any} | {any}\n", .{ @intFromPtr(view.ptr), size, view.dtype, view.size[0..@intCast(view.ndim)], stack.return_addresses });
    }
};

/// Freed buffers held back and read again: with `on`, `Buffer.deinit` fills a buffer with `pattern` and keeps it
/// here instead of freeing it, and when it leaves the ring, or at `drain`, its bytes are read back. A changed byte
/// is a launch or a copy that wrote through a pointer to freed memory. On the Spark the device's addresses are
/// the host's: a freed buffer's range can be the next host mapping, and a write through a stale pointer then
/// lands in the host's memory with no error (spec 105, the rank-1 heap panic). Debugging only:
/// `ZSPARK_QUARANTINE` (`main` reads it). The ring holds at most `most` bytes; a wider buffer, or one freed
/// during a capture, is freed at once. The first buffer written over is kept in `found`, which whoever meets
/// `error.Overwritten` prints (`{f}`): the finding goes with the error, not to the log.
pub const quarantine = struct {
    pub var on: bool = false;
    pub const pattern: u8 = 0xa5;
    pub const slots = 256;
    pub const most: usize = 1 << 30;
    pub const widest: usize = 64 << 20;
    /// Buffers read back, in all.
    pub var checked: usize = 0;
    pub const Failed = Error || error{Overwritten};

    const Held = struct { view: View, bytes: usize, freed_by: usize };

    /// A held buffer read back changed: which buffer, its first changed byte and what the device left there.
    pub const Found = struct {
        held: Held,
        at: usize,
        byte: u8,

        pub fn format(self: Found, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            const view = self.held.view;
            try writer.print("a freed buffer at {x}, {t} {any}, {d} bytes, was written after its free, from byte {d} (0x{x:0>2}); freed at 0x{x}", .{ @intFromPtr(view.ptr), view.dtype, view.size[0..@intCast(view.ndim)], self.held.bytes, self.at, self.byte, self.held.freed_by });
        }
    };
    /// The first buffer found written over since it was last cleared.
    pub var found: ?Found = null;
    var ring: [slots]Held = undefined;
    var first: usize = 0;
    var len: usize = 0;
    var bytes: usize = 0;

    /// Whether `buffer`, being freed by `freed_by`, is held; a buffer written over while held panics here.
    fn hold(buffer: Buffer, freed_by: usize) bool {
        const size = @max(buffer.bytes(), 1);
        if (!on or current_stream != null or size > widest) return false;
        if (cudaMemset(buffer.view.ptr, pattern, size) != 0) return false;
        while (len == slots or (len > 0 and bytes + size > most)) {
            leave() catch |err| switch (err) {
                error.Overwritten => std.debug.panic("cuda.quarantine: {f}", .{found.?}),
                else => std.debug.panic("cuda.quarantine: {t}", .{err}),
            };
        }
        ring[(first + len) % slots] = .{ .view = buffer.view, .bytes = size, .freed_by = freed_by };
        len += 1;
        bytes += size;
        return true;
    }

    /// The oldest held buffer read back and freed.
    fn leave() Failed!void {
        const held = ring[first];
        first = (first + 1) % slots;
        len -= 1;
        bytes -= held.bytes;
        defer Buffer.free(held.view.ptr);
        try check(cudaDeviceSynchronize());
        const host = std.heap.page_allocator.alloc(u8, held.bytes) catch return;
        defer std.heap.page_allocator.free(host);
        try copyOut(host, held.view.ptr.?);
        checked += 1;
        const at = std.mem.indexOfNone(u8, host, &.{pattern}) orelse return;
        if (found == null) found = .{ .held = held, .at = at, .byte = host[at] };
        return error.Overwritten;
    }

    /// Every held buffer read back and freed; the first written over is the error, after all are freed.
    pub fn drain() Failed!void {
        var failed: ?Failed = null;
        while (len > 0) leave() catch |err| {
            if (failed == null) failed = err;
        };
        if (failed) |err| return err;
    }
};

/// Device memory holding `shape` elements of `dtype`, freed by `deinit`.
pub const Buffer = struct {
    view: View,

    pub fn init(dtype: DType, shape: []const usize) Error!Buffer {
        var ptr: ?*anyopaque = null;
        const size = bytesOf(dtype, shape);
        try check(phases.malloc(&ptr, size));
        const self: Buffer = .{ .view = .init(ptr.?, dtype, shape) };
        account.bytes += size;
        account.buffers += 1;
        account.made_ever += 1;
        if (account.trace) account.made(self, size);
        return self;
    }

    /// The bytes `init` asks of the device, and of `account`, for `shape` elements of `dtype`, with nothing
    /// made: a buffer of no element is one byte. What a reckoning of buffers before they are made sums
    /// (`bytesFor` beside an `init`), so that it is the account's figure to the byte.
    pub fn bytesOf(dtype: DType, shape: []const usize) usize {
        var count: usize = 1;
        for (shape) |dim| count *= dim;
        return @max(count * dtype.size(), 1);
    }

    pub fn deinit(self: *Buffer) void {
        if (self.view.ptr != null) {
            account.bytes -|= @max(self.bytes(), 1);
            account.buffers -|= 1;
            if (account.trace) std.debug.print("zspark-memory free {x}\n", .{@intFromPtr(self.view.ptr)});
        }
        if (self.view.ptr != null and !quarantine.hold(self.*, @returnAddress())) free(self.view.ptr);
        self.* = undefined;
    }

    fn free(ptr: ?*anyopaque) void {
        const start = if (phases.on) monotonic() else 0;
        defer if (phases.on) {
            phases.free_ns += monotonic() - start;
        };
        _ = cudaFree(ptr);
    }

    pub fn bytes(self: Buffer) usize {
        return self.view.count() * self.view.dtype.size();
    }

    pub fn zero(self: Buffer) Error!void {
        return check(cudaMemset(self.view.ptr, 0, self.bytes()));
    }

    /// `zero` as a launch, in the stream's order. A capture records it; `zero` waits and cannot.
    pub fn zeroLaunched(self: Buffer) Error!void {
        return check(cudaMemsetAsync(self.view.ptr, 0, self.bytes(), current_stream));
    }

    /// Copies `source` in; it must cover the buffer exactly.
    pub fn upload(self: Buffer, source: []const u8) Error!void {
        std.debug.assert(source.len == self.bytes());
        return check(phases.copy(self.view.ptr, source.ptr, source.len));
    }

    /// Copies another buffer of the same size in, after pending launches on the default stream finish.
    pub fn copyFrom(self: Buffer, source: Buffer) Error!void {
        std.debug.assert(source.bytes() == self.bytes());
        return check(cudaMemcpy(self.view.ptr, source.view.ptr, self.bytes(), device_to_device));
    }

    /// `copyFrom` as a launch: in the stream's order, without waiting for the launches before it.
    pub fn copyLaunched(self: Buffer, source: Buffer) Error!void {
        std.debug.assert(source.bytes() == self.bytes());
        return check(cudaMemcpyAsync(self.view.ptr, source.view.ptr, self.bytes(), device_to_device, current_stream));
    }

    /// Copies `count` rows of `source` from its row `from` to this buffer's rows `to` on, as a launch in the
    /// stream's order; the two have rows of one size, and the ranges may not overlap.
    pub fn copyRows(self: Buffer, to: usize, source: Buffer, from: usize, count: usize) Error!void {
        const row = self.bytes() / @as(usize, @intCast(self.view.size[0]));
        const mine: usize = @intCast(self.view.size[0]);
        const theirs: usize = @intCast(source.view.size[0]);
        // in every build: a copy past either buffer lands in its neighbour's bytes (spec 103)
        if (row != source.bytes() / theirs or to > mine or count > mine - to or from > theirs or count > theirs - from) {
            std.log.warn("cuda.Buffer.copyRows: rows {d} to {d} of {d} into rows {d} to {d} of {d}", .{ from, from + count, theirs, to, to + count, mine });
            return error.CudaFailure;
        }
        const dst: [*]u8 = @ptrCast(self.view.ptr.?);
        const src: [*]const u8 = @ptrCast(source.view.ptr.?);
        std.debug.assert(dst != src or to >= from + count or from >= to + count);
        return check(cudaMemcpyAsync(dst + to * row, src + from * row, count * row, device_to_device, current_stream));
    }

    /// Copies the first `count` bytes of `source` to this buffer's byte `offset` on, as a launch in the
    /// stream's order: a part of a row laid beside another's.
    pub fn copyAt(self: Buffer, offset: usize, source: Buffer, count: usize) Error!void {
        std.debug.assert(offset + count <= self.bytes() and count <= source.bytes() and self.view.ptr != source.view.ptr);
        const dst: [*]u8 = @ptrCast(self.view.ptr.?);
        return check(cudaMemcpyAsync(dst + offset, source.view.ptr, count, device_to_device, current_stream));
    }

    /// Copies `count` contiguous rows of `source` (its second dimension's bytes a row) into this buffer's first
    /// `count` rows at byte `column` of each, as a launch in the stream's order: parts of a row laid beside each
    /// other.
    pub fn copyColumns(self: Buffer, column: usize, source: View, count: usize) Error!void {
        const pitch = self.bytes() / @as(usize, @intCast(self.view.size[0]));
        const width = source.count() / @as(usize, @intCast(source.size[0])) * source.dtype.size();
        std.debug.assert(source.row_stride == 0 and column + width <= pitch and count <= @as(usize, @intCast(self.view.size[0])) and count <= @as(usize, @intCast(source.size[0])));
        const dst: [*]u8 = @ptrCast(self.view.ptr.?);
        return check(cudaMemcpy2DAsync(dst + column, pitch, source.ptr, width, width, count, device_to_device, current_stream));
    }

    /// Copies `source` in at byte `offset`, for a buffer filled from several host arrays.
    pub fn uploadAt(self: Buffer, offset: usize, source: []const u8) Error!void {
        std.debug.assert(offset + source.len <= self.bytes());
        const at: [*]u8 = @ptrCast(self.view.ptr.?);
        return check(phases.copy(at + offset, source.ptr, source.len));
    }

    /// `uploadAt` as a launch, in the stream's order, for a few 4-byte words (`zig/kernels/common/words.cu`): the
    /// words go in the launch's arguments, so the host neither waits for the launches before it nor keeps the
    /// source, and a capture records it with its values (spec 114 WS-02).
    pub fn uploadLaunched(self: Buffer, offset: usize, source: []const u8) Error!void {
        std.debug.assert(offset % 4 == 0 and source.len % 4 == 0 and offset + source.len <= self.bytes());
        var words: [32]u32 = undefined;
        const n = source.len / 4;
        if (n == 0) return;
        if (n > words.len) return self.uploadAt(offset, source);
        @memcpy(std.mem.sliceAsBytes(words[0..n]), source);
        const all: View = .init(self.view.ptr.?, .u32, &.{self.bytes() / 4});
        return launched(zs_words_set(&all, @intCast(offset / 4), &words, @intCast(n)));
    }

    /// `downloadAt` as a launch, in the stream's order, into page-locked `dest` (`HostBuffer`): the host does not
    /// wait; `dest` holds the bytes once an `Event` recorded after it has been waited for.
    pub fn downloadLaunched(self: Buffer, offset: usize, dest: []u8) Error!void {
        std.debug.assert(offset + dest.len <= self.bytes());
        const at: [*]const u8 = @ptrCast(self.view.ptr.?);
        return check(cudaMemcpyAsync(dest.ptr, at + offset, dest.len, device_to_host, current_stream));
    }

    /// Copies `dest.len` bytes out from byte `offset`, after pending launches finish.
    pub fn downloadAt(self: Buffer, offset: usize, dest: []u8) Error!void {
        std.debug.assert(offset + dest.len <= self.bytes());
        const at: [*]const u8 = @ptrCast(self.view.ptr.?);
        return copyOut(dest, at + offset);
    }

    /// The buffer's first `count` rows as a view. Past the buffer it panics in every build, a fast one too: a
    /// view past its buffer is a launch that writes past it (spec 103; `std.debug.assert` held it in safe builds).
    pub fn rows(self: Buffer, count: usize) View {
        const held: usize = @intCast(self.view.size[0]);
        if (count > held) std.debug.panic("cuda.Buffer.rows: {d} rows of a buffer of {d}", .{ count, held });
        var view = self.view;
        view.size[0] = @intCast(count);
        return view;
    }

    /// `count` rows from row `first` as a view; past the buffer it panics in every build, as `rows` does.
    pub fn span(self: Buffer, first: usize, count: usize) View {
        const held: usize = @intCast(self.view.size[0]);
        if (self.view.row_stride != 0 or first > held or count > held - first) std.debug.panic("cuda.Buffer.span: rows {d} to {d} of a buffer of {d}", .{ first, first + count, held });
        var view = self.view;
        const row = self.bytes() / @as(usize, @intCast(self.view.size[0]));
        view.ptr = @ptrCast(@as([*]u8, @ptrCast(self.view.ptr.?)) + first * row);
        view.size[0] = @intCast(count);
        return view;
    }

    /// Copies the buffer out after pending launches finish; `dest` must cover it exactly.
    pub fn download(self: Buffer, dest: []u8) Error!void {
        std.debug.assert(dest.len == self.bytes());
        return copyOut(dest, self.view.ptr.?);
    }

    /// A buffer of `shape` elements of `dtype` with no memory yet, which `deinit` skips: a part for `cut`.
    pub fn shaped(dtype: DType, shape: []const usize) Buffer {
        var view: View = .{ .dtype = dtype, .ndim = @intCast(shape.len) };
        for (shape, 0..) |dim, i| view.size[i] = @intCast(dim);
        return .{ .view = view };
    }
};

/// One device allocation for every buffer of `lists`, each shaped and with no memory yet (`Buffer.shaped`):
/// each becomes a view of it, where `block.zig` places it, and the allocation is returned. The parts own
/// nothing: whoever holds them frees the allocation alone, once, and uses no part past it. One `cudaMalloc`
/// where each part would be one (a pool's kept state was 86, 5.5 ms with the device idle: spec 146 WS-04).
pub fn cut(lists: []const []Buffer) Error!Buffer {
    var end: usize = 0;
    for (lists) |list| for (list) |part| {
        std.debug.assert(part.view.ptr == null);
        end = block_file.place(end, part.bytes()) + part.bytes();
    };
    const whole: Buffer = try .init(.u8, &.{@max(end, 1)});
    const base: [*]u8 = @ptrCast(whole.view.ptr.?);
    end = 0;
    for (lists) |list| for (list) |*part| {
        const at = block_file.place(end, part.bytes());
        part.view.ptr = base + at;
        end = at + part.bytes();
    };
    return whole;
}

/// Buffers of one allocation (`cut`): `list` its views and `block` the allocation, which alone is freed. With
/// no block (`block.view.ptr` null) each buffer of `list` that has memory owns it.
pub const Parts = struct {
    list: []Buffer = &.{},
    block: Buffer = .{ .view = .{} },

    /// The bytes of the buffers that have memory, without the allocation's alignment between them.
    pub fn bytes(self: Parts) usize {
        var total: usize = 0;
        for (self.list) |buffer| if (buffer.view.ptr != null) {
            total += buffer.bytes();
        };
        return total;
    }

    pub fn deinit(self: *Parts, gpa: std.mem.Allocator) void {
        if (self.block.view.ptr != null) {
            self.block.deinit();
        } else for (self.list) |*buffer| {
            if (buffer.view.ptr != null) buffer.deinit();
        }
        gpa.free(self.list);
        self.* = .{};
    }
};

extern fn cudaMemcpyAsync(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: c_int, stream: ?*anyopaque) c_int;
extern fn cudaMemcpy2DAsync(dst: ?*anyopaque, dpitch: usize, src: ?*const anyopaque, spitch: usize, width: usize, height: usize, kind: c_int, stream: ?*anyopaque) c_int;
extern fn cudaMemsetAsync(ptr: ?*anyopaque, value: c_int, count: usize, stream: ?*anyopaque) c_int;

/// The memory linears compute in: a split matmul's K slices, a face's staging.
/// A linear asks for it at init and holds a view; the pool owns the bytes and outlives its linears, so a model
/// has one and frees it after its layers. What a linear leaves there is dead once the kernel that reads it has
/// run, which is before the next linear's first launch: the rule that lets one buffer serve many linears, and
/// the one `poisoned` holds them to.
pub const Scratch = struct {
    gpa: std.mem.Allocator,
    lent: std.ArrayList(Entry) = .empty,
    /// The first row of a call from which a linear that quantizes its activations (`nvfp4.Activations`) does; the
    /// rows before keep bf16 rows. A model sets it around a window whose rows take both forms (GLM FP4-Local, spec
    /// 115: a prompt's rows quantized as the card's vLLM quantizes them, a step's and a verify window's bf16); 0,
    /// every row quantized, otherwise.
    quantize_from: usize = 0,

    /// On, in tests alone: every linear's first launch is preceded by a fill of every buffer of its pool with
    /// NaN patterns, in stream order (a capture records it with the launches), but for the rows the linear is
    /// handed. A model whose bits survive that reads no scratch another linear's call came after. A test sets
    /// it around what it compares; `ZSPARK_POISON_SCRATCH` in the environment sets it for a whole test run.
    pub var poisoned: bool = false;

    /// Fills `buffer` as `enter` fills the pool, when the mode is on: for scratch that is not a linear's and is
    /// held to the same rule at its own points (the attention's, before each layer's block).
    pub fn poison(buffer: Buffer) Error!void {
        if (!poisoning()) return;
        return check(cudaMemsetAsync(buffer.view.ptr, 0xff, buffer.bytes(), current_stream));
    }

    fn poisoning() bool {
        if (poisoned) return true;
        if (!@import("builtin").is_test) return false;
        const asked = struct {
            var read = false;
            var on = false;
        };
        if (!asked.read) {
            asked.on = std.testing.environ.getPosix("ZSPARK_POISON_SCRATCH") != null;
            asked.read = true;
        }
        return asked.on;
    }

    /// What a buffer holds. Two roles never share memory, whatever their shapes.
    pub const Role = enum { part, staging };

    const Entry = struct { role: Role, buffer: Buffer };

    /// A pool and the rows a linear's calls take of it: what a linear's `init` is given.
    pub const Rows = struct {
        pool: *Scratch,
        count: usize,

        /// The same pool for calls of `factor` times the rows.
        pub fn times(self: Rows, factor: usize) Rows {
            return .{ .pool = self.pool, .count = self.count * factor };
        }
    };

    pub fn init(gpa: std.mem.Allocator) Scratch {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Scratch) void {
        for (self.lent.items) |*entry| entry.buffer.deinit();
        self.lent.deinit(self.gpa);
        self.* = undefined;
    }

    /// A pool that stays where it is, for an owner that moves: a model's.
    pub fn create(gpa: std.mem.Allocator) std.mem.Allocator.Error!*Scratch {
        const self = try gpa.create(Scratch);
        self.* = .init(gpa);
        return self;
    }

    /// Frees a pool that `create` made.
    pub fn destroy(self: *Scratch) void {
        const gpa = self.gpa;
        self.deinit();
        gpa.destroy(self);
    }

    /// For linears whose calls take up to `count` rows. The pool must stay where it is while they live.
    pub fn rows(self: *Scratch, count: usize) Rows {
        return .{ .pool = self, .count = count };
    }

    /// Off, in tests alone: every request gets a buffer of its own, as each linear had before spec 033, so a
    /// test can hold the shared buffers to those, bit for bit.
    pub var shared: bool = true;

    /// `shape` elements of `dtype` for `role`, the pool's to free. A part is lent by capacity: the smallest
    /// buffer of that type whose bytes cover the request, and a buffer already returned is never moved, so a
    /// linear that kept its pointer keeps it. Rotated rows and staging stay one buffer for an exact shape. What a buffer holds is
    /// undefined before each call that uses it.
    pub fn lend(self: *Scratch, role: Role, dtype: DType, shape: []const usize) (Error || std.mem.Allocator.Error)!Buffer {
        if (shared) {
            if (role == .part) {
                const need = countOf(shape) * dtype.size();
                var fit: ?usize = null;
                for (self.lent.items, 0..) |entry, index| {
                    if (entry.role != .part or entry.buffer.view.dtype != dtype) continue;
                    const held = entry.buffer.bytes();
                    if (held < need) continue;
                    const smaller = if (fit) |best| held < self.lent.items[best].buffer.bytes() else true;
                    if (smaller) fit = index;
                }
                if (fit) |index| return self.lent.items[index].buffer;
            } else for (self.lent.items) |entry| {
                const view = entry.buffer.view;
                if (entry.role != role or view.dtype != dtype or view.ndim != shape.len) continue;
                for (shape, view.size[0..shape.len]) |asked, held| {
                    if (asked != @as(usize, @intCast(held))) break;
                } else return entry.buffer;
            }
        }
        try self.lent.ensureUnusedCapacity(self.gpa, 1);
        const buffer: Buffer = try .init(dtype, shape);
        self.lent.appendAssumeCapacity(.{ .role = role, .buffer = buffer });
        return buffer;
    }

    fn countOf(shape: []const usize) usize {
        var count: usize = 1;
        for (shape) |dim| count *= dim;
        return count;
    }

    /// Bytes lent.
    pub fn bytes(self: Scratch) usize {
        var total: usize = 0;
        for (self.lent.items) |entry| total += entry.buffer.bytes();
        return total;
    }

    /// A linear is about to launch on rows `x`: nothing in the pool is read from here on but what this linear
    /// writes and `x` where the pool holds it. Nothing to do unless `poisoned`.
    pub fn enter(self: *Scratch, x: View) Error!void {
        if (!poisoning()) return;
        for (self.lent.items) |entry| {
            if (holds(entry.buffer, x.ptr)) continue;
            try check(cudaMemsetAsync(entry.buffer.view.ptr, 0xff, entry.buffer.bytes(), current_stream));
        }
    }

    fn holds(buffer: Buffer, ptr: ?*anyopaque) bool {
        const at = @intFromPtr(ptr orelse return false);
        const first = @intFromPtr(buffer.view.ptr.?);
        return at >= first and at < first + @max(buffer.bytes(), 1);
    }
};

/// The stream launches go to: a capture's, or null for the default one. `zs::stream()` on this side.
var current_stream: ?*anyopaque = null;

/// The stream this thread's launches go to: a capture's while one records, else null, the default stream.
pub fn currentStream() ?*anyopaque {
    return current_stream;
}

/// A split matmul's K slices before they are summed: fp32 [sk, rows, n], slice by slice. The kernel that reads
/// them adds an element's slices in order, which is the sum the matmul's own reduction makes, so a launch is
/// saved and no bit moves (spec 028). One slice is the product itself.
pub const Slices = struct {
    view: View,

    /// Off, in tests alone: every matmul sums its own slices and converts its own rows, as before spec 028, so
    /// a test can hold the folded launches to those, bit for bit.
    pub var folded: bool = true;

    /// `out` fp32 [rows, n], already the product, as one slice.
    pub fn one(out: View) Slices {
        std.debug.assert(out.dtype == .f32 and out.ndim == 2 and out.row_stride == 0);
        var view = out;
        view.ndim = 3;
        view.size = .{ 1, out.size[0], out.size[1], 0, 0, 0 };
        return .{ .view = view };
    }

    /// The first `rows` rows of each of `part`'s `sk` slices of `n` values a row, as the matmul lays them:
    /// slice after slice of `rows` rows, so the view is contiguous whatever `part` holds beyond it.
    pub fn of(part: View, sk: usize, rows: usize, n: usize) Slices {
        return .{ .view = .init(part.ptr.?, .f32, &.{ sk, rows, n }) };
    }
};

/// Zeroes `view`'s contiguous rows, as a launch in the stream's order.
pub fn zeroLaunched(view: View) Error!void {
    std.debug.assert(view.row_stride == 0);
    return check(cudaMemsetAsync(view.ptr, 0, view.count() * view.dtype.size(), current_stream));
}

/// Copies a contiguous view's elements out, after pending launches finish.
pub fn download(view: View, dest: []u8) Error!void {
    std.debug.assert(dest.len == view.count() * view.dtype.size());
    return copyOut(dest, view.ptr.?);
}

/// Copies `source` over a contiguous view's elements.
pub fn upload(view: View, source: []const u8) Error!void {
    std.debug.assert(source.len == view.count() * view.dtype.size());
    return check(phases.copy(view.ptr, source.ptr, source.len));
}

extern fn zs_last_error() [*:0]const u8;
pub extern fn zs_set_stream(stream: ?*anyopaque) void;
extern fn zs_words_set(dst: *const View, at: i64, words: [*]const u32, n: i64) c_int;

/// A bound a launch relies on, checked in every build (spec 103): `std.debug.assert` is gone in a fast one, and a
/// launch past a buffer then writes its neighbour's bytes. Logs `format` as a warning (a test that expects the
/// refusal fails on an error line) and fails as a launch would.
pub fn bound(ok: bool, comptime format: []const u8, args: anytype) Error!void {
    if (ok) return;
    std.log.warn(format, args);
    return error.CudaFailure;
}

/// Turns a launcher's status into an error, logging the reason it recorded.
pub fn launched(status: c_int) Error!void {
    if (status == 0) return;
    std.log.err("kernel: {s}", .{zs_last_error()});
    return error.CudaFailure;
}

pub const kernels = struct {
    pub extern fn zs_b16_gemm(x: *const View, w: *const View, out: ?*const View, part: ?*const View, sk: i64, tile: i64) c_int;
    pub extern fn zs_b16_gemm_entries(x: *const View, m: i64, x_stride: i64, w: [*]const View, count: i64, out: *const View, out_stride: i64, live: *const View, live_column: i64) c_int;
    pub extern fn zs_embed(ids: *const View, table: *const View, out: *const View, streams: i64) c_int;
    pub extern fn zs_embed_rows(features: *const View, pairs: *const View, out: *const View, streams: i64) c_int;
    pub extern fn zs_columns_place(src: *const View, dst: *const View, at: i64) c_int;
    pub extern fn zs_b16_gemm_bias(x: *const View, w: *const View, bias: *const View, out: *const View, act: i64) c_int;
    pub extern fn zs_layernorm(h: *const View, branch: ?*const View, weight: ?*const View, bias: ?*const View, y: ?*const View, eps: f64) c_int;
    pub extern fn zs_nvfp4_qmmf_tile(x: *const View, w: *const View, ts: *const View, out: *const View, n: i64, sk: i64, npad: i64, fuse: c_int, f32: c_int) c_int;
    pub extern fn zs_nvfp4_qmm_a8(x: *const View, w: *const View, bs: *const View, out: *const View, work: ?*const View, mode: i64, n: i64, sk: i64, npad: i64, bm: i64, fuse: c_int, f32: c_int, input_scale: f64) c_int;
    pub extern fn zs_attn_prep(qkv: *const View, q_norm: ?*const View, k_norm: ?*const View, inv: ?*const View, pos: *const View, q: *const View, kc: *const View, vc: *const View, eps: f64) c_int;
    pub extern fn zs_attn_attend(q: *const View, kc: *const View, vc: *const View, pos: *const View, po: *const View, pm: *const View, pl: *const View, out: *const View, ids: ?*const View, nk: ?*const View, sparse: ?*const View, qs: ?*const View, ks: ?*const View, vs: ?*const View, scale: f64, context: i64) c_int;
    pub extern fn zs_attn_attend_group(q: *const View, kc: *const View, vc: *const View, pos: *const View, out: *const View, scale: f64, context: i64) c_int;
    pub extern fn zs_attn_attend_window(q: *const View, kc: *const View, vc: *const View, pos: *const View, out: *const View, ids: ?*const View, nk: ?*const View, sparse: ?*const View, qs: ?*const View, ks: ?*const View, vs: ?*const View, scale: f64, context: i64) c_int;
    pub extern fn zs_qwen4_attn_prep_rows(sid: *const View, tables: *const View, bases: *const View, p: *const View, pos: *const View, q_scale: *const View, k_scale: *const View, iq_scale: *const View, inv_freq: *const View, q: *const View, kc: *const View, vc: *const View, iq: *const View, ikc: *const View, qs: ?*const View, ks: ?*const View, vs: ?*const View, eps: f64, context: i64, ropes: ?*const View, s1: i64, s2: i64) c_int;
    pub extern fn zs_attn_attend_window_rows(sid: *const View, tables: *const View, bases: *const View, q: *const View, kc: *const View, vc: *const View, pos: *const View, out: *const View, ids: ?*const View, nk: ?*const View, sparse: ?*const View, qs: ?*const View, ks: ?*const View, vs: ?*const View, scale: f64, context: i64) c_int;
    pub extern fn zs_attn_attend_rows(sid: *const View, tables: *const View, bases: *const View, q: *const View, kc: *const View, vc: *const View, pos: *const View, po: *const View, pm: *const View, pl: *const View, out: *const View, ids: ?*const View, nk: ?*const View, sparse: ?*const View, qs: ?*const View, ks: ?*const View, vs: ?*const View, scale: f64, context: i64) c_int;
    pub extern fn zs_qwen4_attn_pool_slots(seg: *const View, ptrs: *const View, stride: i64, pooled_col: i64, table_col: i64, bases: *const View, ikc: *const View, scale: *const View, inv_freq: *const View, eps: f64, ratio: i64, blocks: i64, ropes: ?*const View, s1: i64, s2: i64) c_int;
    pub extern fn zs_qwen4_attn_select_window_table(table: *const View, iq: *const View, pooled: *const View, pos: *const View, scores: *const View, ids: *const View, nk: *const View, sparse: *const View, rows: i64, heads: i64, dim: i64, nbl: i64, ratio: i64, top: i64) c_int;
    pub extern fn zs_qwen4_attn_select_table(table: *const View, dots: *const View, pos: *const View, scores: *const View, ids: *const View, nk: *const View, sparse: *const View, heads: i64, dim: i64, ratio: i64, top: i64) c_int;
    pub extern fn zs_qwen4_ple_embed(v: *const View, out: *const View, heads: i64, scale: f64) c_int;
    pub extern fn zs_qwen4_ple_gate(keys: *const View, vals: *const View, h: *const View, norm_key: *const View, norm_query: *const View, gated: *const View, pss: *const View, eps: f64, streams: i64) c_int;
};

pub const DeviceInfo = extern struct {
    name: [256]u8,
    major: i32,
    minor: i32,
    free_bytes: u64,
    total_bytes: u64,
    kernel_sum: u64,

    pub fn nameSlice(self: *const DeviceInfo) []const u8 {
        return std.mem.sliceTo(&self.name, 0);
    }
};

extern fn zspark_probe(info: *DeviceInfo) c_int;

/// Queries the current device and launches one kernel on it.
pub fn probe() Error!DeviceInfo {
    var info: DeviceInfo = undefined;
    try check(zspark_probe(&info));
    return info;
}

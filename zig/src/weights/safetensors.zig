//! A safetensors file, memory-mapped: an 8-byte little-endian header length, a JSON header naming each tensor's
//! dtype, shape and byte range, then the tensors' bytes.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const linux = std.os.linux;

pub const DType = enum {
    BOOL,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    I64,
    U64,
    F64,
    F8_E4M3,
    F8_E5M2,

    pub fn size(self: DType) usize {
        return switch (self) {
            .BOOL, .U8, .I8, .F8_E4M3, .F8_E5M2 => 1,
            .I16, .U16, .F16, .BF16 => 2,
            .I32, .U32, .F32 => 4,
            .I64, .U64, .F64 => 8,
        };
    }
};

pub const Tensor = struct {
    dtype: DType,
    shape: []const u64,
    /// The tensor's bytes in the mapping, valid until the file's `close`.
    data: []const u8,

    pub fn count(self: Tensor) u64 {
        var n: u64 = 1;
        for (self.shape) |dim| n *= dim;
        return n;
    }
};

pub const OpenError = error{ InvalidSafetensors, UnsupportedDType } || Allocator.Error || Io.File.OpenError ||
    Io.File.LengthError || std.posix.MMapError;

pub const File = struct {
    map: []align(std.heap.page_size_min) const u8,
    /// The file's first bytes: its header's length and the header (`fromHead` takes them on another machine).
    head: []const u8,
    /// The file opened again with O_DIRECT, for `Checkpoint.hold` and `Stage.read`; -1 where the filesystem has no direct reads.
    direct: std.posix.fd_t = -1,
    arena: std.heap.ArenaAllocator,
    tensors: std.StringArrayHashMapUnmanaged(Tensor),

    const header_prefix = 8;

    pub fn open(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8) OpenError!File {
        // a directory opens for reading and has a length, and only its mapping fails (ENODEV): refuse it here
        const file = try dir.openFile(io, sub_path, .{ .allow_directory = false });
        defer file.close(io);
        const length = try file.length(io);
        if (length < header_prefix) return error.InvalidSafetensors;
        // pages are read as tensors are touched: a shard can be larger than free memory
        const map = try std.posix.mmap(null, length, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
        errdefer std.posix.munmap(map);

        var self: File = .{ .map = map, .head = try headOf(map), .arena = .init(gpa), .tensors = .empty };
        errdefer self.arena.deinit();
        try self.parse();
        var name: [std.posix.PATH_MAX]u8 = undefined;
        if (sub_path.len < name.len) {
            @memcpy(name[0..sub_path.len], sub_path);
            name[sub_path.len] = 0;
            const fd: isize = @bitCast(linux.openat(dir.handle, @ptrCast(&name), .{ .ACCMODE = .RDONLY, .DIRECT = true, .CLOEXEC = true }, 0));
            if (fd >= 0) self.direct = @intCast(fd);
        }
        return self;
    }

    pub fn close(self: *File) void {
        if (self.direct >= 0) _ = linux.close(self.direct);
        self.arena.deinit();
        std.posix.munmap(self.map);
        self.* = undefined;
    }

    pub fn get(self: *const File, name: []const u8) ?Tensor {
        return self.tensors.get(name);
    }

    /// A file of `length` bytes whose first are `head` (another machine's `File.head`), with no file under it:
    /// its mapping is an address range of that length that nothing may read, so that its tensors' slices are
    /// where the file's would be, and their bytes are staged from elsewhere (`Stage.lay`). A slice read through
    /// it is a fault, not another file's bytes.
    pub fn fromHead(gpa: Allocator, head: []const u8, length: u64) OpenError!File {
        if ((try headOf(head)).len != head.len or length < head.len) return error.InvalidSafetensors;
        const map = try std.posix.mmap(null, length, .{}, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .NORESERVE = true }, -1, 0);
        errdefer std.posix.munmap(map);
        var self: File = .{ .map = map, .head = &.{}, .arena = .init(gpa), .tensors = .empty };
        errdefer self.arena.deinit();
        self.head = try self.arena.allocator().dupe(u8, head);
        try self.parse();
        return self;
    }

    /// The header's length and the header at the start of `bytes`.
    fn headOf(bytes: []const u8) error{InvalidSafetensors}![]const u8 {
        if (bytes.len < header_prefix) return error.InvalidSafetensors;
        const header_len = std.mem.readInt(u64, bytes[0..header_prefix], .little);
        if (header_len > bytes.len - header_prefix) return error.InvalidSafetensors;
        return bytes[0 .. header_prefix + header_len];
    }

    fn parse(self: *File) OpenError!void {
        const arena = self.arena.allocator();
        const data = self.map[self.head.len..];

        const Entry = struct { dtype: []const u8, shape: []const u64, data_offsets: [2]u64 };
        const root = std.json.parseFromSliceLeaky(std.json.Value, arena, self.head[header_prefix..], .{}) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidSafetensors,
        };
        if (root != .object) return error.InvalidSafetensors;
        try self.tensors.ensureTotalCapacity(arena, root.object.count());
        var entries = root.object.iterator();
        while (entries.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const entry = std.json.parseFromValueLeaky(Entry, arena, kv.value_ptr.*, .{}) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.InvalidSafetensors,
            };
            const dtype = std.meta.stringToEnum(DType, entry.dtype) orelse return error.UnsupportedDType;
            const begin, const end = entry.data_offsets;
            const tensor: Tensor = .{ .dtype = dtype, .shape = entry.shape, .data = undefined };
            if (begin > end or end > data.len or end - begin != tensor.count() * dtype.size())
                return error.InvalidSafetensors;
            self.tensors.putAssumeCapacity(kv.key_ptr.*, .{ .dtype = dtype, .shape = entry.shape, .data = data[begin..end] });
        }
    }
};

/// A checkpoint directory: every tensor of its `*.safetensors` shards by name.
pub const Checkpoint = struct {
    shards: []File,
    tensors: std.StringArrayHashMapUnmanaged(Tensor),
    gpa: Allocator,

    pub const Error = OpenError || Io.Dir.OpenError || Io.Dir.Iterator.Error || Io.Dir.ReadFileAllocError || Io.Dir.StatFileError || error{DuplicateTensor};

    pub fn open(gpa: Allocator, io: Io, parent: Io.Dir, sub_path: []const u8) Error!Checkpoint {
        var dir = try parent.openDir(io, sub_path, .{ .iterate = true });
        defer dir.close(io);

        var shards: std.ArrayList(File) = .empty;
        errdefer {
            for (shards.items) |*shard| shard.close();
            shards.deinit(gpa);
        }
        var tensors: std.StringArrayHashMapUnmanaged(Tensor) = .empty;
        errdefer tensors.deinit(gpa);

        var entries = dir.iterate();
        while (try entries.next(io)) |entry| {
            // a download in progress leaves shards the index names but the directory lacks; those come later
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            try addShard(gpa, io, dir, entry.name, &shards, &tensors);
        }
        // shards in a subdirectory (`tensors/model-00001-of-00021.safetensors`) are found by the index's names,
        // as vLLM PR 60646 finds them, beside the directory's own (GLM FP4-Local's `tensors/native-QB.safetensors`)
        {
            var names_arena: std.heap.ArenaAllocator = .init(gpa);
            defer names_arena.deinit();
            for (try indexedShards(names_arena.allocator(), io, dir)) |name| {
                const found = dir.statFile(io, name, .{}) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                if (found.kind != .file and found.kind != .sym_link) continue;
                try addShard(gpa, io, dir, name, &shards, &tensors);
            }
        }
        return .{ .shards = try shards.toOwnedSlice(gpa), .tensors = tensors, .gpa = gpa };
    }

    fn addShard(gpa: Allocator, io: Io, dir: Io.Dir, name: []const u8, shards: *std.ArrayList(File), tensors: *std.StringArrayHashMapUnmanaged(Tensor)) Error!void {
        try shards.ensureUnusedCapacity(gpa, 1);
        try adopt(gpa, try File.open(gpa, io, dir, name), shards, tensors);
    }

    /// The checkpoint of files another machine opened, from their `File.head`s and lengths in its shards'
    /// order (`File.fromHead`): no file is opened, every tensor's bytes are staged from elsewhere.
    pub fn fromHeads(gpa: Allocator, heads: []const []const u8, lengths: []const u64) (OpenError || error{DuplicateTensor})!Checkpoint {
        std.debug.assert(heads.len == lengths.len);
        var shards: std.ArrayList(File) = .empty;
        errdefer {
            for (shards.items) |*shard| shard.close();
            shards.deinit(gpa);
        }
        var tensors: std.StringArrayHashMapUnmanaged(Tensor) = .empty;
        errdefer tensors.deinit(gpa);
        for (heads, lengths) |head, length| {
            try shards.ensureUnusedCapacity(gpa, 1);
            try adopt(gpa, try File.fromHead(gpa, head, length), &shards, &tensors);
        }
        return .{ .shards = try shards.toOwnedSlice(gpa), .tensors = tensors, .gpa = gpa };
    }

    /// `opened` the last of `shards` (which has room for it), its tensors in `tensors`; closed where it fails
    /// before it is `shards`'.
    fn adopt(gpa: Allocator, opened: File, shards: *std.ArrayList(File), tensors: *std.StringArrayHashMapUnmanaged(Tensor)) (Allocator.Error || error{DuplicateTensor})!void {
        var owned = opened;
        tensors.ensureUnusedCapacity(gpa, owned.tensors.count()) catch |err| {
            owned.close();
            return err;
        };
        const shard = shards.addOneAssumeCapacity();
        shard.* = owned;
        for (shard.tensors.keys(), shard.tensors.values()) |tensor_name, tensor| {
            const slot = tensors.getOrPutAssumeCapacity(tensor_name);
            if (slot.found_existing) return error.DuplicateTensor;
            slot.value_ptr.* = tensor;
        }
    }

    pub const index_name = "model.safetensors.index.json";

    /// The distinct shard paths `model.safetensors.index.json` names under a subdirectory, in their order of first
    /// appearance; none without an index. The index of 375,487 tensors is 47 MB: its values are found by a scan
    /// for the quoted names ending `.safetensors`, not by a JSON parse.
    pub fn indexedShards(arena: Allocator, io: Io, dir: Io.Dir) Error![]const []const u8 {
        const text = dir.readFileAlloc(io, index_name, arena, .limited(1 << 30)) catch |err| switch (err) {
            error.FileNotFound => return &.{},
            else => return err,
        };
        var names: std.ArrayList([]const u8) = .empty;
        const suffix = ".safetensors\"";
        var from: usize = 0;
        while (std.mem.indexOfPos(u8, text, from, suffix)) |end| : (from = end + suffix.len) {
            const start = (std.mem.lastIndexOfScalar(u8, text[0..end], '"') orelse continue) + 1;
            const name = text[start .. end + suffix.len - 1];
            // a key never ends so; a value with no directory is one the directory itself would hold
            if (std.mem.indexOfScalar(u8, name, '/') == null or std.mem.indexOf(u8, name, "..") != null) continue;
            for (names.items) |known| {
                if (std.mem.eql(u8, known, name)) break;
            } else try names.append(arena, name);
        }
        return names.items;
    }

    pub fn close(self: *Checkpoint) void {
        self.tensors.deinit(self.gpa);
        for (self.shards) |*shard| shard.close();
        self.gpa.free(self.shards);
        self.* = undefined;
    }

    pub fn get(self: *const Checkpoint, name: []const u8) ?Tensor {
        return self.tensors.get(name);
    }

    /// The shard whose mapping holds `bytes`, a tensor's `data` or a part of it.
    pub fn fileOf(self: *const Checkpoint, bytes: []const u8) ?*const File {
        for (self.shards) |*shard| {
            const base = @intFromPtr(shard.map.ptr);
            if (@intFromPtr(bytes.ptr) >= base and @intFromPtr(bytes.ptr) + bytes.len <= base + shard.map.len) return shard;
        }
        return null;
    }

    /// Has `ranges` (tensors' `data`) in memory until `release`, read from the disk directly and in parallel.
    ///
    /// A mapped file read by page faults comes in at 0.6 GiB/s on the Spark, and its pages stay in the page
    /// cache, which shares the machine's memory with the device: a 74 GiB model then makes every device
    /// allocation reclaim cache first. Here the pages under the ranges are replaced, in place, by anonymous
    /// memory filled with O_DIRECT reads (13 GiB/s on the same disk), and `release` puts the file's mapping
    /// back. The bytes at every address are the file's throughout, so slices taken before stay valid; but no
    /// other thread may read the ranges' pages during `hold` or `release`.
    ///
    /// A hint: where direct reads are not to be had, the ranges are left to be read as they are touched.
    pub fn hold(self: *const Checkpoint, gpa: Allocator, ranges: []const []const u8) Allocator.Error!Held {
        return self.holdBeside(gpa, ranges, null);
    }

    /// `hold`, while another thread reads what `beside` holds: the pages `beside` holds already (two parts'
    /// tensors that share a page) are left out, so that the other thread never sees them replaced.
    pub fn holdBeside(self: *const Checkpoint, gpa: Allocator, ranges: []const []const u8, beside: ?*const Held) Allocator.Error!Held {
        const page = std.heap.pageSize();
        var spans: std.ArrayList(Held.Span) = try .initCapacity(gpa, ranges.len);
        errdefer spans.deinit(gpa);
        for (ranges) |range| {
            if (range.len == 0) continue;
            const shard = self.fileOf(range) orelse continue;
            if (shard.direct < 0) continue;
            const start = std.mem.alignBackward(usize, @intFromPtr(range.ptr), page);
            const end = std.mem.alignForward(usize, @intFromPtr(range.ptr) + range.len, page);
            spans.appendAssumeCapacity(.{ .file = shard, .start = start, .len = end - start });
        }
        // Tensors that share a page, or follow each other, become one span.
        std.mem.sort(Held.Span, spans.items, {}, Held.Span.before);
        var merged: usize = 0;
        for (spans.items) |span| {
            if (merged > 0 and span.start <= spans.items[merged - 1].start + spans.items[merged - 1].len) {
                const last = &spans.items[merged - 1];
                last.len = @max(last.len, span.start + span.len - last.start);
            } else {
                spans.items[merged] = span;
                merged += 1;
            }
        }
        spans.shrinkRetainingCapacity(merged);
        if (beside) |other| try Held.Span.leaveOut(gpa, &spans, other.spans);

        var held: Held = .{ .spans = try spans.toOwnedSlice(gpa), .gpa = gpa };
        const mapping = Held.clock.now();
        for (held.spans, 0..) |span, i| {
            const at: isize = @bitCast(linux.mmap(@ptrFromInt(span.start), span.len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .FIXED = true }, -1, 0));
            if (at < 0) {
                held.covered = i;
                held.release();
                return .{ .spans = &.{}, .gpa = gpa };
            }
            // Fewer, larger faults as the reads fill the pages.
            _ = linux.madvise(@ptrFromInt(span.start), span.len, linux.MADV.HUGEPAGE);
        }
        held.covered = held.spans.len;
        const reading = Held.clock.now();
        Held.clock.map_ns += reading - mapping;

        var total: usize = 0;
        for (held.spans) |span| total += span.len;
        var threads: [Held.max_threads - 1]std.Thread = undefined;
        const wanted = @min(Held.max_threads, @max(1, total / Held.chunk), std.Thread.getCpuCount() catch 1);
        var spawned: usize = 0;
        while (spawned + 1 < wanted) : (spawned += 1) {
            threads[spawned] = std.Thread.spawn(.{}, Held.read, .{&held}) catch break;
        }
        held.read();
        for (threads[0..spawned]) |thread| thread.join();
        Held.clock.read_ns += Held.clock.now() - reading;
        Held.clock.bytes += total;
        if (held.failed.load(.monotonic)) {
            held.release();
            return .{ .spans = &.{}, .gpa = gpa };
        }
        // What the page cache already had of these pages is of no more use, and is in the device's way.
        for (held.spans) |span| _ = linux.fadvise(span.file.direct, @intCast(span.offset()), @intCast(span.len), linux.POSIX_FADV.DONTNEED);
        return held;
    }
};

/// Ranges of a checkpoint's mappings held in memory by `Checkpoint.hold`.
pub const Held = struct {
    spans: []Span,
    /// How many spans have anonymous memory over them.
    covered: usize = 0,
    /// The next chunk to read, counted over all spans.
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
    gpa: Allocator,

    const max_threads = 8;
    const chunk = 32 << 20;

    /// The time `hold` spent mapping anonymous memory over the spans and reading into it, and `release` putting
    /// the files back, summed over a load: what `cuda.phases` reports beside its own (`ZSPARK_LOAD_PHASES`).
    pub const clock = struct {
        pub var map_ns: u64 = 0;
        pub var read_ns: u64 = 0;
        pub var release_ns: u64 = 0;
        pub var bytes: u64 = 0;

        fn now() u64 {
            var ts: linux.timespec = undefined;
            _ = linux.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        }
    };

    const Span = struct {
        file: *const File,
        /// Page-aligned address in the file's mapping.
        start: usize,
        len: usize,

        fn offset(span: Span) usize {
            return span.start - @intFromPtr(span.file.map.ptr);
        }

        fn before(_: void, a: Span, b: Span) bool {
            return a.start < b.start;
        }

        /// `spans` without the pages of `taken`, both sorted and neither overlapping itself.
        fn leaveOut(gpa: Allocator, spans: *std.ArrayList(Span), taken: []const Span) Allocator.Error!void {
            var out: std.ArrayList(Span) = try .initCapacity(gpa, spans.items.len);
            errdefer out.deinit(gpa);
            var first: usize = 0;
            for (spans.items) |span| {
                var start = span.start;
                const end = span.start + span.len;
                while (first < taken.len and taken[first].start + taken[first].len <= start) first += 1;
                var k = first;
                while (k < taken.len and taken[k].start < end) : (k += 1) {
                    if (taken[k].start > start) try out.append(gpa, .{ .file = span.file, .start = start, .len = taken[k].start - start });
                    start = @max(start, taken[k].start + taken[k].len);
                }
                if (start < end) try out.append(gpa, .{ .file = span.file, .start = start, .len = end - start });
            }
            spans.deinit(gpa);
            spans.* = out;
        }
    };

    /// Nothing held: what a loader that reads its parts otherwise (`Stage`) releases.
    pub fn none(gpa: Allocator) Held {
        return .{ .spans = &.{}, .gpa = gpa };
    }

    /// The bytes held, page-aligned spans included.
    pub fn bytes(self: Held) usize {
        var total: usize = 0;
        for (self.spans) |span| total += span.len;
        return total;
    }

    fn read(self: *Held) void {
        while (!self.failed.load(.monotonic)) {
            var index = self.next.fetchAdd(1, .monotonic);
            const span = for (self.spans) |span| {
                const chunks = (span.len + chunk - 1) / chunk;
                if (index < chunks) break span;
                index -= chunks;
            } else return;
            var at = index * chunk;
            const end = @min(at + chunk, span.len);
            // The file may end inside the span's last page: a read that returns nothing has reached its end.
            while (at < end) {
                const got: isize = @bitCast(linux.pread(span.file.direct, @ptrFromInt(span.start + at), end - at, @intCast(span.offset() + at)));
                if (got < 0) return self.failed.store(true, .monotonic);
                if (got == 0) break;
                at += @intCast(got);
            }
        }
    }

    /// Gives the memory back and leaves the file's own mapping at the ranges, as before `hold`.
    pub fn release(self: *Held) void {
        const start = clock.now();
        defer clock.release_ns += clock.now() - start;
        for (self.spans[0..self.covered]) |span| {
            const at: isize = @bitCast(linux.mmap(@ptrFromInt(span.start), span.len, .{ .READ = true }, .{ .TYPE = .PRIVATE, .FIXED = true }, span.file.direct, @intCast(span.offset())));
            // The address range must not be left without the file's bytes: slices into it are still out there.
            if (at < 0) @panic("could not map a checkpoint file back");
        }
        self.gpa.free(self.spans);
        self.* = .{ .spans = &.{}, .gpa = self.gpa };
    }
};

/// Ranges of a checkpoint (tensors' `data`, or parts of them) read from the disk directly and in parallel into a
/// buffer of its own, which the next `read` reuses: a loader that takes a checkpoint a part at a time reads
/// each part's bytes here and finds them by `moved`, and the file's mapping is never touched.
///
/// A mapped file read by page faults comes in at 0.6 GiB/s on the Spark, and its pages stay in the page cache,
/// which shares the machine's memory with the device. O_DIRECT reads come in at the disk's rate (9.5 to 10.6
/// GiB/s), and into a buffer kept from one part to the next the kernel neither zeroes its pages again nor frees
/// them: GLM FP4-Local's load put anonymous memory over the mapping for each part and the file's mapping back
/// after, 3.3 s of 15 for the latter alone under the device's allocations (2026-10-20). The buffer is 2 MiB
/// aligned and asks for huge pages, which the device reads through the host's page tables as well.
///
/// Where a range's file has no direct reads it is left out, and read through the mapping as before.
pub const Stage = struct {
    buffer: []align(huge) u8 = &.{},
    spans: std.ArrayList(Span) = .empty,
    /// The next chunk to read, counted over all spans.
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),
    /// Each buffer the stage grows is registered so, and unregistered before it is freed.
    pin: ?Pin = null,
    pinned: bool = false,

    pub const huge = 2 << 20;
    const max_threads = 16;
    /// Spans of some 450 KiB (Flash Next's routed experts, a rank's halves with down's rows halved) are read at 6.3
    /// to 7.1 GiB/s by 8 threads and 9.3 to 9.7 by 16; spans of 880 KiB 9.1 to 9.7 by 8 and 10.0 by 16. GLM's
    /// spans (2.3 MiB) gain nothing, and one load of GLM's on 16 took 18.0 s against 9.1 (page cache confounded
    /// it): 16 only below this mean (`cuda-gb10`'s loading.md, 2026-10-21)
    const small_span = 640 << 10;
    const chunk = 32 << 20;

    /// The time `read` spent growing the buffer and reading into it, summed over a load: what `cuda.phases`
    /// reports beside its own (`ZSPARK_LOAD_PHASES`).
    pub const clock = struct {
        pub var map_ns: u64 = 0;
        pub var read_ns: u64 = 0;
        pub var bytes: u64 = 0;

        fn now() u64 {
            var ts: linux.timespec = undefined;
            _ = linux.clock_gettime(.MONOTONIC, &ts);
            return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
        }
    };

    const Span = struct {
        file: *const File,
        /// Page-aligned address in the file's mapping, and where its bytes are in `buffer`.
        start: usize,
        len: usize,
        at: usize,

        fn offset(span: Span) usize {
            return span.start - @intFromPtr(span.file.map.ptr);
        }

        fn before(_: void, a: Span, b: Span) bool {
            return a.start < b.start;
        }

        fn laid(_: void, a: Span, b: Span) bool {
            if (a.file != b.file) return @intFromPtr(a.file) < @intFromPtr(b.file);
            return a.start < b.start;
        }
    };

    /// What a buffer is made known to the device as, where its bytes are sent or received by it (`pack.Wire`):
    /// `register` false leaves it as it is.
    pub const Pin = struct {
        register: *const fn ([]u8) bool,
        unregister: *const fn ([]u8) void,
    };

    pub fn deinit(self: *Stage, gpa: Allocator) void {
        self.free();
        self.spans.deinit(gpa);
        self.* = undefined;
    }

    /// Reads `ranges` of `checkpoint`'s mappings into the buffer, in place of what the last `read` left there.
    /// Ranges that share a page, or follow each other, are read once. On a failed read nothing is staged, and
    /// every range reads through the mapping.
    pub fn read(self: *Stage, gpa: Allocator, checkpoint: *const Checkpoint, ranges: []const []const u8) Allocator.Error!void {
        _ = try self.lay(gpa, checkpoint, ranges, false) orelse return;
        _ = self.fetch();
    }

    /// `read`'s spans and buffer without its reads: the bytes `ranges` stage at, laid out alike on any machine
    /// that holds the same files (`File.fromHead`'s too), file by file in the checkpoint's order and in each by
    /// offset. `every`: a range of a file without direct reads is laid out as well, for bytes that come from
    /// elsewhere. Null where the buffer cannot grow, nothing staged.
    pub fn lay(self: *Stage, gpa: Allocator, checkpoint: *const Checkpoint, ranges: []const []const u8, every: bool) Allocator.Error!?[]u8 {
        const page = std.heap.pageSize();
        self.spans.clearRetainingCapacity();
        try self.spans.ensureTotalCapacity(gpa, ranges.len);
        for (ranges) |range| {
            if (range.len == 0) continue;
            const shard = checkpoint.fileOf(range) orelse continue;
            if (shard.direct < 0 and !every) continue;
            const start = std.mem.alignBackward(usize, @intFromPtr(range.ptr), page);
            const end = std.mem.alignForward(usize, @intFromPtr(range.ptr) + range.len, page);
            self.spans.appendAssumeCapacity(.{ .file = shard, .start = start, .len = end - start, .at = 0 });
        }
        // by file (`shards`' order, which is their addresses' in the array) and offset: not by address, which
        // differs between machines, and a span never runs into the next file's mapping
        std.mem.sort(Span, self.spans.items, {}, Span.laid);
        var merged: usize = 0;
        var total: usize = 0;
        for (self.spans.items) |span| {
            const last = if (merged > 0) &self.spans.items[merged - 1] else null;
            if (last != null and last.?.file == span.file and span.start <= last.?.start + last.?.len) {
                const grown = @max(last.?.len, span.start + span.len - last.?.start);
                total += grown - last.?.len;
                last.?.len = grown;
            } else {
                self.spans.items[merged] = span;
                self.spans.items[merged].at = total;
                total += span.len;
                merged += 1;
            }
        }
        self.spans.shrinkRetainingCapacity(merged);
        // `moved` finds a range's span by address
        std.mem.sort(Span, self.spans.items, {}, Span.before);

        const growing = clock.now();
        defer clock.map_ns += clock.now() - growing;
        if (total > self.buffer.len) {
            self.free();
            const size = std.mem.alignForward(usize, total, huge);
            // over-asked by a huge page, so that a 2 MiB-aligned run of `size` bytes is inside, the rest given back
            const whole = std.posix.mmap(null, size + huge, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch {
                self.spans.clearRetainingCapacity();
                return null;
            };
            const base = std.mem.alignForward(usize, @intFromPtr(whole.ptr), huge);
            const lead = base - @intFromPtr(whole.ptr);
            if (lead > 0) std.posix.munmap(@alignCast(whole[0..lead]));
            if (huge - lead > 0) std.posix.munmap(@alignCast(whole[lead + size ..]));
            self.buffer = @as([*]align(huge) u8, @ptrFromInt(base))[0..size];
            _ = linux.madvise(self.buffer.ptr, size, linux.MADV.HUGEPAGE);
            if (self.pin) |pin| self.pinned = pin.register(self.buffer);
        }
        return self.buffer[0..total];
    }

    /// Reads what the last `lay` laid out from the files; false, and nothing staged, where a read failed.
    pub fn fetch(self: *Stage) bool {
        var total: usize = 0;
        for (self.spans.items) |span| total = @max(total, span.at + span.len);
        const reading = clock.now();
        self.next.store(0, .monotonic);
        self.failed.store(false, .monotonic);
        var threads: [max_threads - 1]std.Thread = undefined;
        const most: usize = if (total < small_span * self.spans.items.len) max_threads else max_threads / 2;
        const wanted = @min(most, @max(1, total / chunk), std.Thread.getCpuCount() catch 1);
        var spawned: usize = 0;
        while (spawned + 1 < wanted) : (spawned += 1) {
            threads[spawned] = std.Thread.spawn(.{}, readSome, .{self}) catch break;
        }
        self.readSome();
        for (threads[0..spawned]) |thread| thread.join();
        clock.read_ns += clock.now() - reading;
        clock.bytes += total;
        if (self.failed.load(.monotonic)) {
            self.spans.clearRetainingCapacity();
            return false;
        }
        // What the page cache already had of these pages is of no more use, and is in the device's way.
        for (self.spans.items) |span| _ = linux.fadvise(span.file.direct, @intCast(span.offset()), @intCast(span.len), linux.POSIX_FADV.DONTNEED);
        return true;
    }

    fn free(self: *Stage) void {
        if (self.buffer.len == 0) return;
        if (self.pinned) self.pin.?.unregister(self.buffer);
        self.pinned = false;
        std.posix.munmap(self.buffer);
        self.buffer = &.{};
    }

    fn readSome(self: *Stage) void {
        while (!self.failed.load(.monotonic)) {
            var index = self.next.fetchAdd(1, .monotonic);
            const span = for (self.spans.items) |span| {
                const chunks = (span.len + chunk - 1) / chunk;
                if (index < chunks) break span;
                index -= chunks;
            } else return;
            var at = index * chunk;
            const end = @min(at + chunk, span.len);
            // The file may end inside the span's last page: a read that returns nothing has reached its end.
            while (at < end) {
                const got: isize = @bitCast(linux.pread(span.file.direct, self.buffer.ptr + span.at + at, end - at, @intCast(span.offset() + at)));
                if (got < 0) return self.failed.store(true, .monotonic);
                if (got == 0) break;
                at += @intCast(got);
            }
        }
    }

    /// The bytes read, page-aligned spans included.
    pub fn bytes(self: *const Stage) usize {
        var total: usize = 0;
        for (self.spans.items) |span| total += span.len;
        return total;
    }

    /// Where `range`'s bytes are in the buffer, as the last `read` staged them, or null where it did not.
    pub fn moved(self: *const Stage, range: []const u8) ?[]const u8 {
        if (range.len == 0) return null;
        const at = @intFromPtr(range.ptr);
        const spans = self.spans.items;
        // the last span that starts at or before the range
        var low: usize = 0;
        var high: usize = spans.len;
        while (low < high) {
            const mid = (low + high) / 2;
            if (spans[mid].start <= at) low = mid + 1 else high = mid;
        }
        if (low == 0) return null;
        const span = spans[low - 1];
        if (at + range.len > span.start + span.len) return null;
        return (self.buffer.ptr + span.at + (at - span.start))[0..range.len];
    }

    /// How far `moved` puts the bytes of `range` from where the mapping has them, or null.
    pub fn shift(self: *const Stage, range: []const u8) ?isize {
        const to = self.moved(range) orelse return null;
        return @as(isize, @bitCast(@intFromPtr(to.ptr))) - @as(isize, @bitCast(@intFromPtr(range.ptr)));
    }
};

//! The routed experts' packing at load (spec 121 WS-02 moved it out of `experts.zig`): the host memory the packed
//! blocks pass through (`Staging`), the checkpoint's matrices as stored (`Matrix`), and the packers that place
//! them in the blocks `experts.zig`'s kernels read (`pack` for the bf16 products, `packA4` for the FP4 mma).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");

/// Output columns a block.
pub const cols = 32;
/// int32 a (32 columns, 32 inputs) block: 128 code words, then 16 of e4m3 scales.
pub const block_words = 144;

/// Host memory the packed blocks pass through. Kept from one layer to the next, it is paged in once: a fresh
/// gibibyte a layer costs more in page faults and in freeing than the packing itself.
pub const Staging = struct {
    up: []u32 = &.{},
    down: []u32 = &.{},
    /// The arenas of the threads that read a layer's experts (a share's cuts, decoded scales), kept from one
    /// layer to the next so that their pages are not faulted in again.
    arenas: std.ArrayList(std.heap.ArenaAllocator) = .empty,
    /// Whether a layer is placed by the device (`Experts.initPlaced`): set by a loader whose source's bytes stay
    /// where they are until the device is done with them (`glm5/session/pack.zig`'s parts, qwen4's
    /// `model_load.Reading`).
    place: bool = false,
    /// What the host makes for the device to read of a placed layer (Flash Next's csf scales, decoded): two
    /// arenas taken in turn by `keep`, a layer's kept until the layer after next asks, by when `initPlaced`'s
    /// `settle` has waited for the device's reads of it.
    kept: [2]?std.heap.ArenaAllocator = .{ null, null },
    turn: u1 = 0,
    /// `initPlaced`'s: the device's tables, the csf scales decoded, and the flag a plane that does not decode sets.
    device: ?Device = null,
    /// Where set, down's outputs are halved between two ranks (`Swap`).
    swap: ?Swap = null,

    const Device = struct {
        /// i64: each matrix's csf planes [count, 6], its sources [count, 3], what it gathers [count]
        tables: cuda.Buffer,
        /// u8 [count, dims / 2, width / 2]: `Swap`'s, sent and received
        send: ?cuda.Buffer = null,
        recv: ?cuda.Buffer = null,
        /// u8: one matrix's decoded scales [count, window]
        scales: cuda.Buffer,
        /// u32 [1]: 0, or the mark (1 + the matrix) of a plane that did not validate
        failed: cuda.Buffer,
        /// The layer `failed` stands for, for the message.
        layer: usize = 0,

        fn deinit(self: *Device) void {
            if (self.send) |*one| one.deinit();
            if (self.recv) |*one| one.deinit();
            self.tables.deinit();
            self.scales.deinit();
            self.failed.deinit();
        }
    };

    pub fn deinit(self: *Staging, gpa: Allocator) void {
        // a load that failed may have left placements that read `kept`: the loader's stream, as `settle`
        if (self.device != null) cuda.synchronizeStream() catch {};
        gpa.free(self.up);
        gpa.free(self.down);
        for (self.arenas.items) |*arena| arena.deinit();
        self.arenas.deinit(gpa);
        for (&self.kept) |*one| if (one.*) |*arena| arena.deinit();
        if (self.device) |*device| device.deinit();
        self.* = .{};
    }

    /// Waits for the layers placed so far (their sources may then be written over) and says whether every
    /// plane of theirs decoded; a loader calls it after its last layer.
    pub fn settle(self: *Staging) error{ CudaFailure, UnsupportedShape }!void {
        const device = if (self.device) |*one| one else return;
        // the loader's stream alone: under `serve --weights-from-peer` rank 1's next part is received on a stream
        // of its own, which waits for rank 0 to send it, and rank 0 for this rank's next trade of down (`swap`)
        cuda.synchronizeStream() catch return error.CudaFailure;
        var mark: [4]u8 = undefined;
        device.failed.download(&mark) catch return error.CudaFailure;
        const which = std.mem.readInt(u32, &mark, .little);
        if (which == 0) return;
        const names = [_][]const u8{ "gate_proj", "up_proj", "down_proj" };
        std.log.err("layer {d}: an expert's {s} csf_32x2_v1 scales do not decode", .{ device.layer, names[@min(which - 1, 2)] });
        return error.UnsupportedShape;
    }

    /// The next of `kept`, emptied: memory the device may read until the second `initPlaced` after this.
    pub fn keep(self: *Staging, gpa: Allocator) Allocator {
        self.turn +%= 1;
        const slot = &self.kept[self.turn];
        if (slot.*) |*arena| {
            _ = arena.reset(.retain_capacity);
        } else slot.* = .init(gpa);
        return slot.*.?.allocator();
    }

    /// `count` arenas, emptied, their memory kept.
    pub fn arenasOf(self: *Staging, gpa: Allocator, count: usize) Allocator.Error![]std.heap.ArenaAllocator {
        while (self.arenas.items.len < count) try self.arenas.append(gpa, .init(gpa));
        for (self.arenas.items) |*arena| _ = arena.reset(.retain_capacity);
        return self.arenas.items[0..count];
    }

    /// `count` words of `slot`, grown if it is smaller; what they hold is undefined.
    pub fn take(slot: *[]u32, gpa: Allocator, count: usize) Allocator.Error![]u32 {
        if (slot.len < count) {
            gpa.free(slot.*);
            slot.* = &.{};
            slot.* = try gpa.alloc(u32, count);
        }
        return slot.*[0..count];
    }
};

/// Down's outputs halved between the two ranks at load (`Staging.swap`): each reads its half of the outputs, every
/// input of them (half of down's bytes, where it read them whole), and trades with its peer the peer's inputs of
/// them for its own inputs of the peer's outputs. Down is half of a rank's routed bytes on disk; the trade a
/// quarter of them, over the ranks' link.
pub const Swap = struct {
    rank: usize,
    context: *const anyopaque,
    /// Sends `send` to the other rank and receives the other's into `recv` (u8, as long), in the stream's order.
    trade: *const fn (context: *const anyopaque, send: cuda.View, recv: cuda.View) cuda.Error!void,
};

/// One projection of every expert as the checkpoint stores it, an entry an expert.
pub const Matrix = struct {
    /// each [n, k/2]: two E2M1 codes a byte, the lower input in the low nibble
    codes: []const []const u8,
    /// each e4m3 [n, k/16]
    block_scales: []const []const u8,
    /// each expert's factor (ModelOpt's weight_scale_2)
    scales: []const f32,
    /// each expert's activation scale (ModelOpt's input_scale), read by `Activations.nvfp4` alone
    input_scales: []const f32 = &.{},

    /// `count` equal parts of `bytes`, for experts stored as one array.
    pub fn parts(gpa: Allocator, bytes: []const u8, count: usize) Allocator.Error![]const []const u8 {
        const out = try gpa.alloc([]const u8, count);
        const each = bytes.len / count;
        for (out, 0..) |*part, e| part.* = bytes[e * each ..][0..each];
        return out;
    }

    /// Rank `rank` of `of`'s outputs of experts [n, k] (a share's `gate` and `up`): rows `rank * n / of`
    /// onward, slices of these bytes. The experts' scales stay theirs.
    pub fn rowsOf(self: Matrix, arena: Allocator, n: usize, rank: usize, of: usize) Allocator.Error!Matrix {
        std.debug.assert(n % of == 0);
        const codes = try arena.alloc([]const u8, self.codes.len);
        const scales = try arena.alloc([]const u8, self.block_scales.len);
        for (self.codes, self.block_scales, codes, scales) |code, scale, *c, *s| {
            c.* = code[rank * (code.len / of) ..][0 .. code.len / of];
            s.* = scale[rank * (scale.len / of) ..][0 .. scale.len / of];
        }
        return .{ .codes = codes, .block_scales = scales, .scales = self.scales, .input_scales = self.input_scales };
    }

    /// Rank `rank` of `of`'s inputs of experts [n, k] (a share's `down`): each row's `k / of` codes and their
    /// block scales, copied; `k / of` is whole blocks of 16.
    pub fn colsOf(self: Matrix, arena: Allocator, n: usize, k: usize, rank: usize, of: usize) Allocator.Error!Matrix {
        std.debug.assert(k % (16 * of) == 0);
        const code_row = k / 2;
        const scale_row = k / 16;
        const half_code = code_row / of;
        const half_scale = scale_row / of;
        const codes = try arena.alloc([]const u8, self.codes.len);
        const scales = try arena.alloc([]const u8, self.block_scales.len);
        for (self.codes, self.block_scales, codes, scales) |code, scale, *c, *s| {
            const code_out = try arena.alloc(u8, n * half_code);
            const scale_out = try arena.alloc(u8, n * half_scale);
            for (0..n) |r| {
                @memcpy(code_out[r * half_code ..][0..half_code], code[r * code_row + rank * half_code ..][0..half_code]);
                @memcpy(scale_out[r * half_scale ..][0..half_scale], scale[r * scale_row + rank * half_scale ..][0..half_scale]);
            }
            c.* = code_out;
            s.* = scale_out;
        }
        return .{ .codes = codes, .block_scales = scales, .scales = self.scales, .input_scales = self.input_scales };
    }
};

/// Writes matrix `m` of `per` into blocks [experts, n/32, k/32, per, 144]: a block's 128 code words by lane
/// (output quad, input quad) and n8 tile, then its 64 block scales.
pub fn pack(matrix: Matrix, experts: usize, n: usize, k: usize, out: []u32, per: usize, m: usize) void {
    std.debug.assert(matrix.codes.len == experts and matrix.block_scales.len == experts);
    for (matrix.codes, matrix.block_scales) |codes, scales| std.debug.assert(codes.len == n * k / 2 and scales.len == n * k / 16);
    // an expert's blocks are its own, so experts pack in parallel; the mapped checkpoint pages fault in there too
    const workers = @min(experts, std.Thread.getCpuCount() catch 1);
    var threads: [64]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..@min(workers, threads.len) - 1) |w| {
        threads[spawned] = std.Thread.spawn(.{}, packShare, .{ matrix, experts, n, k, out, per, m, w + 1, workers }) catch break;
        spawned += 1;
    }
    // the shares no thread took, this thread's own included
    packShare(matrix, experts, n, k, out, per, m, 0, workers);
    for (spawned + 1..workers) |w| packShare(matrix, experts, n, k, out, per, m, w, workers);
    for (threads[0..spawned]) |thread| thread.join();
}

/// Experts `share`, `share + of`, ... of `pack`.
fn packShare(matrix: Matrix, experts: usize, n: usize, k: usize, out: []u32, per: usize, m: usize, share: usize, of: usize) void {
    const nb = n / cols;
    const kg = k / 32;
    var e = share;
    while (e < experts) : (e += of) for (0..nb) |cb| for (0..kg) |g| {
        const block = out[(((e * nb + cb) * kg + g) * per + m) * block_words ..][0..block_words];
        for (0..8) |gq| for (0..4) |t| for (0..4) |j| {
            const output = cb * cols + j * 8 + gq;
            var word: u32 = 0;
            for (0..2) |h| for (0..4) |q| {
                const input = g * 32 + h * 16 + t * 4 + q;
                const byte = matrix.codes[e][output * (k / 2) + input / 2];
                const code: u32 = if (input % 2 == 0) byte & 0xf else byte >> 4;
                word |= code << @intCast(4 * (2 * h + q / 2 + 4 * (q % 2)));
            };
            block[(gq * 4 + t) * 4 + j] = word;
        };
        const scales: *[64]u8 = @ptrCast(block[128..]);
        for (0..4) |t| for (0..2) |h| for (0..4) |j| for (0..2) |c| {
            const output = cb * cols + j * 8 + t * 2 + c;
            scales[((t * 2 + h) * 4 + j) * 2 + c] = matrix.block_scales[e][output * (k / 16) + g * 2 + h];
        };
    };
}

/// Writes matrix `m` of `per` into blocks [experts, n/32, ceil(k/64), per, 288] for the FP4 mma (`zig/kernels/nvfp4/
/// experts_a4.cuh`): a (32 columns, 64 inputs) block is 72 uint4, lane l = (q, c)'s B registers of n8 tiles 0
/// and 1 at uint4 l and of tiles 2 and 3 at 32 + l (b0: column 8 j + q's stored bytes 4 c .. 4 c + 3 of the step,
/// inputs 8 c .. 8 c + 7; b1: the bytes 16 on, inputs 32 + 8 c ..), then at 64 + q the four e4m3 scales of
/// columns q, 8 + q, 16 + q and 24 + q, a word each. The checkpoint's bytes as they are, only placed.
pub fn packA4(matrix: Matrix, experts: usize, n: usize, k: usize, out: []u32, per: usize, m: usize) void {
    std.debug.assert(matrix.codes.len == experts and matrix.block_scales.len == experts);
    for (matrix.codes, matrix.block_scales) |codes, scales| std.debug.assert(codes.len == n * k / 2 and scales.len == n * k / 16);
    const workers = @min(experts, std.Thread.getCpuCount() catch 1);
    var threads: [64]std.Thread = undefined;
    var spawned: usize = 0;
    for (0..@min(workers, threads.len) - 1) |w| {
        threads[spawned] = std.Thread.spawn(.{}, packA4Share, .{ matrix, experts, n, k, out, per, m, w + 1, workers }) catch break;
        spawned += 1;
    }
    packA4Share(matrix, experts, n, k, out, per, m, 0, workers);
    for (spawned + 1..workers) |w| packA4Share(matrix, experts, n, k, out, per, m, w, workers);
    for (threads[0..spawned]) |thread| thread.join();
}

/// Experts `share`, `share + of`, ... of `packA4`.
fn packA4Share(matrix: Matrix, experts: usize, n: usize, k: usize, out: []u32, per: usize, m: usize, share: usize, of: usize) void {
    const words = 2 * block_words;
    const nb = n / cols;
    const kg = (k + 63) / 64;
    var e = share;
    while (e < experts) : (e += of) for (0..nb) |cb| for (0..kg) |g| {
        const block = out[(((e * nb + cb) * kg + g) * per + m) * words ..][0..words];
        const codes = matrix.codes[e];
        const scales = matrix.block_scales[e];
        for (0..8) |q| for (0..4) |c| for (0..4) |j| {
            const column = cb * cols + j * 8 + q;
            const at = column * (k / 2) + g * 32 + 4 * c;
            const lane = q * 4 + c;
            const base = (if (j < 2) lane else 32 + lane) * 4 + (j & 1) * 2;
            block[base] = if (g * 32 + 4 * c < k / 2) std.mem.readInt(u32, codes[at..][0..4], .little) else 0;
            block[base + 1] = if (g * 32 + 4 * c + 16 < k / 2) std.mem.readInt(u32, codes[at + 16 ..][0..4], .little) else 0;
        };
        for (0..8) |q| for (0..4) |j| {
            const column = cb * cols + j * 8 + q;
            const first = g * 4;
            const offset = column * (k / 16) + first;
            var word: u32 = 0;
            if (first + 4 <= k / 16) {
                word = std.mem.readInt(u32, scales[offset..][0..4], .little);
            } else {
                for (0..4) |b| if (first + b < k / 16) {
                    word |= @as(u32, scales[offset + b]) << @intCast(8 * b);
                };
            }
            block[(64 + q) * 4 + j] = word;
        };
    };
}

//! Compares a forward with reference tensors exported from TensorFold (tools/reference/flash_next_export.py):
//! at each point the device's rows against the reference's, then, unless running free, the reference's rows are
//! written over them so the next block starts from TensorFold's input and a point's error is that block's alone.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const forward = @import("forward.zig");
const numeric = @import("../../weights/numeric.zig");
const safetensors = @import("../../weights/safetensors.zig");

pub const Stats = struct {
    /// Largest |mine - reference|.
    max_abs: f64 = 0,
    /// rms(mine - reference) / rms(reference), the worst layer's.
    relative: f64 = 0,
    worst_layer: usize = 0,
    /// Elements that differ at all, for integer points.
    mismatched: u64 = 0,
    elements: u64 = 0,
    layers: usize = 0,
};

/// Every point's distance from the reference row by row, which `Stats` folds into one number a point: where a
/// free-running forward leaves TensorFold's, by row and by layer.
pub const ByRow = struct {
    rows: usize,
    /// The model's layers; a point after them (`final_h`, `logits`) is kept at this index.
    layers: usize,
    /// [points, layers + 1, rows]: rms(mine - reference) / rms(reference) of a row; NaN where nothing was observed.
    relative: []f64,
    /// [rows] each: the first largest logit of a row, here and in the reference.
    mine_top: []u32,
    reference_top: []u32,
    /// [rows] each: a row's sums while a point is taken.
    diff: []f64,
    ref: []f64,
    gpa: Allocator,

    const points = std.enums.values(forward.Point).len;

    pub fn init(gpa: Allocator, rows: usize, layers: usize) !ByRow {
        const relative = try gpa.alloc(f64, points * (layers + 1) * rows);
        errdefer gpa.free(relative);
        @memset(relative, std.math.nan(f64));
        const tops = try gpa.alloc(u32, 2 * rows);
        errdefer gpa.free(tops);
        @memset(tops, 0);
        const sums = try gpa.alloc(f64, 2 * rows);
        return .{ .rows = rows, .layers = layers, .relative = relative, .mine_top = tops[0..rows], .reference_top = tops[rows..], .diff = sums[0..rows], .ref = sums[rows..], .gpa = gpa };
    }

    pub fn deinit(self: *ByRow) void {
        self.gpa.free(self.relative);
        self.gpa.free(self.mine_top.ptr[0 .. 2 * self.rows]);
        self.gpa.free(self.diff.ptr[0 .. 2 * self.rows]);
        self.* = undefined;
    }

    /// The rows of `point` at `layer`: [rows].
    pub fn get(self: ByRow, point: forward.Point, layer: usize) []f64 {
        return self.relative[(@as(usize, @backingInt(point)) * (self.layers + 1) + layer) * self.rows ..][0..self.rows];
    }
};

/// A forward against this forward: the unperturbed run's last streams and logits, kept on the host, and how far a
/// later run's rows are from them.
pub const Own = struct {
    /// The run in hand is the one to keep.
    record: bool = true,
    /// bf16 bits [rows, streams * hidden] and [rows, vocab].
    final: []u16,
    logits: []u16,
    /// [rows] each: a later run's relative rms from the kept one.
    final_relative: []f64,
    logits_relative: []f64,
    gpa: Allocator,

    pub fn init(gpa: Allocator, rows: usize, wide: usize, vocab: usize) !Own {
        const final = try gpa.alloc(u16, rows * wide);
        errdefer gpa.free(final);
        const logits = try gpa.alloc(u16, rows * vocab);
        errdefer gpa.free(logits);
        const relative = try gpa.alloc(f64, 2 * rows);
        return .{ .final = final, .logits = logits, .final_relative = relative[0..rows], .logits_relative = relative[rows..], .gpa = gpa };
    }

    pub fn deinit(self: *Own) void {
        self.gpa.free(self.final);
        self.gpa.free(self.logits);
        self.gpa.free(self.final_relative.ptr[0 .. 2 * self.final_relative.len]);
        self.* = undefined;
    }

    fn take(self: *Own, point: forward.Point, mine: []const u8) void {
        const kept = switch (point) {
            .final_h => self.final,
            .logits => self.logits,
            else => return,
        };
        const bits: []align(1) const u16 = @ptrCast(mine);
        std.debug.assert(bits.len == kept.len);
        if (self.record) return @memcpy(kept, bits);
        const out = if (point == .final_h) self.final_relative else self.logits_relative;
        const per_row = kept.len / out.len;
        for (out, 0..) |*relative, row| {
            var diff_squares: f64 = 0;
            var ref_squares: f64 = 0;
            for (bits[row * per_row ..][0..per_row], kept[row * per_row ..][0..per_row]) |a, b| {
                const reference: f64 = numeric.f32FromBf16(b);
                const d = @as(f64, numeric.f32FromBf16(a)) - reference;
                diff_squares += d * d;
                ref_squares += reference * reference;
            }
            relative.* = if (ref_squares > 0) @sqrt(diff_squares / ref_squares) else @sqrt(diff_squares);
        }
    }
};

/// Noise added to the streams a layer starts from, to see what a difference of that size becomes downstream.
pub const Perturb = struct {
    /// The noise's rms over the rms of the row it is added to, before the sum is rounded to bf16.
    relative: f64,
    seed: u64,
    /// The layer whose `h_in` takes it.
    layer: usize = 0,
    /// The row that takes it; null: every row.
    row: ?usize = null,
    /// Mixed into the seed, so each window of a sequence draws its own noise (`score`); 0 leaves the seed alone.
    window: u64 = 0,
    /// What the rounding left: rms(perturbed - original) / rms(original) over the rows that took it, every call's.
    achieved: f64 = 0,
    diff_squares: f64 = 0,
    all_squares: f64 = 0,

    /// `SIZE[@LAYER[:ROW]]`, as `compare` and `score` take it after `perturb=`.
    pub fn parse(given: []const u8) !Perturb {
        var size = given;
        var perturb: Perturb = .{ .relative = 0, .seed = 0 };
        if (std.mem.indexOfScalar(u8, size, '@')) |at| {
            var layer = size[at + 1 ..];
            size = size[0..at];
            if (std.mem.indexOfScalar(u8, layer, ':')) |colon| {
                perturb.row = try std.fmt.parseInt(usize, layer[colon + 1 ..], 10);
                layer = layer[0..colon];
            }
            perturb.layer = try std.fmt.parseInt(usize, layer, 10);
        }
        perturb.relative = try std.fmt.parseFloat(f64, size);
        return perturb;
    }

    pub fn apply(self: *Perturb, bytes: []u8, rows: usize) void {
        const bits: []align(1) u16 = @ptrCast(bytes);
        const per_row = bits.len / rows;
        var prng: std.Random.DefaultPrng = .init(self.seed +% self.window *% 0x9E3779B97F4A7C15);
        const random = prng.random();
        for (0..rows) |row| {
            if (self.row) |only| if (row != only) continue;
            const values = bits[row * per_row ..][0..per_row];
            var squares: f64 = 0;
            for (values) |b| squares += @as(f64, numeric.f32FromBf16(b)) * numeric.f32FromBf16(b);
            const rms = @sqrt(squares / @as(f64, @floatFromInt(per_row)));
            for (values) |*b| {
                const before = numeric.f32FromBf16(b.*);
                b.* = numeric.bf16FromF32(before + @as(f32, @floatCast(self.relative * rms)) * random.floatNorm(f32));
                const d = @as(f64, numeric.f32FromBf16(b.*)) - before;
                self.diff_squares += d * d;
            }
            self.all_squares += squares;
        }
        self.achieved = if (self.all_squares > 0) @sqrt(self.diff_squares / self.all_squares) else 0;
    }
};

pub const Comparison = struct {
    reference: *const safetensors.File,
    /// Slots a row: the routed picks, then the shared expert.
    slots: usize,
    /// Overwrite each point with the reference after comparing it.
    force: bool,
    /// The points are the MTP head's (`NAME.mtp.*.safetensors`), not the main model's.
    mtp: bool = false,
    /// The model holds only its first layers, the reference all of them. What the last write-back leaves
    /// (`final_h`) is then compared with the reference's input of the layer after them, and the mixer and the
    /// head start from the reference's own last streams (`final.h`), so every point is still one block's from
    /// TensorFold's input. Running free, the mixer's read-out and the logits have no reference and are not
    /// compared.
    partial: bool = false,
    stats: std.EnumArray(forward.Point, Stats) = .initFill(.{}),
    scratch: std.ArrayList(u8) = .empty,
    /// Slots whose pick differs from the reference's, and the widest gap, in the reference's router logits,
    /// between the two experts at such a slot: a pick can only differ where the logits nearly tie.
    picks_differing: usize = 0,
    widest_pick_gap: f64 = 0,
    /// Takes every point row by row too.
    by_row: ?*ByRow = null,
    /// Keeps this forward's last streams and logits, or compares them with the kept ones.
    own: ?*Own = null,
    /// Added to one layer's `h_in` before it is compared.
    perturb: ?*Perturb = null,
    gpa: Allocator,

    pub fn deinit(self: *Comparison) void {
        self.scratch.deinit(self.gpa);
    }

    pub fn tap(self: *Comparison) forward.Tap {
        return .{ .context = self, .at = at };
    }

    fn name(self: *const Comparison, buffer: []u8, point: forward.Point, layer: usize) ![]const u8 {
        if (self.mtp) {
            // the head's one layer and what follows it, as the exporter's `--mtp` names them
            var plain_buffer: [96]u8 = undefined;
            const main = try plain(&plain_buffer, point, 0);
            return switch (point) {
                .final_h => "mtp.streams",
                .final_mixed => "mtp.mixed",
                .logits => "mtp.logits",
                .embed_h, .h_after_ple => error.MissingReference,
                else => rename: {
                    break :rename std.fmt.bufPrint(buffer, "mtp.layer{s}", .{main["layers.0".len..]});
                },
            };
        }
        if (self.partial and point == .final_h) return plain(buffer, .h_in, layer);
        return plain(buffer, point, layer);
    }

    pub fn plain(buffer: []u8, point: forward.Point, layer: usize) ![]const u8 {
        return switch (point) {
            .embed_h => "embed.h",
            .final_h => "final.h",
            .final_mixed => "final.mixed",
            .logits => "logits",
            .h_after_ple => std.fmt.bufPrint(buffer, "layers.{d}.h_after_ple", .{layer}),
            .h_in => std.fmt.bufPrint(buffer, "layers.{d}.h_in", .{layer}),
            .attn_mixed => std.fmt.bufPrint(buffer, "layers.{d}.attn_hc.mixed", .{layer}),
            .attn_inject => std.fmt.bufPrint(buffer, "layers.{d}.attn_hc.inject", .{layer}),
            .projection => std.fmt.bufPrint(buffer, "layers.{d}.block.projection", .{layer}),
            .block_out => std.fmt.bufPrint(buffer, "layers.{d}.block.out", .{layer}),
            .branch => std.fmt.bufPrint(buffer, "layers.{d}.block.branch", .{layer}),
            .h_mid => std.fmt.bufPrint(buffer, "layers.{d}.h_mid", .{layer}),
            .mlp_mixed => std.fmt.bufPrint(buffer, "layers.{d}.mlp_hc.mixed", .{layer}),
            .mlp_inject => std.fmt.bufPrint(buffer, "layers.{d}.mlp_hc.inject", .{layer}),
            .moe_logits => std.fmt.bufPrint(buffer, "layers.{d}.moe.logits", .{layer}),
            .moe_picks => std.fmt.bufPrint(buffer, "layers.{d}.moe.picks", .{layer}),
            .moe_weights => std.fmt.bufPrint(buffer, "layers.{d}.moe.weights", .{layer}),
            .moe_y, .moe_shared => std.fmt.bufPrint(buffer, "layers.{d}.moe.y", .{layer}),
        };
    }

    fn at(context: *anyopaque, point: forward.Point, layer: usize, view: cuda.View) anyerror!void {
        const self: *Comparison = @ptrCast(@alignCast(context));
        if (self.partial and !self.force and (point == .final_mixed or point == .logits)) return;
        var name_buffer: [96]u8 = undefined;
        const tensor = self.reference.get(try self.name(&name_buffer, point, layer)) orelse return error.MissingReference;
        const width = view.dtype.size();
        const count = view.count();
        try self.scratch.resize(self.gpa, count * width);
        const mine = self.scratch.items;
        try cuda.synchronize();
        try cuda.download(view, mine);
        if (self.perturb) |noise| if (point == .h_in and layer == noise.layer) {
            noise.apply(mine, @intCast(view.size[0]));
            try cuda.upload(view, mine);
        };
        if (self.own) |own| own.take(point, mine);

        // the reference holds a row's slots together; here the shared expert's rows live apart from the routed
        // slots, whose last slot is never written
        var theirs = tensor.data;
        var owned: ?[]u8 = null;
        defer if (owned) |bytes| self.gpa.free(bytes);
        var skip_slot: ?usize = null;
        if (point == .moe_shared) {
            const dims = count / @as(usize, @intCast(view.size[0]));
            const rows: usize = @intCast(view.size[0]);
            const picked = try self.gpa.alloc(u8, count * width);
            owned = picked;
            for (0..rows) |r| @memcpy(picked[r * dims * width ..][0 .. dims * width], tensor.data[(r * self.slots + self.slots - 1) * dims * width ..][0 .. dims * width]);
            theirs = picked;
        } else if (point == .moe_y) {
            skip_slot = self.slots - 1;
        }
        if (theirs.len != mine.len or tensor.dtype.size() != width) return error.ReferenceShape;

        if (point == .moe_picks) {
            var logits_name: [96]u8 = undefined;
            const logits = self.reference.get(try self.name(&logits_name, .moe_logits, layer)) orelse return error.MissingReference;
            const per_row = logits.data.len / 4 / @as(usize, @intCast(view.size[0]));
            for (0..count) |i| {
                const a = std.mem.readInt(i32, mine[i * 4 ..][0..4], .little);
                const b = std.mem.readInt(i32, theirs[i * 4 ..][0..4], .little);
                if (a == b) continue;
                const row = i / self.slots;
                const la = value(.f32, logits.data[(row * per_row + @as(usize, @intCast(a))) * 4 ..][0..4]);
                const lb = value(.f32, logits.data[(row * per_row + @as(usize, @intCast(b))) * 4 ..][0..4]);
                self.picks_differing += 1;
                self.widest_pick_gap = @max(self.widest_pick_gap, @abs(la - lb));
            }
        }

        const stats = self.stats.getPtr(point);
        var diff_squares: f64 = 0;
        var ref_squares: f64 = 0;
        var max_abs: f64 = 0;
        const per_slot = if (skip_slot != null) count / (@as(usize, @intCast(view.size[0]))) else 0;
        // a view's first axis is the window's rows, or rows * slots
        const per_row = if (self.by_row) |by| count / by.rows else 0;
        if (self.by_row) |by| {
            std.debug.assert(count % by.rows == 0);
            @memset(by.diff, 0);
            @memset(by.ref, 0);
        }
        for (0..count) |i| {
            if (skip_slot) |slot| {
                if ((i / per_slot) % self.slots == slot) continue;
            }
            const a = value(view.dtype, mine[i * width ..][0..width]);
            const b = value(view.dtype, theirs[i * width ..][0..width]);
            if (a != b) stats.mismatched += 1;
            const d = a - b;
            diff_squares += d * d;
            ref_squares += b * b;
            max_abs = @max(max_abs, @abs(d));
            stats.elements += 1;
            if (self.by_row) |by| {
                by.diff[i / per_row] += d * d;
                by.ref[i / per_row] += b * b;
            }
        }
        if (self.by_row) |by| {
            for (by.get(point, @min(layer, by.layers)), by.diff, by.ref) |*relative, diff, ref| relative.* = if (ref > 0) @sqrt(diff / ref) else @sqrt(diff);
            if (point == .logits) for (by.mine_top, by.reference_top, 0..) |*mine_top, *reference_top, row| {
                mine_top.* = first(view.dtype, mine[row * per_row * width ..][0 .. per_row * width]);
                reference_top.* = first(view.dtype, theirs[row * per_row * width ..][0 .. per_row * width]);
            };
        }
        const relative = if (ref_squares > 0) @sqrt(diff_squares / ref_squares) else @sqrt(diff_squares);
        if (relative >= stats.relative) {
            stats.relative = relative;
            stats.worst_layer = layer;
        }
        stats.max_abs = @max(stats.max_abs, max_abs);
        stats.layers += 1;

        if (!self.force) return;
        if (self.partial and point == .final_h) {
            const final = self.reference.get("final.h") orelse return error.MissingReference;
            if (final.data.len != mine.len) return error.ReferenceShape;
            return cuda.upload(view, final.data);
        }
        try cuda.upload(view, theirs);
    }
};

/// The index of a row's first largest value, as torch's argmax and `forward.top` take it.
fn first(dtype: cuda.DType, row: []const u8) u32 {
    const width = dtype.size();
    var best: usize = 0;
    var largest = -std.math.inf(f64);
    for (0..row.len / width) |i| {
        const v = value(dtype, row[i * width ..][0..width]);
        if (v > largest) {
            largest = v;
            best = i;
        }
    }
    return @intCast(best);
}

fn value(dtype: cuda.DType, bytes: []const u8) f64 {
    return switch (dtype) {
        .bf16 => numeric.f32FromBf16(std.mem.readInt(u16, bytes[0..2], .little)),
        .f32 => @as(f32, @bitCast(std.mem.readInt(u32, bytes[0..4], .little))),
        .i32 => @floatFromInt(std.mem.readInt(i32, bytes[0..4], .little)),
        else => unreachable,
    };
}

/// A reference file's image prompt (spec 019; `flash_next_export.py --vision` writes `vision.features` bf16
/// [placeholders, hidden], `vision.rows` i32, `vision.positions` i32 [3, rows] and `vision.rope_delta`), on the
/// device and ready for `forward.State.attach`. TensorFold's own features: the bits are then the forward's alone.
pub const ReferenceImage = struct {
    features: cuda.Buffer,
    rows: []u32,
    /// [rows, 3]: the exporter's [3, rows] transposed.
    positions: [][3]i32,
    delta: i32,

    /// Null where the file holds no image.
    pub fn read(gpa: Allocator, reference: *const safetensors.File) !?ReferenceImage {
        const features = reference.get("vision.features") orelse return null;
        const rows = reference.get("vision.rows") orelse return error.MissingReference;
        const positions = reference.get("vision.positions") orelse return error.MissingReference;
        const delta = reference.get("vision.rope_delta") orelse return error.MissingReference;
        if (features.dtype != .BF16 or features.shape.len != 2 or rows.dtype != .I32 or rows.count() != features.shape[0] or
            positions.dtype != .I32 or positions.shape.len != 2 or positions.shape[0] != 3 or delta.dtype != .I32 or delta.count() != 1)
            return error.MissingReference;
        const n: usize = @intCast(positions.shape[1]);
        const out_rows = try gpa.alloc(u32, @intCast(rows.count()));
        errdefer gpa.free(out_rows);
        for (out_rows, 0..) |*row, i| row.* = @intCast(std.mem.readInt(i32, rows.data[4 * i ..][0..4], .little));
        const out_positions = try gpa.alloc([3]i32, n);
        errdefer gpa.free(out_positions);
        for (out_positions, 0..) |*p, i| {
            for (p, 0..) |*axis, a| axis.* = std.mem.readInt(i32, positions.data[4 * (a * n + i) ..][0..4], .little);
        }
        var buffer: cuda.Buffer = try .init(.bf16, &.{ @intCast(features.shape[0]), @intCast(features.shape[1]) });
        errdefer buffer.deinit();
        try buffer.upload(features.data);
        return .{ .features = buffer, .rows = out_rows, .positions = out_positions, .delta = std.mem.readInt(i32, delta.data[0..4], .little) };
    }

    /// Before the prompt's first row; the image stays this one's until the state lets it go.
    pub fn attach(self: *const ReferenceImage, state: *forward.State) !void {
        return state.attach(.{ .features = self.features.view, .rows = self.rows }, self.positions, self.delta);
    }

    pub fn deinit(self: *ReferenceImage, gpa: Allocator) void {
        self.features.deinit();
        gpa.free(self.rows);
        gpa.free(self.positions);
        self.* = undefined;
    }
};

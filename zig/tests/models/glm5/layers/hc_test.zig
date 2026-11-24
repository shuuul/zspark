//! `hc.zig`'s tests: each kernel against `glue_ref.zig`'s steps, bit for bit where no `exp` stands between them,
//! and a row alone against the row among others.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const hc = @import("../../../../src/models/glm5/layers/hc.zig");
const host = @import("../../../../src/models/glm5/layers/glue_ref.zig");
const shared = @import("../testing.zig");
const device = shared.device;
const fetch = shared.fetch;

/// Fills `row` so that its RMSNorm says how its squares were summed: a 4096 under whose square the ones beside it
/// are lost when added one by one, small values of every mantissa, and one value swept until a quotient's
/// rounding falls between the folded sum's and a serial sum's. `gain` is what the read-out's collapse multiplies
/// a value by before it rounds it (1 for RMSNorm alone); `scratch` is the fold's power of two.
fn tellingRow(row: []u16, gain: f32, scratch: []f32) !void {
    const dims = row.len;
    @memset(row, numeric.bf16FromF32(1));
    row[0] = numeric.bf16FromF32(4096);
    for (0..128) |j| row[dims - 1 - j] = numeric.bf16FromF32(@as(f32, @floatFromInt(128 + j)) / (128.0 * 1024.0));
    const count: f32 = @floatFromInt(dims);
    for (1..256) |k| {
        row[1] = numeric.bf16FromF32(@floatFromInt(4 * k));
        var serial: f32 = 0;
        @memset(scratch, 0);
        for (scratch[0..dims], row) |*sq, bits| {
            const v = numeric.roundBf16(gain * numeric.f32FromBf16(bits));
            sq.* = v * v;
            serial += sq.*;
        }
        const folded = 1.0 / @sqrt(host.fold(scratch) / count + Case.eps);
        const plain = 1.0 / @sqrt(serial / count + Case.eps);
        for (row) |bits| {
            const v = numeric.roundBf16(gain * numeric.f32FromBf16(bits));
            if (numeric.roundBf16(v * folded) != numeric.roundBf16(v * plain)) return;
        }
    }
    return error.NoTellingRow;
}

/// A site's inputs for `rows` rows of `dims`, on the host and the device.
const Case = struct {
    rows: usize,
    dims: usize,
    x: []u16,
    dots: []u16,
    base: []f32,
    scale: [3]f32,
    norm: []u16,
    x_dev: cuda.Buffer,
    site: hc.Site,
    held: [4]cuda.Buffer,

    const eps = 1e-5;
    // the model's is 1e-6; larger here, so that a round without it stands out of exp's last bits
    const hc_eps = 1e-3;
    const iters = 20;

    fn init(gpa: std.mem.Allocator, seed: u64, rows: usize, dims: usize) !Case {
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        const wide = hc.streams * dims;
        const x = try gpa.alloc(u16, rows * wide);
        errdefer gpa.free(x);
        // streams that differ in size, as a residual's do after a few layers
        for (0..rows) |r| for (0..hc.streams) |s| shared.randomBf16(random, x[(r * hc.streams + s) * dims ..][0..dims], @as(f32, @floatFromInt(1 + s * r % 5)));
        const dots = try gpa.alloc(u16, host.mix * wide);
        errdefer gpa.free(dots);
        shared.randomBf16(random, dots, 0.02);
        const base = try gpa.alloc(f32, host.mix);
        errdefer gpa.free(base);
        shared.randomF32(random, base, 1.5);
        const norm = try gpa.alloc(u16, dims);
        errdefer gpa.free(norm);
        shared.randomBf16(random, norm, 1.0);
        const scale = [3]f32{ 0.8, 1.7, 2.9 };
        var x_dev = try device(u16, .bf16, &.{ rows, wide }, x);
        errdefer x_dev.deinit();
        var held: [4]cuda.Buffer = undefined;
        held[0] = try device(u16, .bf16, &.{ host.mix, wide }, dots);
        errdefer held[0].deinit();
        held[1] = try device(f32, .f32, &.{host.mix}, base);
        errdefer held[1].deinit();
        held[2] = try device(f32, .f32, &.{3}, &scale);
        errdefer held[2].deinit();
        held[3] = try device(u16, .bf16, &.{dims}, norm);
        return .{
            .rows = rows,
            .dims = dims,
            .x = x,
            .dots = dots,
            .base = base,
            .scale = scale,
            .norm = norm,
            .x_dev = x_dev,
            .site = .{ .dots = held[0].view, .base = held[1].view, .scale = held[2].view, .norm = held[3].view },
            .held = held,
        };
    }

    fn deinit(case: *Case, gpa: std.mem.Allocator) void {
        for (&case.held) |*buffer| buffer.deinit();
        case.x_dev.deinit();
        gpa.free(case.norm);
        gpa.free(case.base);
        gpa.free(case.dots);
        gpa.free(case.x);
    }

    fn hostSite(case: *const Case) host.Site {
        return .{ .base = case.base, .scale = &case.scale, .norm = .{ .bf16 = case.norm }, .eps = eps, .hc_eps = hc_eps, .iters = iters };
    }
};

test "the mix dots and the squares are the host's bits, at one step a block and at eight, a row alone or among others" {
    const gpa = testing.allocator;
    for ([_]usize{ 512, 4096 }, [_]usize{ 5, 3 }) |dims, rows| {
        var case: Case = try .init(gpa, dims, rows, dims);
        defer case.deinit(gpa);
        const wide = hc.streams * dims;
        var part: cuda.Buffer = try .init(.f32, &hc.partShape(rows));
        defer part.deinit();
        try hc.mix(case.x_dev.view, case.site.dots, part.view);
        const got = try fetch(u32, gpa, part);
        defer gpa.free(got);
        const row = host.parts * host.entry;
        var want: [host.parts * host.entry]f32 = undefined;
        var alone: cuda.Buffer = try .init(.f32, &hc.partShape(1));
        defer alone.deinit();
        for (0..rows) |r| {
            host.hcMix(case.x[r * wide ..][0..wide], case.dots, &want);
            for (0..host.parts) |b| {
                const at = b * host.entry;
                try testing.expectEqualSlices(u32, @ptrCast(want[at..][0 .. host.mix + 1]), got[r * row + at ..][0 .. host.mix + 1]);
            }
            try hc.mix(case.x_dev.span(r, 1), case.site.dots, alone.view);
            const one = try fetch(u32, gpa, alone);
            defer gpa.free(one);
            for (0..host.parts) |b| {
                const at = b * host.entry;
                try testing.expectEqualSlices(u32, got[r * row + at ..][0 .. host.mix + 1], one[at..][0 .. host.mix + 1]);
            }
        }
    }
}

test "a read-out is the host's within a rounding, its gates within fp32's, and a row alone has the window's bits" {
    const gpa = testing.allocator;
    for ([_]usize{ 512, 4096 }, [_]usize{ 6, 3 }) |dims, rows| {
        var case: Case = try .init(gpa, 7 + dims, rows, dims);
        defer case.deinit(gpa);
        const wide = hc.streams * dims;
        var part: cuda.Buffer = try .init(.f32, &hc.partShape(rows));
        defer part.deinit();
        var out: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
        defer out.deinit();
        var post: cuda.Buffer = try .init(.f32, &.{ rows, hc.streams });
        defer post.deinit();
        var comb: cuda.Buffer = try .init(.f32, &.{ rows, hc.streams * hc.streams });
        defer comb.deinit();
        try hc.mix(case.x_dev.view, case.site.dots, part.view);
        try hc.pre(case.x_dev.view, part.view, case.site, out.view, post.view, comb.view, Case.eps, Case.hc_eps, Case.iters);
        const got = try fetch(u16, gpa, out);
        defer gpa.free(got);
        const got_post = try fetch(f32, gpa, post);
        defer gpa.free(got_post);
        const got_comb = try fetch(f32, gpa, comb);
        defer gpa.free(got_comb);

        const scratch = try gpa.alloc(f32, dims);
        defer gpa.free(scratch);
        const want = try gpa.alloc(u16, dims);
        defer gpa.free(want);
        var want_part: [host.parts * host.entry]f32 = undefined;
        var want_post: [4]f32 = undefined;
        var want_comb: [16]f32 = undefined;
        var far: usize = 0;
        for (0..rows) |r| {
            host.hcMix(case.x[r * wide ..][0..wide], case.dots, &want_part);
            host.hcPre(case.x[r * wide ..][0..wide], &want_part, case.hostSite(), want, &want_post, &want_comb, scratch);
            for (want_post, got_post[r * 4 ..][0..4]) |w, g| try testing.expectApproxEqRel(w, g, 1e-5);
            for (want_comb, got_comb[r * 16 ..][0..16]) |w, g| try testing.expectApproxEqAbs(w, g, 2e-6);
            for (want, got[r * dims ..][0..dims]) |w, g| {
                try shared.expectClose(w, g);
                far += @intFromBool(w != g);
            }
        }
        // a gate an fp32 step off moves few of the collapsed values over a rounding
        try testing.expect(far * 20 < rows * dims);

        var part1: cuda.Buffer = try .init(.f32, &hc.partShape(1));
        defer part1.deinit();
        var out1: cuda.Buffer = try .init(.bf16, &.{ 1, dims });
        defer out1.deinit();
        var post1: cuda.Buffer = try .init(.f32, &.{ 1, hc.streams });
        defer post1.deinit();
        var comb1: cuda.Buffer = try .init(.f32, &.{ 1, hc.streams * hc.streams });
        defer comb1.deinit();
        for (0..rows) |r| {
            const x = case.x_dev.span(r, 1);
            try hc.mix(x, case.site.dots, part1.view);
            try hc.pre(x, part1.view, case.site, out1.view, post1.view, comb1.view, Case.eps, Case.hc_eps, Case.iters);
            const one = try fetch(u16, gpa, out1);
            defer gpa.free(one);
            try testing.expectEqualSlices(u16, got[r * dims ..][0..dims], one);
            const one_post = try fetch(u32, gpa, post1);
            defer gpa.free(one_post);
            try testing.expectEqualSlices(u32, @ptrCast(got_post[r * 4 ..][0..4]), one_post);
            const one_comb = try fetch(u32, gpa, comb1);
            defer gpa.free(one_comb);
            try testing.expectEqualSlices(u32, @ptrCast(got_comb[r * 16 ..][0..16]), one_comb);
        }
    }
}

/// A read-out's every result as bits: part, out, post, comb.
const ReadOut = struct {
    part: []u32,
    out: []u16,
    post: []u32,
    comb: []u32,

    fn of(gpa: std.mem.Allocator, case: *const Case, x: cuda.View, rows: usize) !ReadOut {
        var part: cuda.Buffer = try .init(.f32, &hc.partShape(rows));
        defer part.deinit();
        var out: cuda.Buffer = try .init(.bf16, &.{ rows, case.dims });
        defer out.deinit();
        var post: cuda.Buffer = try .init(.f32, &.{ rows, hc.streams });
        defer post.deinit();
        var comb: cuda.Buffer = try .init(.f32, &.{ rows, hc.streams * hc.streams });
        defer comb.deinit();
        try hc.mix(x, case.site.dots, part.view);
        try hc.pre(x, part.view, case.site, out.view, post.view, comb.view, Case.eps, Case.hc_eps, Case.iters);
        var self: ReadOut = undefined;
        self.part = try fetch(u32, gpa, part);
        errdefer gpa.free(self.part);
        self.out = try fetch(u16, gpa, out);
        errdefer gpa.free(self.out);
        self.post = try fetch(u32, gpa, post);
        errdefer gpa.free(self.post);
        self.comb = try fetch(u32, gpa, comb);
        return self;
    }

    fn deinit(self: ReadOut, gpa: std.mem.Allocator) void {
        gpa.free(self.part);
        gpa.free(self.out);
        gpa.free(self.post);
        gpa.free(self.comb);
    }
};

test "the read-out's other forms (ZSPARK_GLM_HC_LANES, ZSPARK_GLM_HC_OCTS) are the base's bits, a row alone too" {
    const gpa = testing.allocator;
    defer {
        hc.lanes = false;
        hc.octs = false;
        hc.setForms();
    }
    // 75 rows: the window's form, groups of eight and a last of three (spec 107 WS-03); 19 rows too
    for ([_]usize{ 512, 2048, 4096, 4096, 4096, 4096 }, [_]usize{ 5, 3, 1, 8, 19, 75 }, 0..) |dims, rows, seed| {
        var case: Case = try .init(gpa, 300 + seed, rows, dims);
        defer case.deinit(gpa);
        hc.lanes = false;
        hc.octs = false;
        hc.setForms();
        const base: ReadOut = try .of(gpa, &case, case.x_dev.view, rows);
        defer base.deinit(gpa);
        hc.lanes = true;
        hc.octs = true;
        hc.setForms();
        const forms: ReadOut = try .of(gpa, &case, case.x_dev.view, rows);
        defer forms.deinit(gpa);
        // a part's entries past the squares' sum are no kernel's
        for (0..rows * host.parts) |b| try testing.expectEqualSlices(u32, base.part[b * host.entry ..][0 .. host.mix + 1], forms.part[b * host.entry ..][0 .. host.mix + 1]);
        try testing.expectEqualSlices(u16, base.out, forms.out);
        try testing.expectEqualSlices(u32, base.post, forms.post);
        try testing.expectEqualSlices(u32, base.comb, forms.comb);
        // rows off 16 bytes take the base mix, not a misread of the wide loads: the same bits (spec 106 WS-05)
        var off: cuda.Buffer = try .init(.bf16, &.{case.x.len + 8});
        defer off.deinit();
        try off.uploadAt(2, std.mem.sliceAsBytes(case.x));
        const shifted: ReadOut = try .of(gpa, &case, .init(@ptrCast(@as([*]u8, @ptrCast(off.view.ptr.?)) + 2), .bf16, &.{ rows, hc.streams * dims }), rows);
        defer shifted.deinit(gpa);
        for (0..rows * host.parts) |b| try testing.expectEqualSlices(u32, base.part[b * host.entry ..][0 .. host.mix + 1], shifted.part[b * host.entry ..][0 .. host.mix + 1]);
        try testing.expectEqualSlices(u16, base.out, shifted.out);
        const entry = host.parts * host.entry;
        for (0..rows) |r| {
            const one: ReadOut = try .of(gpa, &case, case.x_dev.span(r, 1), 1);
            defer one.deinit(gpa);
            for (0..host.parts) |b| try testing.expectEqualSlices(u32, forms.part[r * entry + b * host.entry ..][0 .. host.mix + 1], one.part[b * host.entry ..][0 .. host.mix + 1]);
            try testing.expectEqualSlices(u16, forms.out[r * dims ..][0..dims], one.out);
            try testing.expectEqualSlices(u32, forms.post[r * 4 ..][0..4], one.post);
            try testing.expectEqualSlices(u32, forms.comb[r * 16 ..][0..16], one.comb);
        }
    }
}

test "with no dots the gates are a half exactly, and the collapsed row and its norm are the host's bits" {
    // exp stands between the host and the device wherever a dot is not zero; here sigmoid(0) and a softmax of
    // equals are exact on both, so the collapse's rounding and the norm's are held bit for bit
    const gpa = testing.allocator;
    const rows = 4;
    const dims = 4096;
    const wide = hc.streams * dims;
    var case: Case = try .init(gpa, 31, rows, dims);
    defer case.deinit(gpa);
    @memset(case.dots, 0);
    @memset(case.base, 0);
    // the last row's collapse is one stream's values times the gate, a row whose norm says how its sum was folded
    const scratch = try gpa.alloc(f32, dims);
    defer gpa.free(scratch);
    @memset(case.x[(rows - 1) * wide ..][0..wide], 0);
    try tellingRow(case.x[(rows - 1) * wide ..][0..dims], @as(f32, 0.5) + Case.hc_eps, scratch);
    try case.x_dev.upload(std.mem.sliceAsBytes(case.x));
    try case.held[0].zero();
    try case.held[1].zero();
    var part: cuda.Buffer = try .init(.f32, &hc.partShape(rows));
    defer part.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
    defer out.deinit();
    var post: cuda.Buffer = try .init(.f32, &.{ rows, hc.streams });
    defer post.deinit();
    var comb: cuda.Buffer = try .init(.f32, &.{ rows, hc.streams * hc.streams });
    defer comb.deinit();
    try hc.mix(case.x_dev.view, case.site.dots, part.view);
    try hc.pre(case.x_dev.view, part.view, case.site, out.view, post.view, comb.view, Case.eps, Case.hc_eps, Case.iters);
    const got = try fetch(u16, gpa, out);
    defer gpa.free(got);
    const got_post = try fetch(u32, gpa, post);
    defer gpa.free(got_post);
    const got_comb = try fetch(u32, gpa, comb);
    defer gpa.free(got_comb);
    const want = try gpa.alloc(u16, dims);
    defer gpa.free(want);
    var want_part: [host.parts * host.entry]f32 = undefined;
    var want_post: [4]f32 = undefined;
    var want_comb: [16]f32 = undefined;
    for (0..rows) |r| {
        host.hcMix(case.x[r * wide ..][0..wide], case.dots, &want_part);
        host.hcPre(case.x[r * wide ..][0..wide], &want_part, case.hostSite(), want, &want_post, &want_comb, scratch);
        try testing.expectEqual(@as(f32, 1), want_post[0]);
        try testing.expectEqualSlices(u32, @ptrCast(&want_post), got_post[r * 4 ..][0..4]);
        try testing.expectEqualSlices(u32, @ptrCast(&want_comb), got_comb[r * 16 ..][0..16]);
        try testing.expectEqualSlices(u16, want, got[r * dims ..][0..dims]);
    }
}

test "the write-back, the plain residual and the streams' mean are the host's bits for two ranks, a row alone too" {
    const gpa = testing.allocator;
    const rows = 5;
    const dims = 512;
    const wide = hc.streams * dims;
    const world = 2;
    const cap = rows + 2; // a rank's rows lie `cap` apart, more than the window
    var case: Case = try .init(gpa, 11, rows, dims);
    defer case.deinit(gpa);
    var prng: std.Random.DefaultPrng = .init(12);
    const random = prng.random();
    const gathered = try gpa.alloc(f32, world * cap * dims);
    defer gpa.free(gathered);
    // partials whose sum's rounding to bf16 the second rank decides
    shared.randomF32(random, gathered, 0.7);
    const post = try gpa.alloc(f32, rows * 4);
    defer gpa.free(post);
    for (post) |*p| p.* = random.float(f32) * 2;
    const comb = try gpa.alloc(f32, rows * 16);
    defer gpa.free(comb);
    for (comb) |*c| c.* = random.float(f32) * 0.5;

    var g_dev = try device(f32, .f32, &.{ world, cap, dims }, gathered);
    defer g_dev.deinit();
    var post_dev = try device(f32, .f32, &.{ rows, 4 }, post);
    defer post_dev.deinit();
    var comb_dev = try device(f32, .f32, &.{ rows, 16 }, comb);
    defer comb_dev.deinit();

    // streams far enough apart that fp32 loses a one: the mean's bits then say in which order they were added
    const far = [4]f32{ 3.0e7, 1, 1, -3.0e7 };
    for (0..8) |d| for (far, 0..) |v, s| {
        case.x[s * dims + d] = numeric.bf16FromF32(if (d % 2 == 0) v else far[(s + d / 2) % 4]);
    };
    try case.x_dev.upload(std.mem.sliceAsBytes(case.x));

    // the mean of the streams before the write
    var mean: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
    defer mean.deinit();
    try hc.streamMean(case.x_dev.view, mean.view);
    const got_mean = try fetch(u16, gpa, mean);
    defer gpa.free(got_mean);
    const want_mean = try gpa.alloc(u16, dims);
    defer gpa.free(want_mean);
    var mean1: cuda.Buffer = try .init(.bf16, &.{ 1, dims });
    defer mean1.deinit();
    for (0..rows) |r| {
        host.streamMean(case.x[r * wide ..][0..wide], want_mean);
        try testing.expectEqualSlices(u16, want_mean, got_mean[r * dims ..][0..dims]);
        try hc.streamMean(case.x_dev.span(r, 1), mean1.view);
        const one = try fetch(u16, gpa, mean1);
        defer gpa.free(one);
        try testing.expectEqualSlices(u16, want_mean, one);
    }

    // into the middle third of wider rows, as a tap among the drafter's: the same bits, the columns beside untouched
    const beside = try gpa.alloc(u16, rows * 3 * dims);
    defer gpa.free(beside);
    @memset(beside, 0xabcd);
    var wider = try device(u16, .bf16, &.{ rows, 3 * dims }, beside);
    defer wider.deinit();
    const base: [*]u8 = @ptrCast(wider.view.ptr.?);
    var middle: cuda.View = .init(@ptrCast(base + dims * 2), .bf16, &.{ rows, dims });
    middle.row_stride = 3 * dims;
    try hc.streamMean(case.x_dev.view, middle);
    const got_wider = try fetch(u16, gpa, wider);
    defer gpa.free(got_wider);
    for (0..rows) |r| {
        const row = got_wider[r * 3 * dims ..][0 .. 3 * dims];
        try testing.expectEqualSlices(u16, got_mean[r * dims ..][0..dims], row[dims .. 2 * dims]);
        for (row[0..dims]) |bits| try testing.expectEqual(@as(u16, 0xabcd), bits);
        for (row[2 * dims ..]) |bits| try testing.expectEqual(@as(u16, 0xabcd), bits);
    }

    // the plain residual on the mean's rows
    var plain = try device(u16, .bf16, &.{ rows, dims }, got_mean);
    defer plain.deinit();
    try hc.residualAdd(plain.view, g_dev.view);
    const got_plain = try fetch(u16, gpa, plain);
    defer gpa.free(got_plain);
    for (0..rows) |r| {
        @memcpy(want_mean, got_mean[r * dims ..][0..dims]);
        host.residualAdd(want_mean, &.{ gathered[r * dims ..][0..dims], gathered[(cap + r) * dims ..][0..dims] });
        try testing.expectEqualSlices(u16, want_mean, got_plain[r * dims ..][0..dims]);
    }

    // the branch alone, as the drafter takes it
    var alone: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
    defer alone.deinit();
    try hc.branch(g_dev.view, alone.view);
    const got_alone = try fetch(u16, gpa, alone);
    defer gpa.free(got_alone);
    for (0..rows) |r| {
        host.branchRow(&.{ gathered[r * dims ..][0..dims], gathered[(cap + r) * dims ..][0..dims] }, want_mean);
        try testing.expectEqualSlices(u16, want_mean, got_alone[r * dims ..][0..dims]);
    }

    // the write-back, in place
    var written = try device(u16, .bf16, &.{ rows, wide }, case.x);
    defer written.deinit();
    try hc.post(written.view, g_dev.view, post_dev.view, comb_dev.view);
    const got = try fetch(u16, gpa, written);
    defer gpa.free(got);
    const want = try gpa.alloc(u16, wide);
    defer gpa.free(want);
    var changed: usize = 0;
    for (0..rows) |r| {
        @memcpy(want, case.x[r * wide ..][0..wide]);
        host.hcPost(want, &.{ gathered[r * dims ..][0..dims], gathered[(cap + r) * dims ..][0..dims] }, post[r * 4 ..][0..4], comb[r * 16 ..][0..16]);
        try testing.expectEqualSlices(u16, want, got[r * wide ..][0..wide]);
        for (want, case.x[r * wide ..][0..wide]) |after, before| changed += @intFromBool(after != before);
    }
    try testing.expect(changed * 2 > rows * wide);

    // a row alone: its ranks' partials one row apart
    var g1: cuda.Buffer = try .init(.f32, &.{ world, 1, dims });
    defer g1.deinit();
    var x1: cuda.Buffer = try .init(.bf16, &.{ 1, wide });
    defer x1.deinit();
    for ([_]usize{ 0, 3, rows - 1 }) |r| {
        try g1.uploadAt(0, std.mem.sliceAsBytes(gathered[r * dims ..][0..dims]));
        try g1.uploadAt(dims * @sizeOf(f32), std.mem.sliceAsBytes(gathered[(cap + r) * dims ..][0..dims]));
        try x1.upload(std.mem.sliceAsBytes(case.x[r * wide ..][0..wide]));
        try hc.post(x1.view, g1.view, post_dev.span(r, 1), comb_dev.span(r, 1));
        const one = try fetch(u16, gpa, x1);
        defer gpa.free(one);
        try testing.expectEqualSlices(u16, got[r * wide ..][0..wide], one);
    }

    // bf16 partials (spec 116 WS-04): the host's sum of the values widened, and fp32 partials rounded in place
    // (`roundBf16`, a round of both forms) give post the same bits
    const halves = try gpa.alloc(u16, gathered.len);
    defer gpa.free(halves);
    const widened = try gpa.alloc(f32, gathered.len);
    defer gpa.free(widened);
    for (gathered, halves, widened) |value, *half, *wide_value| {
        half.* = numeric.bf16FromF32(value);
        wide_value.* = numeric.f32FromBf16(half.*);
    }
    var g16 = try device(u16, .bf16, &.{ world, cap, dims }, halves);
    defer g16.deinit();
    var rounded = try device(f32, .f32, &.{ world * cap, dims }, gathered);
    defer rounded.deinit();
    try hc.roundBf16(rounded.view);
    const got_rounded = try fetch(u32, gpa, rounded);
    defer gpa.free(got_rounded);
    try testing.expectEqualSlices(u32, @ptrCast(widened), got_rounded);
    for ([_]cuda.View{ g16.view, .init(rounded.view.ptr.?, .f32, &.{ world, cap, dims }) }) |g| {
        var narrow = try device(u16, .bf16, &.{ rows, wide }, case.x);
        defer narrow.deinit();
        try hc.post(narrow.view, g, post_dev.view, comb_dev.view);
        const got_narrow = try fetch(u16, gpa, narrow);
        defer gpa.free(got_narrow);
        for (0..rows) |r| {
            @memcpy(want, case.x[r * wide ..][0..wide]);
            host.hcPost(want, &.{ widened[r * dims ..][0..dims], widened[(cap + r) * dims ..][0..dims] }, post[r * 4 ..][0..4], comb[r * 16 ..][0..16]);
            try testing.expectEqualSlices(u16, want, got_narrow[r * wide ..][0..wide]);
        }
    }
}

test "RMSNorm is the host's bits at every width, on the first columns of wider rows, a row alone too" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(21);
    const random = prng.random();
    const rows = 4;
    // a power of two, GLM's ranks (2048 and 512 of a row of 2560), a width the fold pads, the widest
    for ([_]usize{ 4096, 2048, 512, 96, 1 }) |dims| {
        const stride = dims + 37;
        const x = try gpa.alloc(u16, rows * stride);
        defer gpa.free(x);
        shared.randomBf16(random, x, 3.0);
        const scratch = try gpa.alloc(f32, try std.math.ceilPowerOfTwo(usize, dims));
        defer gpa.free(scratch);
        // a row whose output says how its squares were folded, where the width has room for one
        if (dims >= 512) try tellingRow(x[0..dims], 1, scratch);
        const w = try gpa.alloc(u16, dims);
        defer gpa.free(w);
        shared.randomBf16(random, w, 1.0);
        var x_dev = try device(u16, .bf16, &.{ rows, stride }, x);
        defer x_dev.deinit();
        var w_dev = try device(u16, .bf16, &.{dims}, w);
        defer w_dev.deinit();
        var out: cuda.Buffer = try .init(.bf16, &.{ rows, dims });
        defer out.deinit();
        var view = x_dev.view;
        view.size[1] = @intCast(dims);
        view.row_stride = @intCast(stride);
        try hc.rmsnorm(view, w_dev.view, out.view, 1e-5);
        const got = try fetch(u16, gpa, out);
        defer gpa.free(got);
        const want = try gpa.alloc(u16, dims);
        defer gpa.free(want);
        var out1: cuda.Buffer = try .init(.bf16, &.{ 1, dims });
        defer out1.deinit();
        for (0..rows) |r| {
            host.rmsnorm(x[r * stride ..][0..dims], .{ .bf16 = w }, want, 1e-5, scratch);
            try testing.expectEqualSlices(u16, want, got[r * dims ..][0..dims]);
            var alone = x_dev.span(r, 1);
            alone.size[1] = @intCast(dims);
            alone.row_stride = @intCast(stride);
            try hc.rmsnorm(alone, w_dev.view, out1.view, 1e-5);
            const one = try fetch(u16, gpa, out1);
            defer gpa.free(one);
            try testing.expectEqualSlices(u16, want, one);
        }
    }
}

test "embed copies the token's row into the four streams" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(1);
    const vocab = 9;
    const dims = 512;
    const table = try gpa.alloc(u16, vocab * dims);
    defer gpa.free(table);
    shared.randomBf16(prng.random(), table, 1.0);
    const ids = [_]i32{ 7, 0, 8 };
    var table_dev = try device(u16, .bf16, &.{ vocab, dims }, table);
    defer table_dev.deinit();
    var ids_dev = try device(i32, .i32, &.{ids.len}, &ids);
    defer ids_dev.deinit();
    var out: cuda.Buffer = try .init(.bf16, &.{ ids.len, hc.streams * dims });
    defer out.deinit();
    try hc.embed(ids_dev.view, table_dev.view, out.view);
    const got = try fetch(u16, gpa, out);
    defer gpa.free(got);
    for (ids, 0..) |id, r| for (0..hc.streams) |s| {
        try testing.expectEqualSlices(u16, table[@as(usize, @intCast(id)) * dims ..][0..dims], got[(r * hc.streams + s) * dims ..][0..dims]);
    };
}

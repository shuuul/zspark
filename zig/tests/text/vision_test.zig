//! `vision.zig`'s tests, against TensorFold 0.6.5's processor: `zig/tests/fixtures/vision/cases.json`
//! (`tools/reference/vision_export.py`), `processor_cases.json` (`vision_cases.py`) and, where
//! `ZSPARK_REFERENCE_VISION` names it, their tensors; the whole prompt where `ZSPARK_FLASH_NEXT_FP4_LOCAL` names the
//! checkpoint too. Every test's name says "image", for `tools/check.sh text image`.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Value = std.json.Value;
const testing = std.testing;
const vision = @import("../../src/text/vision.zig");
const frontend_file = @import("../../src/text/frontend.zig");
const safetensors = @import("../../src/weights/safetensors.zig");

const Settings = vision.Settings;

/// Flash Next's settings, as its config.json and processor_config.json give them.
const flash: Settings = .{ .patch = 16, .temporal = 2, .merge = 2, .image_token = 248056, .video_token = 248057, .vision_start = 248053, .vision_end = 248054 };

fn fixture(arena: Allocator, comptime name: []const u8) !Value {
    return std.json.parseFromSliceLeaky(Value, arena, @embedFile("fixtures/vision/" ++ name), .{});
}

fn int(v: Value) u32 {
    return @intCast(v.integer);
}

fn field(o: Value, name: []const u8) Value {
    return o.object.get(name).?;
}

fn budgetOf(kwargs: Value) vision.Budget {
    var b: vision.Budget = .{};
    if (kwargs.object.get("max_visual_tokens")) |v| b.max_visual_tokens = int(v);
    if (kwargs.object.get("max_prompt_tokens")) |v| b.max_prompt_tokens = int(v);
    return b;
}

fn sha256Hex(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

fn openEnv(name: []const u8) !Io.Dir {
    const path = testing.environ.getPosix(name) orelse return error.SkipZigTest;
    return Io.Dir.cwd().openDir(testing.io, path, .{});
}

test "image grids and aspect refusals equal the processor's at the probes' sizes and budgets" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const doc = try fixture(arena_state.allocator(), "processor_cases.json");
    var problem: vision.Problem = .{};
    var checked: usize = 0;
    for (field(doc, "probes").array.items) |probe| {
        const sizes = field(probe, "sizes").array.items;
        const detail = std.meta.stringToEnum(vision.Detail, field(probe, "detail").string).?;
        const budget = budgetOf(field(probe, "kwargs"));
        if (probe.object.get("grids")) |grids| {
            for (sizes, grids.array.items) |size, grid| {
                const b = vision.bounds(flash, sizes.len, detail, budget);
                const got = try vision.smartResize(int(size.array.items[1]), int(size.array.items[0]), 32, b[0], b[1], &problem);
                try testing.expectEqual([3]u32{ 1, int(grid.array.items[1]), int(grid.array.items[2]) }, [3]u32{ 1, got.h / 16, got.w / 16 });
                checked += 1;
            }
        } else {
            const want = field(probe, "error").string;
            if (!std.mem.startsWith(u8, want, "absolute aspect ratio")) continue;
            const size = sizes[0].array.items;
            try testing.expectError(error.BadRequest, vision.smartResize(int(size[1]), int(size[0]), 32, 1024, 4096 * 1024, &problem));
            try testing.expectEqualStrings(want, problem.message());
            checked += 1;
        }
    }
    for (field(doc, "random").array.items) |r| {
        const b = vision.bounds(flash, 1, .auto, .{ .max_visual_tokens = int(field(r, "max_visual_tokens")) });
        const got = try vision.smartResize(int(field(r, "height")), int(field(r, "width")), 32, b[0], b[1], &problem);
        const grid = field(r, "grid").array.items;
        try testing.expectEqual([2]u32{ int(grid[1]), int(grid[2]) }, [2]u32{ got.h / 16, got.w / 16 });
        checked += 1;
    }
    try testing.expectEqual(@as(usize, 30), checked);
}

test "image normalization is torchvision's fused one, and Python's round and float repr" {
    const lut = vision.table(flash);
    for (0..3) |c| for (0..256) |i| {
        try testing.expectEqual((@as(f32, @floatFromInt(i)) - 127.5) / 127.5, lut[c][i]);
    };
    try testing.expectEqual(@as(f64, 2), vision.roundEven(2.5));
    try testing.expectEqual(@as(f64, 4), vision.roundEven(3.5));
    try testing.expectEqual(@as(f64, 3), vision.roundEven(2.75));
    var buffer: [32]u8 = undefined;
    try testing.expectEqualStrings("201.0", vision.pyFloat(&buffer, 201));
    try testing.expectEqualStrings("200.5", vision.pyFloat(&buffer, 200.5));
    try testing.expectEqualStrings("333.3333333333333", vision.pyFloat(&buffer, 1000.0 / 3.0));
}

test "image settings come from the config and the processor files, and a disagreeing one is refused" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: vision.Problem = .{};
    const config =
        \\{"model_type":"qwen4_exp","image_token_id":248056,"video_token_id":248057,"vision_start_token_id":248053,
        \\ "vision_end_token_id":248054,"vision_config":{"patch_size":16,"temporal_patch_size":2,"spatial_merge_size":2,"in_channels":3}}
    ;
    const processor =
        \\{"image_processor":{"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"rescale_factor":0.00392156862745098,
        \\ "do_rescale":true,"do_normalize":true,"patch_size":16,"merge_size":2,"size":{"shortest_edge":65536,"longest_edge":16777216}}}
    ;
    const s = (try Settings.parse(arena, config, null, processor, &problem)).?;
    try testing.expectEqual(flash.patch, s.patch);
    try testing.expectEqual(flash.video_token, s.video_token);
    try testing.expectEqual(flash.mean, s.mean);
    try testing.expectEqual(vision.table(flash), vision.table(s));
    try testing.expectEqual(@as(?Settings, null), try Settings.parse(arena, "{\"model_type\":\"qwen3_5_text\"}", null, null, &problem));
    try testing.expectError(error.InvalidSettings, Settings.parse(arena, config, "{\"merge_size\":4}", null, &problem));
    try testing.expectEqualStrings("Image processor merge_size disagrees with the vision tower", problem.message());
    try testing.expectError(error.InvalidSettings, Settings.parse(arena, config, null, "{\"image_processor\":{\"do_normalize\":false}}", &problem));
}

const marks: vision.Ids = .{ .start = 50, .image = 51, .end = 52, .video = 53 };

test "image positions: text counts up, an image sits at one position with its grid, text resumes past its larger side" {
    const gpa = testing.allocator;
    var problem: vision.Problem = .{};
    // as TensorFold 1.0.12's `rope.zig` tests: one 4 x 6 image (2 x 3 merged)
    const prompt = [_]u32{ 7, 50, 51, 51, 51, 51, 51, 51, 52, 9 };
    var l = try vision.Layout.init(gpa, &prompt, &.{.{ 1, 4, 6 }}, marks, 2, &problem);
    defer l.deinit(gpa);
    try testing.expectEqual(@as(i32, -3), l.shift);
    var axes: [3][10]i32 = undefined;
    l.fill(.{ &axes[0], &axes[1], &axes[2] });
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6 }, &axes[0]);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 3, 3, 3, 5, 6 }, &axes[1]);
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 2, 3, 4, 5, 6 }, &axes[2]);
    try testing.expectEqual([3]i32{ 97, 97, 97 }, l.position(100)); // a decode row
    try testing.expect(l.holds(0, 3) and l.holds(7, 1) and !l.holds(8, 2) and !l.holds(0, 2));

    // a tall image (3 x 1 merged) then a square one (2 x 2)
    const two = [_]u32{ 50, 51, 51, 51, 52, 8, 50, 51, 51, 51, 51, 52, 9 };
    var m = try vision.Layout.init(gpa, &two, &.{ .{ 1, 6, 2 }, .{ 1, 4, 4 } }, marks, 2, &problem);
    defer m.deinit(gpa);
    try testing.expectEqual([3]i32{ 1, 3, 1 }, m.position(3));
    try testing.expectEqual([3]i32{ 4, 4, 4 }, m.position(4));
    try testing.expectEqual([3]i32{ 7, 8, 7 }, m.position(9));
    try testing.expectEqual([3]i32{ 9, 9, 9 }, m.position(11));
    try testing.expectEqual(@as(i32, -2), m.shift);
    const none: vision.Layout = .{};
    try testing.expectEqual([3]i32{ 41, 41, 41 }, none.position(41));
}

test "image placeholders that do not match their grids are refused with TensorFold's words" {
    const gpa = testing.allocator;
    var problem: vision.Problem = .{};
    const Case = struct { prompt: []const u32, grids: []const [3]u32, want: []const u8 };
    const placeholders = "Image placeholders must match the processed image grid exactly";
    const cases = [_]Case{
        .{ .prompt = &.{ 49, 51, 51, 52 }, .grids = &.{.{ 1, 2, 4 }}, .want = placeholders },
        .{ .prompt = &.{ 50, 51, 51, 54 }, .grids = &.{.{ 1, 2, 4 }}, .want = placeholders },
        .{ .prompt = &.{ 50, 51, 7, 52 }, .grids = &.{.{ 1, 2, 4 }}, .want = placeholders },
        .{ .prompt = &.{ 50, 51, 51 }, .grids = &.{.{ 1, 2, 4 }}, .want = placeholders },
        .{ .prompt = &.{ 50, 51, 51, 51, 52 }, .grids = &.{.{ 1, 2, 4 }}, .want = placeholders },
        .{ .prompt = &.{ 51, 51, 52 }, .grids = &.{.{ 1, 2, 4 }}, .want = placeholders },
        .{ .prompt = &.{ 50, 51, 51, 52, 50, 51, 52 }, .grids = &.{.{ 1, 2, 4 }}, .want = "The prompt contains image tokens without a corresponding image" },
        .{ .prompt = &.{ 50, 51, 51, 52 }, .grids = &.{ .{ 1, 2, 4 }, .{ 1, 2, 2 } }, .want = "Image grid has no matching image tokens in the prompt" },
        .{ .prompt = &.{ 50, 51, 51, 52 }, .grids = &.{.{ 2, 2, 4 }}, .want = "An image grid must contain one frame and merge-aligned positive dimensions" },
        .{ .prompt = &.{ 50, 51, 51, 52 }, .grids = &.{.{ 1, 3, 4 }}, .want = "An image grid must contain one frame and merge-aligned positive dimensions" },
        .{ .prompt = &.{ 50, 51, 51, 52, 53 }, .grids = &.{.{ 1, 2, 4 }}, .want = "Video inputs are not supported by the Qwen image frontend" },
    };
    for (cases) |k| {
        try testing.expectError(error.BadRequest, vision.Layout.init(gpa, k.prompt, k.grids, marks, 2, &problem));
        try testing.expectEqualStrings(k.want, problem.message());
    }
}

test "image resize, patches and content_hash equal the processor's on random images and the fixtures' pictures" {
    // Runs with ZSPARK_REFERENCE_VISION set; skipped without.
    var dir = try openEnv("ZSPARK_REFERENCE_VISION");
    defer dir.close(testing.io);
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: vision.Problem = .{};
    const lut = vision.table(flash);

    var more = try safetensors.File.open(gpa, testing.io, dir, "processor_more.safetensors");
    defer more.close();
    const doc = try fixture(arena, "processor_cases.json");
    var differ: usize = 0;
    var worst: u8 = 0;
    for (field(doc, "random").array.items) |r| {
        const name = field(r, "name").string;
        const rgb = more.get(try std.fmt.allocPrint(arena, "{s}.rgb", .{name})).?;
        const want = more.get(try std.fmt.allocPrint(arena, "{s}.resized", .{name})).?;
        const img: vision.Image = .{ .width = int(field(r, "width")), .height = int(field(r, "height")), .rgb = rgb.data };
        try testing.expectEqual(@as(u64, img.height), rgb.shape[0]);
        try testing.expectEqualStrings(field(r, "content_hash").string, &img.contentHash());
        const size, const got = try vision.resized(gpa, flash, img, 1, .{ .max_visual_tokens = int(field(r, "max_visual_tokens")) }, &problem);
        defer gpa.free(got);
        try testing.expectEqual([2]u64{ size.h, size.w }, [2]u64{ want.shape[0], want.shape[1] });
        for (got, want.data) |a, b| if (a != b) {
            differ += 1;
            worst = @max(worst, @max(a, b) - @min(a, b));
        };
        const pv = try gpa.alloc(f32, @as(usize, size.h / 16) * (size.w / 16) * flash.patchWidth());
        defer gpa.free(pv);
        vision.patchify(flash, &lut, got, size, pv);
        try testing.expectEqualStrings(field(r, "pixel_values_sha256").string, &sha256Hex(std.mem.sliceAsBytes(pv)));
    }
    // torchvision's bytes exactly: none differs (`worst` says by how much one would)
    try testing.expectEqual([2]usize{ 0, 0 }, [2]usize{ differ, worst });

    var pictures = try safetensors.File.open(gpa, testing.io, dir, "images.safetensors");
    defer pictures.close();
    const cases = try fixture(arena, "cases.json");
    var it = field(cases, "images").object.iterator();
    var hashed: usize = 0;
    while (it.next()) |e| {
        const info = e.value_ptr.*;
        const rgb = pictures.get(field(info, "rgb_name").string).?;
        const img: vision.Image = .of(.{ .width = int(field(info, "width")), .height = int(field(info, "height")), .pixels = @constCast(rgb.data) }, .auto);
        try testing.expectEqualStrings(field(info, "rgb_sha256").string, &sha256Hex(rgb.data));
        const want = field(info, "content_hash");
        if (want == .null) continue; // the GIF: TensorFold takes none
        try testing.expectEqualStrings(want.string, &img.contentHash());
        hashed += 1;
    }
    try testing.expectEqual(@as(usize, 5), hashed);
}

/// cases.json's case as a request's messages: one user turn, its images, then its question.
fn caseMessages(arena: Allocator, case: Value) !Value {
    var parts: std.json.Array = .init(arena);
    for (field(case, "images").array.items) |name| {
        var url: std.json.ObjectMap = .empty;
        try url.put(arena, "url", .{ .string = try std.fmt.allocPrint(arena, "file://{s}.png", .{name.string}) });
        var part: std.json.ObjectMap = .empty;
        try part.put(arena, "type", .{ .string = "image_url" });
        try part.put(arena, "image_url", .{ .object = url });
        try parts.append(.{ .object = part });
    }
    var text: std.json.ObjectMap = .empty;
    try text.put(arena, "type", .{ .string = "text" });
    try text.put(arena, "text", field(case, "question"));
    try parts.append(.{ .object = text });
    var message: std.json.ObjectMap = .empty;
    try message.put(arena, "role", .{ .string = "user" });
    try message.put(arena, "content", .{ .array = parts });
    var list: std.json.Array = .init(arena);
    try list.append(.{ .object = message });
    return .{ .array = list };
}

test "image prompts of the reference cases: render, ids, grids, spans, positions, delta and pixel_values equal TensorFold's" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL and ZSPARK_REFERENCE_VISION set; skipped without either.
    var reference = try openEnv("ZSPARK_REFERENCE_VISION");
    defer reference.close(testing.io);
    var checkpoint = try openEnv("ZSPARK_FLASH_NEXT_FP4_LOCAL");
    defer checkpoint.close(testing.io);
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: frontend_file.Diagnostic = .{};
    var frontend: frontend_file.Frontend = try .load(gpa, testing.io, checkpoint, null, &diagnostic);
    defer frontend.deinit(gpa);
    var problem: vision.Problem = .{};
    const s = (try Settings.load(arena, testing.io, checkpoint, &problem)).?;
    try testing.expectEqual(vision.table(flash), vision.table(s));
    try testing.expectEqualStrings("<|image_pad|>", try s.imageText(&frontend.tokenizer, &problem));

    var pictures = try safetensors.File.open(gpa, testing.io, reference, "images.safetensors");
    defer pictures.close();
    const doc = try fixture(arena, "cases.json");
    var it = field(doc, "cases").object.iterator();
    while (it.next()) |e| {
        const case = e.value_ptr.*;
        const rendered = try frontend.render(arena, .{ .messages = try caseMessages(arena, case), .thinking = false, .allow_images = true }, &diagnostic);
        try testing.expectEqualStrings(field(case, "rendered").string, rendered);

        var images: std.ArrayList(vision.Image) = .empty;
        for (field(case, "images").array.items) |name| {
            const t = pictures.get(try std.fmt.allocPrint(arena, "{s}.rgb", .{name.string})).?;
            try images.append(arena, .{ .width = @intCast(t.shape[1]), .height = @intCast(t.shape[0]), .rgb = t.data });
        }
        var p = try vision.prepare(gpa, s, &frontend.tokenizer, rendered, images.items, .{}, &problem);
        defer p.deinit(gpa);

        const ids = field(case, "ids").array.items;
        try testing.expectEqual(ids.len, p.ids.len);
        for (ids, p.ids) |want, got| try testing.expectEqual(int(want), got);
        const grids = field(case, "image_grid_thw").array.items;
        try testing.expectEqual(grids.len, p.grids.len);
        for (grids, p.grids) |want, got| for (want.array.items, got) |w, g| try testing.expectEqual(int(w), g);
        const spans = field(case, "image_spans").array.items;
        for (spans, p.layout.spans) |want, got| {
            try testing.expectEqual([2]u32{ int(want.array.items[0]), int(want.array.items[1]) }, [2]u32{ got.at, got.end() });
        }
        try testing.expectEqual(@as(i32, @intCast(field(case, "rope_delta").integer)), p.delta());
        try testing.expectEqual(@as(usize, int(field(case, "visual_tokens"))), p.visualTokens());
        const axes = field(case, "position_ids").array.items;
        for (axes, 0..) |axis, a| for (axis.array.items, 0..) |want, row| {
            try testing.expectEqual(@as(i32, @intCast(want.integer)), p.layout.position(@intCast(row))[a]);
        };
        const hashes = field(case, "image_hashes").array.items;
        for (hashes, p.hashes) |want, got| try testing.expectEqualStrings(want.string, &got);

        // the processor's own tensors: the positions as stored, and pixel_values to the bit
        var processed = try safetensors.File.open(gpa, testing.io, reference, try std.fmt.allocPrint(arena, "{s}.processor.safetensors", .{e.key_ptr.*}));
        defer processed.close();
        const pv = processed.get("pixel_values").?;
        try testing.expectEqual(@as(u64, p.pixel_values.len), pv.count());
        try testing.expectEqualSlices(u8, pv.data, std.mem.sliceAsBytes(p.pixel_values));
        try testing.expectEqualStrings(field(field(case, "pixel_values"), "sha256").string, &sha256Hex(std.mem.sliceAsBytes(p.pixel_values)));
        const stored = processed.get("position_ids").?;
        const positions = try arena.alloc(i32, 3 * p.ids.len);
        p.layout.fill(.{ positions[0..p.ids.len], positions[p.ids.len..][0..p.ids.len], positions[2 * p.ids.len ..] });
        try testing.expectEqualSlices(u8, stored.data, std.mem.sliceAsBytes(positions));
        const stored_ids = processed.get("ids").?;
        try testing.expectEqualSlices(u8, stored_ids.data, std.mem.sliceAsBytes(p.ids));
    }
}

test "image prompts at the probes' sizes and budgets: lengths, deltas and refusals equal TensorFold's" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set; skipped without.
    var checkpoint = try openEnv("ZSPARK_FLASH_NEXT_FP4_LOCAL");
    defer checkpoint.close(testing.io);
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diagnostic: frontend_file.Diagnostic = .{};
    var frontend: frontend_file.Frontend = try .load(gpa, testing.io, checkpoint, null, &diagnostic);
    defer frontend.deinit(gpa);
    var problem: vision.Problem = .{};
    const s = (try Settings.load(arena, testing.io, checkpoint, &problem)).?;

    const doc = try fixture(arena, "processor_cases.json");
    const provenance = field(doc, "provenance");
    const marker = field(provenance, "marker").string;
    const shape = field(provenance, "rendered").string;
    const hole = std.mem.find(u8, shape, "{images}").?;
    for (field(doc, "probes").array.items) |probe| {
        const sizes = field(probe, "sizes").array.items;
        var rendered: std.ArrayList(u8) = .empty;
        try rendered.appendSlice(arena, shape[0..hole]);
        var images: std.ArrayList(vision.Image) = .empty;
        const detail = std.meta.stringToEnum(vision.Detail, field(probe, "detail").string).?;
        for (sizes) |size| {
            const w = int(size.array.items[0]);
            const h = int(size.array.items[1]);
            const rgb = try arena.alloc(u8, @as(usize, w) * h * 3);
            for (rgb, 0..) |*b, i| b.* = @truncate(i *% 2654435761 >> 13);
            try images.append(arena, .{ .width = w, .height = h, .rgb = rgb, .detail = detail });
            try rendered.appendSlice(arena, marker);
        }
        try rendered.appendSlice(arena, shape[hole + "{images}".len ..]);
        const budget = budgetOf(field(probe, "kwargs"));
        if (probe.object.get("error")) |want| {
            const err = vision.prepare(gpa, s, &frontend.tokenizer, rendered.items, images.items, budget, &problem);
            try testing.expectError(if (std.mem.startsWith(u8, want.string, "This server's")) error.ContextLength else error.BadRequest, err);
            try testing.expectEqualStrings(want.string, problem.message());
            continue;
        }
        var p = try vision.prepare(gpa, s, &frontend.tokenizer, rendered.items, images.items, budget, &problem);
        defer p.deinit(gpa);
        try testing.expectEqual(@as(usize, int(field(probe, "ids"))), p.ids.len);
        try testing.expectEqual(@as(i32, @intCast(field(probe, "delta").integer)), p.delta());
        for (field(probe, "grids").array.items, p.grids) |want, got| for (want.array.items, got) |w, g| try testing.expectEqual(int(w), g);
    }
    // a marker short, or one too many
    const one = [_]vision.Image{.{ .width = 1, .height = 1, .rgb = "abc" }};
    try testing.expectError(error.BadRequest, vision.prepare(gpa, s, &frontend.tokenizer, "no marker", &one, .{}, &problem));
    try testing.expectEqualStrings("The rendered prompt must contain exactly one image marker for each image", problem.message());
    const bad = [_]vision.Image{.{ .width = 2, .height = 1, .rgb = "abc" }};
    try testing.expectError(error.BadRequest, vision.prepare(gpa, s, &frontend.tokenizer, marker, &bad, .{}, &problem));
    try testing.expectEqualStrings("image input requires RGB bytes, matching dimensions and valid detail", problem.message());
}

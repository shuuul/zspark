//! `layout.zig`'s tests: the names `unused` takes as read, and, from the checkpoint's safetensors headers alone
//! (mapped, no tensor's bytes read, no device), Unlimited-OCR-2's language model as `Layout` finds it
//! (`$ZSPARK_UNLIMITED_OCR_2`, spec 147 WS-02) with a DASH router a layer, and every tensor under
//! `model.layers.` either found or `unused`.

const std = @import("std");
const Io = std.Io;
const testing = std.testing;
const config_file = @import("../../../src/models/unlimited_ocr/config.zig");
const layout_file = @import("../../../src/models/unlimited_ocr/layout.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");

const Config = config_file.Config;
const Layout = layout_file.Layout;
const Tensor = safetensors.Tensor;

test "unlimited_ocr layout: unused names v2's indexer and CLIP's position ids alone" {
    for ([_][]const u8{
        "model.layers.0.self_attn.indexer.q_proj.weight",
        "model.layers.11.self_attn.indexer.k_norm.weight",
        "model.vision_model.embeddings.position_ids",
    }) |name| try testing.expect(layout_file.unused(name));
    for ([_][]const u8{
        "model.layers.0.self_attn.dash_router.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.x.self_attn.indexer.q_proj.weight",
        "model.layers.0.self_attn.indexer.q_proj",
        "model.vision_model.embeddings.position_embedding.weight",
    }) |name| try testing.expect(!layout_file.unused(name));
}

test "unlimited_ocr layout: each checkpoint's tensors from its headers, v2's routers, nothing unread" {
    var checked: usize = 0;
    for ([_][]const u8{"ZSPARK_UNLIMITED_OCR_2"}) |name| {
        const path = testing.environ.getPosix(name) orelse continue;
        if (path.len == 0) continue;
        try check(path);
        checked += 1;
    }
    if (checked == 0) return error.SkipZigTest;
}

fn check(path: []const u8) !void {
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var dir = try Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var diagnostic: config_file.Diagnostic = .{};
    const config = Config.read(arena.allocator(), testing.io, dir, &diagnostic) catch |err| {
        std.debug.print("{s}: {s}\n", .{ path, diagnostic.message });
        return err;
    };
    var checkpoint: safetensors.Checkpoint = try .open(gpa, testing.io, .cwd(), path);
    defer checkpoint.close();
    const layout = Layout.init(arena.allocator(), &config, &checkpoint, &diagnostic) catch |err| {
        std.debug.print("{s}: {s}\n", .{ path, diagnostic.message });
        return err;
    };

    // what the layout found, by where its bytes lie
    var found: std.AutoHashMapUnmanaged(usize, void) = .empty;
    defer found.deinit(gpa);
    for ([_]Tensor{ layout.embed, layout.norm, layout.head }) |t| try found.put(gpa, @intFromPtr(t.data.ptr), {});
    for (layout.layers) |layer| {
        for ([_]Tensor{ layer.input_norm, layer.post_norm, layer.q, layer.k, layer.v, layer.o }) |t| try found.put(gpa, @intFromPtr(t.data.ptr), {});
        switch (layer.mlp) {
            .dense => |d| for ([_]Tensor{ d.gate, d.up, d.down }) |t| try found.put(gpa, @intFromPtr(t.data.ptr), {}),
            .experts => |e| {
                try found.put(gpa, @intFromPtr(e.router.data.ptr), {});
                for (e.gate, e.up, e.down) |g, u, d| for ([_]Tensor{ g, u, d }) |t| try found.put(gpa, @intFromPtr(t.data.ptr), {});
                for ([_]Tensor{ e.shared.gate, e.shared.up, e.shared.down }) |t| try found.put(gpa, @intFromPtr(t.data.ptr), {});
            },
        }
        for ([_]Tensor{ layer.router.head_cls, layer.router.q, layer.router.k, layer.router.q_norm, layer.router.k_norm }) |t| try found.put(gpa, @intFromPtr(t.data.ptr), {});
    }

    var unread: usize = 0;
    var skipped: usize = 0;
    for (checkpoint.tensors.keys(), checkpoint.tensors.values()) |name, t| {
        if (layout_file.unused(name)) {
            skipped += 1;
            continue;
        }
        if (!std.mem.startsWith(u8, name, "model.layers.")) continue;
        if (found.contains(@intFromPtr(t.data.ptr))) continue;
        std.debug.print("{s}: {s} is neither read nor unused\n", .{ path, name });
        unread += 1;
    }
    std.debug.print("{s}: {d} tensors, {d} found, DASH, {d} unused\n", .{ path, checkpoint.tensors.count(), found.count(), skipped });
    try testing.expectEqual(@as(usize, 0), unread);
    // four indexer tensors a layer and CLIP's position ids
    try testing.expectEqual(4 * config.layers + 1, skipped);
}

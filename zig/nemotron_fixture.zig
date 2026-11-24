//! `nemotron-fixture DIR [HF_DIR]`: writes the small synthetic Nemotron-3.5-Transcribe checkpoint of
//! `zig/tests/models/nemotron/fixture.zig` as a real checkpoint directory `zspark generate DIR` opens:
//! the three config files, every tensor of `layout.kinds` at the fixture's shrunk shapes in one
//! safetensors shard, and the tokenizer files copied from the published checkpoint's public files
//! (`HF_DIR`, default `~/.cache/zspark/scratch-151/hf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`).
//! `fill` and `agreeable` seed the weights: every reply token argmaxes id 0, so `generate` with and
//! without its MTP drafter prints the same reply. No GPU.

const std = @import("std");
const Io = std.Io;
const fixture = @import("tests/models/nemotron/fixture.zig");
const safetensors = @import("src/weights/safetensors.zig");
const config_file = @import("src/models/nemotron/config.zig");

const usage = "usage: nemotron-fixture DIR [HF_DIR]";

/// The files `generate` (or a tokenizer user) expects beside the tensors.
const copied = [_][]const u8{ "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json" };

pub fn main(init: std.process.Init) !u8 {
    var buffer: [4096]u8 = undefined;
    var stdout_file: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const stdout = &stdout_file.interface;
    defer stdout.flush() catch {};
    const arena = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len < 2 or argv.len > 3) {
        try stdout.print("{s}\n", .{usage});
        return 2;
    }
    const home = init.minimal.environ.getPosix("HOME") orelse ".";
    const hf = if (argv.len == 3) argv[2] else try std.fs.path.join(arena, &.{ home, ".cache/zspark/scratch-151/hf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4" });
    var dir = try Io.Dir.cwd().createDirPathOpen(io, argv[1], .{});
    defer dir.close(io);
    var hf_dir = Io.Dir.cwd().openDir(io, hf, .{}) catch |err| {
        std.log.err("{s}: {t}", .{ hf, err });
        return 1;
    };
    defer hf_dir.close(io);

    try dir.writeFile(io, .{ .sub_path = "config.json", .data = fixture.config_text });
    try dir.createDirPath(io, "llm_backbone");
    try dir.writeFile(io, .{ .sub_path = "llm_backbone/config.json", .data = fixture.backbone_text });
    try dir.writeFile(io, .{ .sub_path = "hf_quant_config.json", .data = fixture.quant_text });
    for (copied) |name| {
        const text = hf_dir.readFileAlloc(io, name, arena, .limited(1 << 26)) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };
        try dir.writeFile(io, .{ .sub_path = name, .data = text });
    }

    // the tensors: fixture.synthetic's names, shapes and dtypes as one shard, `fill` + `agreeable`'s bytes
    var diagnostic: config_file.Diagnostic = .{};
    const c = try fixture.parse(arena, &diagnostic);
    var checkpoint = try fixture.synthetic(arena, c, 0x5151);
    fixture.fill(&checkpoint, 0xf17);
    fixture.agreeable(&checkpoint);
    try writeShard(arena, io, dir, "model-00001-of-00001.safetensors", &checkpoint);
    try stdout.print("{s}: {d} tensors\n", .{ argv[1], checkpoint.tensors.count() });
    return 0;
}

/// `checkpoint`'s tensors as one safetensors file: the 8-byte header length, the JSON header of names,
/// dtypes, shapes and offsets, then the tensors' bytes in the map's own order.
fn writeShard(arena: std.mem.Allocator, io: Io, dir: Io.Dir, name: []const u8, checkpoint: *const safetensors.Checkpoint) !void {
    var json: std.ArrayList(u8) = .empty;
    var data: std.ArrayList(u8) = .empty;
    try json.appendSlice(arena, "{");
    var first = true;
    var it = checkpoint.tensors.iterator();
    var offset: u64 = 0;
    while (it.next()) |entry| {
        const t = entry.value_ptr;
        if (!first) try json.appendSlice(arena, ",");
        first = false;
        try json.print(arena, "\"{s}\":{{\"dtype\":\"{t}\",\"shape\":[", .{ entry.key_ptr.*, t.dtype });
        for (t.shape, 0..) |dim, i| {
            if (i != 0) try json.appendSlice(arena, ",");
            try json.print(arena, "{d}", .{dim});
        }
        try json.print(arena, "],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + t.data.len });
        offset += t.data.len;
        try data.appendSlice(arena, t.data);
    }
    try json.appendSlice(arena, "}");
    const shard = try dir.createFile(io, name, .{});
    defer shard.close(io);
    var head: [8]u8 = undefined;
    std.mem.writeInt(u64, &head, @intCast(json.items.len), .little);
    try shard.writeStreamingAll(io, &head);
    try shard.writeStreamingAll(io, json.items);
    try shard.writeStreamingAll(io, data.items);
}

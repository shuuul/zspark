//! `zspark stack`'s command line to its children: names, ports, waves, the router's words, the remote lines.

const std = @import("std");
const testing = std.testing;
const plan_file = @import("../../src/stack/plan.zig");
const command_file = @import("../../src/stack/command.zig");

const hf_flash = "/home/example/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300";

fn joined(arena: std.mem.Allocator, argv: []const []const u8) []const u8 {
    return std.mem.join(arena, " ", argv) catch @panic("oom");
}

test "a checkpoint's name is its directory's, or a Hugging Face cache's repository's" {
    try testing.expectEqualStrings("Qwen3.8-Flash-Next-FP4-Local", plan_file.modelName(hf_flash));
    try testing.expectEqualStrings("Qwen3.8-Flash-Next-FP4-Local", plan_file.modelName(hf_flash ++ "/"));
    try testing.expectEqualStrings("Unlimited-OCR", plan_file.modelName("/home/example/models/Unlimited-OCR"));
    try testing.expectEqualStrings("Qwen3-ASR-1.7B", plan_file.modelName("Qwen3-ASR-1.7B/"));
    try testing.expectEqualStrings("abc", plan_file.modelName("/x/snapshots/abc")); // no models-- above it
}

test "the command line's refusals" {
    var said: []const u8 = "";
    try testing.expect(plan_file.parse(&.{}, &said) == null);
    try testing.expectEqualStrings("no FLASH_DIR", said);
    try testing.expect(plan_file.parse(&.{ "a", "b" }, &said) == null);
    try testing.expect(plan_file.parse(&.{ "a", "--ocr" }, &said) == null);
    try testing.expect(plan_file.parse(&.{ "a", "--parallel", "2" }, &said) == null);
    try testing.expect(plan_file.parse(&.{ "a", "--peer", "spark2" }, &said) == null);
    try testing.expect(plan_file.parse(&.{ "a", "--listen", "8888" }, &said) == null);
    const options = plan_file.parse(&.{ "a", "--port", "19000", "--", "--parallel", "2", "--ocr", "x" }, &said).?;
    try testing.expectEqual(@as(u16, 19000), options.port);
    try testing.expect(options.ocr == null); // after --, Flash Next's
    try testing.expectEqual(@as(usize, 4), options.serve_flags.len);
}

test "one Spark: OCR, ASR, Flash Next, the router, each its own wave, on loopback ports" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var said: []const u8 = "";
    const options = plan_file.parse(&.{ hf_flash, "--ocr", "/m/Unlimited-OCR", "--asr", "/m/Qwen3-ASR-1.7B", "--listen", "0.0.0.0:8888", "--", "--parallel", "2" }, &said).?;
    const plan = try plan_file.build(arena, options, .{ .zspark = "/bin/zspark", .router = "/bin/zspark-router" });
    try testing.expectEqual(@as(usize, 4), plan.waves.len);
    for (plan.waves) |wave| try testing.expectEqual(@as(usize, 1), wave.len);
    try testing.expectEqualStrings("/bin/zspark serve /m/Unlimited-OCR --host 127.0.0.1 --port 18101 --name Unlimited-OCR --parallel 1", joined(arena, plan.waves[0][0].argv));
    try testing.expectEqualStrings("/bin/zspark serve /m/Qwen3-ASR-1.7B --host 127.0.0.1 --port 18102 --name Qwen3-ASR-1.7B --parallel 1", joined(arena, plan.waves[1][0].argv));
    try testing.expectEqualStrings("/bin/zspark serve " ++ hf_flash ++ " --host 127.0.0.1 --port 18103 --name Qwen3.8-Flash-Next-FP4-Local --parallel 2", joined(arena, plan.waves[2][0].argv));
    try testing.expect(plan.chosen_parallel == null); // the flags gave one
    try testing.expectEqualStrings("/bin/zspark-router --listen 0.0.0.0:8888 --default Qwen3.8-Flash-Next-FP4-Local --parallel 2 --parallel-of Unlimited-OCR=1 --audio Qwen3-ASR-1.7B --parallel-of Qwen3-ASR-1.7B=1 " ++
        "Qwen3.8-Flash-Next-FP4-Local=127.0.0.1:18103 Unlimited-OCR=127.0.0.1:18101 Qwen3-ASR-1.7B=127.0.0.1:18102", joined(arena, plan.waves[3][0].argv));
    for (plan.waves) |wave| try testing.expect(wave[0].health != null and !wave[0].remote);
}

test "with no --parallel, Flash Next takes 2 beside both small models on one Spark, 3 beside one, serve's own alone, 4 on two" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var said: []const u8 = "";
    const both = try plan_file.build(arena, plan_file.parse(&.{ "/m/flash", "--ocr", "/m/Unlimited-OCR", "--asr", "/m/Qwen3-ASR-1.7B" }, &said).?, .{ .zspark = "z", .router = "r" });
    try testing.expectEqual(@as(?u8, 2), both.chosen_parallel);
    try testing.expectEqualStrings("z serve /m/flash --host 127.0.0.1 --port 18103 --name flash --parallel 2", joined(arena, both.waves[2][0].argv));
    try testing.expect(std.mem.startsWith(u8, joined(arena, both.waves[3][0].argv), "r --listen 127.0.0.1:8888 --default flash --parallel 2 "));
    const one = try plan_file.build(arena, plan_file.parse(&.{ "/m/flash", "--asr", "/m/Qwen3-ASR-1.7B", "--", "--context", "65536" }, &said).?, .{ .zspark = "z", .router = "r" });
    try testing.expectEqual(@as(?u8, 3), one.chosen_parallel);
    try testing.expectEqualStrings("z serve /m/flash --host 127.0.0.1 --port 18103 --name flash --parallel 3 --context 65536", joined(arena, one.waves[1][0].argv));
    // --asr-name keeps a build under its base's id (spec 140 WS-05)
    const named = try plan_file.build(arena, plan_file.parse(&.{ "/m/flash", "--asr", "/m/Qwen3-ASR-1.7B-NVFP4-W4A16", "--asr-name", "Qwen3-ASR-1.7B" }, &said).?, .{ .zspark = "z", .router = "r" });
    try testing.expectEqualStrings("z serve /m/Qwen3-ASR-1.7B-NVFP4-W4A16 --host 127.0.0.1 --port 18102 --name Qwen3-ASR-1.7B --parallel 1", joined(arena, named.waves[0][0].argv));
    try testing.expect(std.mem.indexOf(u8, joined(arena, named.waves[2][0].argv), "--audio Qwen3-ASR-1.7B ") != null);
    const alone = try plan_file.build(arena, plan_file.parse(&.{"/m/flash"}, &said).?, .{ .zspark = "z", .router = "r" });
    try testing.expect(alone.chosen_parallel == null);
    const two = try plan_file.build(arena, plan_file.parse(&.{ "/m/flash", "--ocr", "/m/o", "--asr", "/m/a", "--peer", "198.51.100.2" }, &said).?, .{ .zspark = "z", .router = "r" });
    try testing.expectEqual(@as(?u8, 4), two.chosen_parallel);
    // both ranks run 4 streams, not two ranks' serve's 1, and the router holds Flash Next to that
    try testing.expect(std.mem.indexOf(u8, joined(arena, two.waves[1][0].argv), "--parallel'\\'' '\\''4") != null); // rank 1, quoted twice under ssh
    try testing.expect(std.mem.indexOf(u8, joined(arena, two.waves[1][1].argv), " --parallel 4 ") != null);
    try testing.expect(std.mem.startsWith(u8, joined(arena, two.waves[2][0].argv), "r --listen 127.0.0.1:8888 --default flash --parallel 4 "));
}

test "Flash Next alone keeps the name its own --name gives" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var said: []const u8 = "";
    const options = plan_file.parse(&.{ "/m/flash", "--", "--name", "flash" }, &said).?;
    const plan = try plan_file.build(arena, options, .{ .zspark = "z", .router = "r" });
    try testing.expectEqual(@as(usize, 2), plan.waves.len);
    try testing.expectEqualStrings("z serve /m/flash --host 127.0.0.1 --port 18103 --name flash", joined(arena, plan.waves[0][0].argv));
    try testing.expectEqualStrings("r --listen 127.0.0.1:8888 --default flash --parallel 4 flash=127.0.0.1:18103", joined(arena, plan.waves[1][0].argv));
}

test "two Sparks: OCR here and ASR there at once, then rank 1 there and rank 0 here, then the router" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var said: []const u8 = "";
    const options = plan_file.parse(&.{ hf_flash, "--ocr", "/m/Unlimited-OCR", "--asr", "/m/Qwen3-ASR-1.7B", "--peer", "198.51.100.2", "--peer-dir", "/m/Qwen3.8-Flash-Next-FP4-Local" }, &said).?;
    const plan = try plan_file.build(arena, options, .{ .zspark = "/z", .router = "/r", .environ_there = &.{"NCCL_IB_HCA=a,b"} });
    try testing.expectEqual(@as(usize, 3), plan.waves.len);
    try testing.expectEqual(@as(usize, 2), plan.waves[0].len);
    const asr = plan.waves[0][1];
    try testing.expect(asr.remote);
    try testing.expectEqualStrings("ssh -o BatchMode=yes 198.51.100.2", joined(arena, asr.argv[0..4]));
    try testing.expect(std.mem.indexOf(u8, asr.argv[4], "'\\''/z'\\'' '\\''serve'\\'' '\\''/m/Qwen3-ASR-1.7B'\\'' '\\''--host'\\'' '\\''198.51.100.2'\\''") != null);
    try testing.expectEqual(@as(u16, 18102), asr.health.?.ip4.port);
    const rank1 = plan.waves[1][0];
    const rank0 = plan.waves[1][1];
    try testing.expectEqualStrings("Qwen3.8-Flash-Next-FP4-Local rank 1", rank1.name);
    try testing.expect(rank1.remote and rank1.health == null);
    try testing.expect(std.mem.indexOf(u8, rank1.argv[4], "env '\\''NCCL_IB_HCA=a,b'\\''") != null);
    try testing.expect(std.mem.indexOf(u8, rank1.argv[4], "'\\''/m/Qwen3.8-Flash-Next-FP4-Local'\\'' '\\''--rank'\\'' '\\''1'\\'' '\\''--at'\\'' '\\''198.51.100.2:29551'\\''") != null);
    try testing.expectEqualStrings("/z serve " ++ hf_flash ++ " --host 127.0.0.1 --port 18103 --name Qwen3.8-Flash-Next-FP4-Local --parallel 4 --rank 0 --at 198.51.100.2:29551", joined(arena, rank0.argv));
    try testing.expect(std.mem.endsWith(u8, joined(arena, plan.waves[2][0].argv), "Qwen3-ASR-1.7B=198.51.100.2:18102"));
}

test "a two-rank ASR starts both ranks before awaiting health, before Flash Next" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const options: plan_file.Options = .{ .flash = "/m/flash", .ocr = "/m/ocr", .asr = "/m/renamed-asr", .peer = "198.51.100.2" };
    const bin: plan_file.Binaries = .{ .zspark = "/z", .router = "/r", .asr_split = true };
    const plan = try plan_file.build(arena, options, bin);
    try testing.expectEqual(@as(usize, 4), plan.waves.len);
    const asr = plan.waves[1];
    try testing.expectEqual(@as(usize, 2), asr.len);
    try testing.expect(asr[0].remote and asr[0].health == null);
    try testing.expect(!asr[1].remote and asr[1].health != null);
    try testing.expectEqual(@as(u16, options.port + 1), asr[1].health.?.ip4.port);
    try testing.expect(plan.mps_here and !plan.mps_there);
    // a split ASR's ranks carry no --parallel (the model serialises anyway) and no MPS cap:
    // rank 1 gets only the link env, rank 0 only this Spark's
    for (asr) |child| {
        const argv = joined(arena, child.argv);
        try testing.expect(std.mem.indexOf(u8, argv, "--parallel") == null);
        try testing.expect(std.mem.indexOf(u8, argv, "CUDA_MPS") == null);
    }
    const flash = plan.waves[2];
    try testing.expect(flash[0].remote and flash[0].health == null);
    try testing.expect(!flash[1].remote and flash[1].health != null);
    try testing.expectEqual(@as(u16, options.port + 2), flash[1].health.?.ip4.port);
    var no_port = options;
    no_port.at = std.math.maxInt(u16);
    try testing.expectError(error.InvalidPort, plan_file.build(arena, no_port, bin));
    var alone = options;
    alone.peer = null;
    const single = try plan_file.build(arena, alone, bin);
    try testing.expectEqual(@as(usize, 1), single.waves[1].len);
    try testing.expect(!single.waves[1][0].remote and single.waves[1][0].health != null);
}

test "a word after its key, as `ip route get` prints them" {
    const line = "198.51.100.2 dev eth0 src 198.51.100.1 uid 1000 \\    cache";
    try testing.expectEqualStrings("eth0", command_file.wordAfter(line, "dev").?);
    try testing.expectEqualStrings("198.51.100.1", command_file.wordAfter(line, "src").?);
    try testing.expect(command_file.wordAfter(line, "via") == null);
}

test "MPS: the small models at --mps of the SMs and Flash Next at all, clients of the stack's own daemon; --mps 0 none" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var said: []const u8 = "";
    var base: std.process.Environ.Map = .init(arena);
    try base.put("PATH", "/usr/bin");
    const bin: plan_file.Binaries = .{ .zspark = "z", .router = "r", .environ = &base };
    const one = try plan_file.build(arena, plan_file.parse(&.{ "/m/flash", "--ocr", "/m/o", "--asr", "/m/a" }, &said).?, bin);
    try testing.expectEqualStrings("/tmp/zspark-mps-18101", one.mps_dir.?);
    try testing.expect(one.mps_here and !one.mps_there);
    for ([_]usize{ 0, 1 }) |w| {
        const env = one.waves[w][0].environ.?;
        try testing.expectEqualStrings("25", env.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE").?);
        try testing.expectEqualStrings("1", env.get("CUDA_MPS_CLIENT_PRIORITY").?);
        try testing.expectEqualStrings("/tmp/zspark-mps-18101/pipe", env.get("CUDA_MPS_PIPE_DIRECTORY").?);
        try testing.expectEqualStrings("/usr/bin", env.get("PATH").?); // the stack's own kept
    }
    const flash = one.waves[2][0].environ.?;
    try testing.expectEqualStrings("/tmp/zspark-mps-18101/pipe", flash.get("CUDA_MPS_PIPE_DIRECTORY").?);
    try testing.expect(flash.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") == null and flash.get("CUDA_MPS_CLIENT_PRIORITY") == null);
    try testing.expect(one.waves[3][0].environ == null); // the router runs nothing on the device
    try testing.expectEqualStrings("export CUDA_MPS_PIPE_DIRECTORY=/tmp/zspark-mps-18101/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/zspark-mps-18101/log; " ++
        "mkdir -p /tmp/zspark-mps-18101/pipe /tmp/zspark-mps-18101/log && { echo 'quit -t 20' | timeout 60 nvidia-cuda-mps-control >/dev/null 2>&1; " ++
        "nvidia-cuda-mps-control -d </dev/null >/dev/null 2>&1; }", try plan_file.mpsLine(arena, one.mps_dir.?, true));

    const off = try plan_file.build(arena, plan_file.parse(&.{ "/m/flash", "--ocr", "/m/o", "--asr", "/m/a", "--mps", "0" }, &said).?, bin);
    try testing.expect(off.mps_dir == null);
    for (off.waves) |wave| try testing.expect(wave[0].environ == null);
    const alone = try plan_file.build(arena, plan_file.parse(&.{"/m/flash"}, &said).?, bin);
    try testing.expect(alone.mps_dir == null and alone.waves[0][0].environ == null); // nothing beside it to share with
    try testing.expect(plan_file.parse(&.{ "/m/flash", "--mps", "101" }, &said) == null);

    // two Sparks: a daemon on each, OCR with rank 0 here, ASR with rank 1 there
    var here: std.process.Environ.Map = .init(arena);
    try here.put("NCCL_SOCKET_IFNAME", "eth0");
    const two = try plan_file.build(arena, plan_file.parse(&.{ "/m/flash", "--ocr", "/m/o", "--asr", "/m/a", "--peer", "198.51.100.2", "--mps", "50" }, &said).?, .{ .zspark = "z", .router = "r", .environ = &base, .environ_here = &here, .environ_there = &.{"NCCL_IB_HCA=a,b"} });
    try testing.expect(two.mps_here and two.mps_there);
    try testing.expectEqualStrings("50", two.waves[0][0].environ.?.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE").?);
    try testing.expect(std.mem.indexOf(u8, two.waves[0][1].argv[4], "env '\\''CUDA_MPS_PIPE_DIRECTORY=/tmp/zspark-mps-18101/pipe'\\'' '\\''CUDA_MPS_LOG_DIRECTORY=/tmp/zspark-mps-18101/log'\\'' '\\''CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50'\\'' '\\''CUDA_MPS_CLIENT_PRIORITY=1'\\''") != null);
    try testing.expect(std.mem.indexOf(u8, two.waves[1][0].argv[4], "'\\''NCCL_IB_HCA=a,b'\\'' '\\''CUDA_MPS_PIPE_DIRECTORY=/tmp/zspark-mps-18101/pipe'\\''") != null);
    try testing.expect(std.mem.indexOf(u8, two.waves[1][0].argv[4], "PERCENTAGE") == null);
    const rank0 = two.waves[1][1].environ.?;
    try testing.expectEqualStrings("eth0", rank0.get("NCCL_SOCKET_IFNAME").?);
    try testing.expectEqualStrings("/tmp/zspark-mps-18101/pipe", rank0.get("CUDA_MPS_PIPE_DIRECTORY").?);
}

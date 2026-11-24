//! TensorFold's vendored host as one definition (spec 168): the modules `build.zig`'s `Host` compiles and the
//! files `tools/vendor_import.sh` takes from a checkout of upstream, each by name and by path below
//! `zig/vendor/tensorfold/`. The two were kept apart until this file; an upgrade missed `nemotron_limits` and
//! once built `core.startup_receipt` as a second module, beside `profile`'s own file (spec 167). A module's
//! `name` is the name its importers give `@import`; its `root` is the file at that path, a stub's `stub` the
//! text a `WriteFiles` module is made from. A name that imports resolve to but no module defines fails this
//! file's own check (compile time, and `main -- check`), and so does a file two modules would own, an import
//! naming a module that is not here, or a cycle of imports.
//!
//! `zig run tools/host_descriptor.zig -- all` prints `SECTION PATH` a line for the shell importer;
//! `main -- check` prints each problem and exits 1. `build.zig` reads the declarations.

const std = @import("std");
const Io = std.Io;

/// Where the vendored tree stands in this repository.
pub const vendored = "zig/vendor/tensorfold/";

/// This repository's files at the vendored directory's top; everything else there is upstream's.
pub const own = [_][]const u8{ "MANIFEST.sha256", "UPSTREAM", "PATCHES.md", "AGENTS.md" };

/// The notices taken whole, wherever upstream keeps them.
pub const notices = [_][]const u8{ "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md" };

/// Directories taken whole: what test-golden reads, and the licence texts.
pub const whole = [_][]const u8{ "zig/tests/server/golden", "zig/tests/server/fixtures", "LICENSES" };

/// Files the import takes beside the roots' closure: what the build compiles or reads that no `@import`
/// reaches (the grammar library's sources — `tf_grammar.cc` includes `tf_budget.h` and `tf_compile.h` and links `tf_budget.cc`,
/// which nothing walks since the closure is Zig's alone; the tests of a helper module, which nothing imports).
pub const beside = [_][]const u8{
    "zig/src/grammar/tf_budget.h",
    "zig/src/grammar/tf_budget.cc",
    "zig/src/grammar/tf_compile.h",
    "zig/src/grammar/tf_grammar.cc",
    "zig/src/core/safetensors_copied_test.zig",
    "zig/src/core/safetensors_security_test.zig",
    "zig/src/core/benchmark_group_test.zig",
    "zig/src/core/checkpoint_index.zig",
    "zig/src/core/checkpoint_file_cache.zig",
    "zig/src/core/checkpoint_drop.zig",
    "zig/src/core/exl3_format.zig",
    "zig/src/core/exlr8_codec.zig",
    "zig/src/core/exlr8_experts.zig",
    "zig/src/core/xxhash64.zig",
};

/// Names upstream's files import that nothing taken defines, so they stay unresolved on purpose; `UPSTREAM`
/// there says why each stands (2026-11-12).
pub const unresolved = [_][]const u8{ "native_engines", "build_options", "qwen27_masks", "qwen27_mask_fake" };

/// The module names handed to every `.shared` module, as upstream's `zig/build/sys.zig`, `save_guard.zig`,
/// `checkpoint_config.zig` hand them to each of its own by one walk of the build; `profile` and `bf16_round`
/// the same way since 1.2.3 (its `zig/build/profile.zig`, `zig/build/bf16_round.zig`).
pub const shared = [_][]const u8{ "sys", "save_guard", "checkpoint_config", "profile", "bf16_round" };

/// Who builds the module: `.host` is part of the serving host `zig/src/` imports (a field of `build.zig`'s
/// `Host`), `.helper` is built only as another module's import, `.stub` stands in for a family zspark does
/// not serve, `.test_golden` is test-golden's own program.
pub const Role = enum { host, helper, stub, test_golden };

pub const Module = struct {
    /// The name importers give `@import`; where it differs from the file's own name it is the alias the
    /// upstream build defines (`engine_api` for engine_api.zig, `dsv41_chat` for the family's template).
    name: []const u8,
    /// The module's root file below `vendored`, at upstream's path; null for a stub.
    root: ?[]const u8 = null,
    role: Role = .host,
    /// Given every name of `shared` as an import.
    shared: bool = false,
    /// Linked against libc (upstream's modules are; `bf16_round` and the `drama_prompt` stub are not).
    libc: bool = true,
    /// Imports beyond the shared ones, by name of a module here.
    imports: []const []const u8 = &.{},
    /// The text a stub's module is written from; null for a file module.
    stub: ?[]const u8 = null,
};

/// Every module the vendored host builds into, in dependency order where it matters (helpers before their
/// only importer is customary, not required: resolution memoizes). A root here is a file the importer takes;
/// the importer's own set is `roots`, each module's `root` that is not null.
pub const modules = [_]Module{
    // the shared five (upstream's per-module walk, above): the platform layer, the save admission, the
    // checkpoint config reader, the switch registry since 1.2.3, bf16 host rounding since 1.2.3
    .{ .name = "sys", .root = "zig/src/platform/sys.zig", .role = .helper },
    .{ .name = "save_guard", .root = "zig/src/core/save_guard.zig", .role = .helper },
    .{ .name = "checkpoint_config", .root = "zig/src/core/checkpoint_config.zig", .role = .helper },
    .{ .name = "profile", .root = "zig/src/core/profile.zig", .role = .helper },
    .{ .name = "bf16_round", .root = "zig/src/core/bf16_round.zig", .role = .helper, .libc = false },
    .{ .name = "checkpoint_eos", .root = "zig/src/core/tokenizer/eos.zig", .role = .helper, .libc = false },
    // the serving host's modules, as upstream's `zig/tests/server/build.sh` names them; `json` a module of
    // its own since 1.0.12 (upstream's `zig/build/server.zig`), `living` (Living Weights, unserved) the
    // server's since 1.1.1
    .{ .name = "lanes", .root = "zig/src/core/lanes/lanes.zig", .shared = true },
    .{ .name = "engine_api", .root = "zig/src/core/engine_api.zig", .shared = true, .imports = &.{"lanes"} },
    .{ .name = "tokenizer", .root = "zig/src/core/tokenizer/tokenizer.zig", .shared = true, .imports = &.{"checkpoint_eos"} },
    .{ .name = "template", .root = "zig/src/core/template/template.zig", .shared = true },
    .{ .name = "json", .root = "zig/src/server/json.zig", .shared = true },
    .{ .name = "living", .root = "zig/src/living/living.zig", .role = .helper, .shared = true },
    .{ .name = "server", .root = "zig/src/server/root.zig", .shared = true, .imports = &.{ "engine_api", "tokenizer", "template", "json", "lanes", "living", "dsv41_chat", "uocr_chat", "nemotron_limits", "drama_prompt", "core" } },
    // since 1.1.4 (upstream's `zig/build/server.zig`): `hf_text.zig` renders DeepSeek V4.1's chat natively
    // through its family's template
    .{ .name = "dsv41_chat", .root = "zig/src/families/deepseek_v41/serve/template.zig", .role = .helper, .shared = true, .imports = &.{"json"} },
    .{ .name = "uocr_chat", .root = "zig/src/families/uocr/chat.zig", .role = .helper, .libc = false },
    // since 1.2.7 (upstream's `zig/build/server.zig`): status_routes.zig reads the Nemotron family's
    // position limits by this name
    .{ .name = "nemotron_limits", .root = "zig/src/families/nemotron/limits.zig", .role = .helper, .shared = true },
    // `zig/src/core/picture/`: PNG to RGB, the orientation tags, alpha over white (spec 019 WS-07)
    .{ .name = "picture", .root = "zig/src/core/picture/picture.zig", .shared = true },
    // the `core` stub's helpers: since 1.2.3 server/slide_model.zig reads a shard's header through
    // `core.safetensors` and benchmark_group_fixture.zig through `core.benchmark_group`; since 1.2.7
    // openai.zig and server/startup_receipt.zig read `core.startup_receipt`, which is `profile`'s own file
    // re-exported — a module here would own the file a second time, so the stub names the `profile` module
    .{ .name = "tf_safetensors", .root = "zig/src/core/safetensors.zig", .role = .helper, .shared = true },
    .{ .name = "tf_benchmark_group", .root = "zig/src/core/benchmark_group.zig", .role = .helper, .shared = true },
    // Since 1.3.3 the server's execution witness reads core.digest_cache.
    .{ .name = "tf_digest_cache", .root = "zig/src/core/digest_cache.zig", .role = .helper, .shared = true },
    // Since 1.3.4 the speech route's audio encoder reads core.audio for resampling.
    .{ .name = "tf_audio", .root = "zig/src/core/audio.zig", .role = .helper, .shared = true },
    // `drama_prompt` (1.1.4's speech.zig and drama_voice.zig: Drama's zig/src/families/drama3/prompt.zig,
    // whose closure is that family's engine): no model of zspark speaks, so a stub that links and builds
    // nothing stands for it (spec 141's decision of 2026-10-26; 1.1.5's `build` returns `Built`, the frames
    // and the reference prompt's rows)
    .{ .name = "drama_prompt", .role = .stub, .libc = false, .stub =
    \\pub const Reference = struct { transcript: []const u8, codes: []const [10]i32 };
    \\pub const Built = struct { frames: [][11]i32, reference_rows: u32, text_rows: u32 = 0 };
    \\pub fn build(gpa: @import("std").mem.Allocator, tok: anytype, text: []const u8, refs: []const Reference) error{Unsupported}!Built {
    \\    _ = .{ gpa, tok, text, refs };
    \\    return error.Unsupported;
    \\}
    \\
    },
    // upstream's `zig/src/core/root.zig` re-exports every file of zig/src/core/ (300+ into the closure);
    // keep only the host's uses, sharing `profile`'s receipt and, since 1.2.8, the OCR routes' picture module
    .{ .name = "core", .role = .stub, .imports = &.{ "tf_safetensors", "tf_benchmark_group", "tf_digest_cache", "tf_audio", "profile", "picture" }, .stub =
    \\pub const audio = @import("tf_audio");
    \\pub const safetensors = @import("tf_safetensors");
    \\pub const benchmark_group = @import("tf_benchmark_group");
    \\pub const digest_cache = @import("tf_digest_cache");
    \\pub const startup_receipt = @import("profile").startup_receipt;
    \\pub const picture = @import("picture");
    \\
    },
    // test-golden's two programs (upstream's `zig/build/golden.zig`), compiled Debug by `Host.golden`
    .{ .name = "fake_serve", .root = "zig/tests/server/fake_serve.zig", .role = .test_golden, .shared = true, .imports = &.{ "server", "engine_api", "tokenizer", "template" } },
    .{ .name = "golden_check", .root = "zig/tests/server/golden_check.zig", .role = .test_golden, .shared = true },
};

const root_count = blk: {
    var n: usize = 0;
    for (modules) |module| {
        if (module.root != null) n += 1;
    }
    break :blk n;
};

/// Every module's root file, in the order of `modules`: what the importer's closure starts from.
pub const roots: [root_count][]const u8 = blk: {
    var out: [root_count][]const u8 = undefined;
    var i: usize = 0;
    for (modules) |module| {
        if (module.root) |root| {
            out[i] = root;
            i += 1;
        }
    }
    break :blk out;
};

const module_count = modules.len;

/// Every module's name, in the order of `modules`.
pub const names: [module_count][]const u8 = blk: {
    var out: [module_count][]const u8 = undefined;
    for (modules, 0..) |module, i| out[i] = module.name;
    break :blk out;
};

/// The module `name` names, or null.
pub fn find(name: []const u8) ?Module {
    for (modules) |module| {
        if (std.mem.eql(u8, module.name, name)) return module;
    }
    return null;
}

/// What is wrong, and the module it names (the offender, or the module whose import is).
pub const Problem = struct { message: []const u8, offender: []const u8 };

/// The first problem the lists have, or null: a name or a root held twice, a shared or import name no
/// module defines, a module importing itself or a cycle of imports, a stub without text or a root module
/// with it. Comptime-evaluable; the `comptime` block below holds `modules` to it, so the build fails where
/// it is read.
pub fn firstProblem(mods: []const Module, shared_names: []const []const u8) ?Problem {
    @setEvalBranchQuota(100_000);
    for (mods, 0..) |a, i| {
        for (mods[i + 1 ..]) |b| {
            if (std.mem.eql(u8, a.name, b.name)) return .{ .message = "two modules share a name", .offender = a.name };
            if (a.root != null and std.mem.eql(u8, a.root.?, b.root orelse continue))
                return .{ .message = "two modules share a root (a file is one module's; a stub names the module that owns it)", .offender = a.name };
        }
        if (a.root == null and a.stub == null) return .{ .message = "has no root and no stub text", .offender = a.name };
        if (a.root != null and a.stub != null) return .{ .message = "has a root and stub text: one or the other", .offender = a.name };
        if ((a.role == .stub) != (a.stub != null)) return .{ .message = "is a stub without text, or a file module with it", .offender = a.name };
        if (hasName(shared_names, a.name) and a.shared)
            return .{ .message = "is a shared module and is marked .shared: it cannot import itself", .offender = a.name };
    }
    if (mods.len > 64) return .{ .message = "the import-cycle check holds at most 64 modules", .offender = mods[0].name };
    for (mods) |module| {
        for (module.imports) |name| {
            if (std.mem.eql(u8, name, module.name)) return .{ .message = "imports itself", .offender = name };
            if (!defined(mods, name)) return .{ .message = "imports a module none defines (an upstream name left unresolved goes in `unresolved`)", .offender = name };
        }
    }
    // before the walk: every shared name must resolve, or visit() reads past the slice's end
    for (shared_names) |name| {
        if (!defined(mods, name)) return .{ .message = "is a shared name no module defines", .offender = name };
    }
    var state: [64]u8 = @splat(0);
    for (mods, 0..) |module, i| {
        if (visit(mods, shared_names, &state, i)) return .{ .message = "is in a cycle of imports", .offender = module.name };
    }
    return null;
}

/// Depth-first: state 0 unvisited, 1 on the way down, 2 done. True where a module meets itself again.
fn visit(mods: []const Module, shared_names: []const []const u8, state: *[64]u8, i: usize) bool {
    if (state[i] == 2) return false;
    if (state[i] == 1) return true;
    state[i] = 1;
    if (mods[i].shared) {
        for (shared_names) |next| {
            if (visit(mods, shared_names, state, indexOfName(mods, next))) return true;
        }
    }
    for (mods[i].imports) |next| {
        if (visit(mods, shared_names, state, indexOfName(mods, next))) return true;
    }
    state[i] = 2;
    return false;
}

fn indexOfName(mods: []const Module, name: []const u8) usize {
    for (mods, 0..) |module, i| {
        if (std.mem.eql(u8, module.name, name)) return i;
    }
    return mods.len;
}

fn hasName(list: []const []const u8, name: []const u8) bool {
    for (list) |each| {
        if (std.mem.eql(u8, each, name)) return true;
    }
    return false;
}

fn defined(mods: []const Module, name: []const u8) bool {
    for (mods) |module| {
        if (std.mem.eql(u8, module.name, name)) return true;
    }
    return false;
}

comptime {
    if (firstProblem(&modules, &shared)) |problem| @compileError(problem.offender ++ ": " ++ problem.message);
}

/// One `SECTION PATH` line a section, for `tools/vendor_import.sh` and `tools/vendor_c.sh`.
fn emit(writer: *Io.Writer) !void {
    const sections = .{
        .{ "roots", &roots },
        .{ "beside", &beside },
        .{ "whole", &whole },
        .{ "notices", &notices },
        .{ "own", &own },
        .{ "modules", &names },
        .{ "shared", &shared },
        .{ "unresolved", &unresolved },
    };
    inline for (sections) |section| {
        for (section[1]) |item| try writer.print("{s} {s}\n", .{ section[0], item });
    }
}

pub fn main(init: std.process.Init) !u8 {
    var buffer: [4096]u8 = undefined;
    var stdout_file: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const stdout = &stdout_file.interface;
    defer stdout.flush() catch {};

    const arena = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(arena);
    if (argv.len != 2 or !std.mem.eql(u8, argv[1], "all") and !std.mem.eql(u8, argv[1], "check")) {
        try stdout.writeAll("usage: host_descriptor SECTION\n  all     every list, `SECTION PATH` a line\n  check   the problems the lists have, if any\n");
        return 2;
    }
    if (std.mem.eql(u8, argv[1], "check")) {
        if (firstProblem(&modules, &shared)) |problem| {
            try stdout.print("{s}: {s}\n", .{ problem.offender, problem.message });
            return 1;
        }
        return 0;
    }
    try emit(stdout);
    return 0;
}

// The tests hold the descriptor's own invariants, and that a bad one is refused by the same function.

const testing = std.testing;

test "a second owner of one file, an unknown import, a bad shared name and a cycle are each refused" {
    const dup_root = [_]Module{
        .{ .name = "a", .root = "x.zig" },
        .{ .name = "b", .root = "x.zig" },
    };
    try testing.expect(firstProblem(@constCast(&dup_root), &.{}) != null);
    const unknown = [_]Module{.{ .name = "a", .root = "x.zig", .imports = &.{"missing"} }};
    try testing.expect(firstProblem(@constCast(&unknown), &.{}) != null);
    const cyclic = [_]Module{
        .{ .name = "a", .root = "a.zig", .imports = &.{"b"} },
        .{ .name = "b", .root = "b.zig", .imports = &.{"a"} },
    };
    try testing.expect(firstProblem(@constCast(&cyclic), &.{}) != null);
    const self_shared = [_]Module{.{ .name = "a", .root = "a.zig", .shared = true }};
    try testing.expect(firstProblem(@constCast(&self_shared), &.{"a"}) != null);
    const bare_stub = [_]Module{.{ .name = "a", .role = .stub }};
    try testing.expect(firstProblem(@constCast(&bare_stub), &.{}) != null);

    // a shared name no module defines must refuse before the cycle walk reads past the slice's end
    const bad_shared = [_]Module{.{ .name = "a", .root = "a.zig" }};
    try testing.expect(firstProblem(@constCast(&bad_shared), &.{"ghost"}) != null);
}

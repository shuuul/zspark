const std = @import("std");

/// The package manifest, imported at comptime: `version` is the calendar version (spec 170) the stamp's
/// command writes beside the commit, the one place it is named.
const zon = @import("build.zig.zon");

/// CUDA sources compiled by nvcc and linked into the executable. Each exposes `extern "C"` launchers
/// declared in zig/src/cuda/cuda.zig.
const Kernel = struct {
    source: []const u8,
    /// Extra nvcc flags; a kernel whose replay must reproduce its serial steps turns FMA contraction off.
    flags: []const []const u8 = &.{},
    /// Built on CUTLASS: its headers (the `cutlass` package of build.zig.zon) go on the include path.
    cutlass: bool = false,
};

/// Kernels only tests launch (`docs/architecture.md`, rule 4): linked into the test binaries, not the executable.
const test_kernels = [_]Kernel{
    // the hand-written BF16 matmul the GEMM's bits are held against
    .{ .source = "zig/tests/kernels/bf16/matmul.cu" },
    // spec 158 WS-02: one planted break a gate of the kernel contract
    .{ .source = "zig/tests/kernels/contract/controls.cu" },
    // spec 070 WS-04: the read-out as one launch, a comparison the forward does not take
};

const kernels = [_]Kernel{
    .{ .source = "zig/kernels/zs/probe.cu" },
    .{ .source = "zig/kernels/zs/host.cu" },
    .{ .source = "zig/kernels/common/plan.cu" },
    .{ .source = "zig/kernels/common/columns.cu" },
    .{ .source = "zig/kernels/common/words.cu" },
    .{ .source = "zig/kernels/common/embed.cu", .flags = &.{"-lineinfo"} },
    .{ .source = "zig/kernels/common/pick.cu" },
    // spec 106 WS-02: the rank-order sum of rows a rank glues
    .{ .source = "zig/kernels/common/rank_sum.cu" },
    // spec 101 WS-02: a row's log-sum-exp and best k, for a request's `logprobs` (TensorFold 1.0.12's)
    .{ .source = "zig/kernels/common/logprob.cu" },
    // spec 136 WS-01: the row glue GLM and Flash Next's DFlash2 launch (RMS norm, branch, SwiGLU)
    .{ .source = "zig/kernels/common/glue.cu" },
    // spec 138 WS-03: LayerNorm with bias at any width, the residual add before it
    .{ .source = "zig/kernels/common/layernorm.cu" },
    // CUTLASS calls constexpr std functions in device code and defaults an annotated constructor (warning 20012).
    .{ .source = "zig/kernels/bf16/gemm.cu", .flags = &.{ "--expt-relaxed-constexpr", "-diag-suppress", "20012" }, .cutlass = true },
    // spec 019 WS-02: the vision tower's linears, bias and GELU in the epilogue
    .{ .source = "zig/kernels/bf16/gemm_bias.cu", .flags = &.{ "--expt-relaxed-constexpr", "-diag-suppress", "20012" }, .cutlass = true },
    // spec 137 WS-04: bf16 routed experts grouped by common/plan.cu's plan, SwiGLU in the up product's epilogue
    .{ .source = "zig/kernels/bf16/experts.cu" },
    .{ .source = "zig/kernels/qwen4/attention.cu" },
    // spec 138 WS-04: the text attention two families launch, moved from qwen4/attention.cu
    .{ .source = "zig/kernels/common/attention.cu" },
    // spec 019 WS-02: the vision tower's steps between its linears, and its attention by whole image
    .{ .source = "zig/kernels/qwen4/vision.cu" },
    .{ .source = "zig/kernels/qwen4/vision_attention.cu" },
    // spec 137 WS-03: Unlimited-OCR's SAM and CLIP between their linears, and their attention
    .{ .source = "zig/kernels/unlimited_ocr/vision.cu" },
    .{ .source = "zig/kernels/unlimited_ocr/vision_attention.cu" },
    // spec 138 WS-03: Qwen3-ASRs audio tower, its convolutions patches and positions, its attention in blocks
    .{ .source = "zig/kernels/qwen3_asr/tower.cu" },
    .{ .source = "zig/kernels/qwen3_asr/tower_attention.cu" },
    // spec 151 WS-03: Nemotron's perception encoder — qk LayerNorm and RoPE on the qkv rows, a chunk a segment
    .{ .source = "zig/kernels/nemotron/perception.cu" },
    .{ .source = "zig/kernels/nemotron/perception_attention.cu" },
    .{ .source = "zig/kernels/nemotron/speaker.cu" },
    // spec 142 WS-04 (138 WS-04's): the text attention's prep Qwen3-ASR and Unlimited-OCR take (q and k norms, RoPE, the caches)
    .{ .source = "zig/kernels/common/attention_prep.cu" },
    // spec 105 WS-05: a prompt window's rows, a row a block over its chunks, the merge in registers
    .{ .source = "zig/kernels/common/attention_window.cu" },
    .{ .source = "zig/kernels/common/attention_group.cu" },
    // spec 018: attention.cu, common/attention.cu and qsa.cu again with the caches behind a table of extents, off where zspark serves
    // (`attention.Cache.table`)
    .{ .source = "zig/kernels/qwen4/attention_table.cu" },
    .{ .source = "zig/kernels/qwen4/qsa_table.cu" },
    // spec 018 WS-B3: a round's rows across streams, off where zspark serves
    .{ .source = "zig/kernels/qwen4/attention_rows.cu" },
    .{ .source = "zig/kernels/qwen4/qsa_rows.cu" },
    // spec 149 WS-02: attention_window.cu's form over a round's rows, a pool's prompt rounds
    .{ .source = "zig/kernels/qwen4/attention_window_rows.cu" },
    // spec 018 WS-B4: the selection's pack and per-row list, for a graph that holds one GEMM a slot
    .{ .source = "zig/kernels/qwen4/select_slots.cu" },
    .{ .source = "zig/kernels/qwen4/gdn_slots.cu", .flags = &.{"--fmad=false"} },
    .{ .source = "zig/kernels/qwen4/ple_slots.cu", .flags = &.{"-lineinfo"} },
    .{ .source = "zig/kernels/qwen4/state_slots.cu" },
    .{ .source = "zig/kernels/qwen4/hc.cu" },
    .{ .source = "zig/kernels/qwen4/partial.cu" },
    .{ .source = "zig/kernels/qwen4/hc_window.cu" },
    .{ .source = "zig/kernels/qwen4/hc_down.cu" },
    .{ .source = "zig/kernels/qwen4/moe.cu" },
    .{ .source = "zig/kernels/qwen4/mtp.cu", .flags = &.{"-lineinfo"} },
    .{ .source = "zig/kernels/qwen4/ple.cu", .flags = &.{"-lineinfo"} },
    .{ .source = "zig/kernels/qwen4/qsa.cu" },
    .{ .source = "zig/kernels/qwen4/pos_add.cu" },
    .{ .source = "zig/kernels/qwen4/state.cu" },
    .{ .source = "zig/kernels/qwen4/gdn.cu", .flags = &.{"--fmad=false"} },
    .{ .source = "zig/kernels/qwen4/gdn_wide.cu", .flags = &.{"--fmad=false"} },
    .{ .source = "zig/kernels/glm5/hc.cu" },
    .{ .source = "zig/kernels/glm5/moe.cu" },
    .{ .source = "zig/kernels/glm5/kda.cu", .flags = &.{"--fmad=false"} },
    .{ .source = "zig/kernels/glm5/kda_quads.cu", .flags = &.{"--fmad=false"} },
    .{ .source = "zig/kernels/glm5/kda_chunk.cu", .flags = &.{"--fmad=false"} },
    .{ .source = "zig/kernels/glm5/kda_state.cu" },
    .{ .source = "zig/kernels/glm5/mla.cu" },
    .{ .source = "zig/kernels/glm5/mla_proj.cu" },
    .{ .source = "zig/kernels/glm5/mla_q8.cu" },
    // spec 151 WS-02: NemotronH's glue (norms, router and top-k, serial feed, routed plan, keyed greedy), taken
    .{ .source = "zig/kernels/nemotron/norms.cu" },
    .{ .source = "zig/kernels/nemotron/route.cu" },
    .{ .source = "zig/kernels/nemotron/ops.cu" },
    .{ .source = "zig/kernels/nemotron/keyed.cu" },
    // spec 151 WS-06: the two ranks' MoE partial fold (the rest of the split is `common/rank_sum.cu`)
    .{ .source = "zig/kernels/nemotron/ranks.cu" },
    // the Mamba-2 conv and scan: upstream's flags; the prompt kernels keep the steps' bits
    .{ .source = "zig/kernels/nemotron/mamba.cu", .flags = &.{ "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true" } },
    .{ .source = "zig/kernels/nemotron/mamba_wide.cu", .flags = &.{ "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true" } },
    // spec 137 WS-04: the Unlimited-OCR decoder's MoE routing and sums, shared by both generations
    .{ .source = "zig/kernels/unlimited_ocr/moe.cu" },
    // spec 147 WS-04: Unlimited-OCR-2's DASH attend, a tile kernel and a step kernel of the same bits
    .{ .source = "zig/kernels/unlimited_ocr/dash_attention.cu" },
    // spec 147 WS-03: Unlimited-OCR-2's DASH router past its GEMM
    .{ .source = "zig/kernels/unlimited_ocr/router.cu" },
    // the host model's sums: a fused multiply-add is a different addition
    .{ .source = "zig/kernels/glm5/index.cu", .flags = &.{"--fmad=false"} },
    // every fused multiply-add is written, so that the host model is the same arithmetic
    .{ .source = "zig/kernels/common/dflash2.cu", .flags = &.{"--fmad=false"} },
    .{ .source = "zig/kernels/common/dflash2_attend.cu" },
    .{ .source = "zig/kernels/nvfp4/abi.cu" },
    .{ .source = "zig/kernels/nvfp4/qmmf.cu" },
    .{ .source = "zig/kernels/nvfp4/qmmf_tile.cu" },
    .{ .source = "zig/kernels/nvfp4/qmm_a8.cu" },
    .{ .source = "zig/kernels/nvfp4/experts.cu" },
    .{ .source = "zig/kernels/nvfp4/experts_wide.cu" },
    // spec 152 WS-09: the decode gate and up two warps a unit where every half unit fits (TF 0afe9dd5)
    .{ .source = "zig/kernels/nvfp4/experts_half.cu" },
    .{ .source = "zig/kernels/nvfp4/experts_tile.cu" },
    // spec 114 WS-03: the W4A4 routed experts (NVFP4 activations on the block-scaled FP4 mma)
    .{ .source = "zig/kernels/nvfp4/experts_quant.cu" },
    .{ .source = "zig/kernels/nvfp4/experts_a4.cu" },
    .{ .source = "zig/kernels/nvfp4/experts_tile_a4.cu" },
    // the routed experts placed for the FP4 mma at load, and their csf_32x2_v1 scales decoded, on the device
    .{ .source = "zig/kernels/nvfp4/experts_pack.cu" },
    .{ .source = "zig/kernels/nvfp4/prompt.cu" },
    // spec 019 WS-08: prompt.cu's loop with a bias and GELU epilogue, the vision tower's quantized linears
    .{ .source = "zig/kernels/nvfp4/prompt_bias.cu" },
};

/// A Spark has ten Cortex-X925 cores and ten Cortex-A725: a native target names the core the compiler happens
/// to run on, the two differ in three tuning features, and the model is part of every compilation's cache key,
/// so a build with nothing changed compiled everything again whenever the other kind of core was met (spec 053:
/// 43 s of the host tier, 73 s of the quick one, since retired, about one run in two). The A725's detection is made the X925's
/// here, the features the machine reports kept; any other CPU, and a target asked for with -Dtarget or -Dcpu,
/// is left as it is.
fn steady(b: *std.Build, target: std.Build.ResolvedTarget) std.Build.ResolvedTarget {
    if (!target.query.isNative() or target.result.cpu.arch != .aarch64) return target;
    const models = &std.Target.aarch64.cpu;
    var cpu = target.result.cpu;
    if (cpu.model != &models.cortex_a725 and cpu.model != &models.cortex_x925) return target;
    if (cpu.model == &models.cortex_a725) {
        cpu.model = &models.cortex_x925;
        cpu.features.addFeature(@backingInt(std.Target.aarch64.Feature.avoid_ldapur));
        cpu.features.addFeature(@backingInt(std.Target.aarch64.Feature.use_fixed_over_scalable_if_equal_cost));
        cpu.features.removeFeature(@backingInt(std.Target.aarch64.Feature.cmp_bcc_fusion));
    }
    var named = target.result;
    named.cpu = cpu;
    const whole = std.Target.Query.fromTarget(&named);
    // The CPU alone is named: the OS, the ABI and the libc stay the machine's own, found as for a native target.
    return b.resolveTargetQuery(.{
        .cpu_model = whole.cpu_model,
        .cpu_features_add = whole.cpu_features_add,
        .cpu_features_sub = whole.cpu_features_sub,
    });
}

/// Prints the stamp's source: the commit of the tree the build ran in and whether anything in it differed
/// from that commit (untracked files count; ignored ones do not). Outside a git tree the commit is `unknown`.
const stamp_script =
    \\c=$(git rev-parse HEAD 2>/dev/null) || c=unknown
    \\v=$1; o=$2
    \\if [ "$c" = unknown ] || [ -n "$(git --no-optional-locks status --porcelain 2>/dev/null)" ]; then d=true; else d=false; fi
    \\printf 'pub const commit = "%s";\npub const dirty = %s;\npub const version = "%s";\npub const optimize = "%s";\n' "$c" "$d" "$v" "$o"
;

/// The tests' fixtures, under `zig/tests/fixtures/`: a test embeds one by the name `fixtures/PATH`, which each module
/// of `src/` is given here (a file outside a module's directory cannot be embedded by its path).
const fixtures = [_][]const u8{
    "audio/cases.json",
    "csf_blocks_cases.bin",
    "image/cases.json",
    "image/resize_cases.json",
    "ngram_cases.bin",
    "requantize_cases.bin",
    "sampling_cases.bin",
    "text/grammar_masks.txt",
    "text/grammar_masks_checkpoint.txt",
    "text/message_cases.json",
    "text/template_cases.json",
    "text/template_unit_cases.json",
    "text/tiny_cases.json",
    "text/tiny_tokenizer.json",
    "text/tiny_tokenizer_pairs.json",
    "text/tokenizer_cases.json",
    "unlimited_ocr/config_dash.json",
    "vision/bands.png",
    "vision/cases.json",
    "vision/processor_cases.json",
    "vision/shapes.gif",
    "vision/shapes.jpeg",
    "vision/shapes.png",
    "vision/shapes.webp",
    "vision/tiny.png",
};

fn withFixtures(b: *std.Build, module: *std.Build.Module) void {
    for (fixtures) |each| module.addAnonymousImport(b.fmt("fixtures/{s}", .{each}), .{ .root_source_file = b.path(b.fmt("zig/tests/fixtures/{s}", .{each})) });
}
/// The test areas, from `tools/test_areas.tsv` (spec 171): the one table `tools/check.sh`'s inventory
/// reads too, whose header names the columns (`kind` whole is gpu with `tier.whole` true). A malformed
/// row fails the configuration here; the schema's whole check is the shell's `areas_load`.
const test_areas_src = @embedFile("tools/test_areas.tsv");

const Area = struct {
    step: []const u8,
    kind: enum { host, gpu, whole },
    tier: bool,
    split: bool,
    seconds: u16,
    model: []const u8,
    root: []const u8,
    filters: []const u8,
    about: []const u8,
};

fn parseAreas(comptime src: []const u8) []const Area {
    @setEvalBranchQuota(100_000);
    comptime {
        var list: []const Area = &.{};
        var lines = std.mem.splitScalar(u8, src, '\n');
        while (lines.next()) |line| {
            if (line.len == 0 or line[0] == '#') continue;
            var fields: [9][]const u8 = undefined;
            var it = std.mem.splitScalar(u8, line, '\t');
            var i: usize = 0;
            while (it.next()) |f| {
                if (i == fields.len) @compileError("test_areas.tsv: a row with more than nine columns");
                fields[i] = f;
                i += 1;
            }
            if (i != fields.len) @compileError("test_areas.tsv: a row with fewer than nine columns");
            const kind: @FieldType(Area, "kind") = if (std.mem.eql(u8, fields[1], "host")) .host else if (std.mem.eql(u8, fields[1], "gpu")) .gpu else if (std.mem.eql(u8, fields[1], "whole")) .whole else @compileError("test_areas.tsv: kind is host, gpu or whole");
            if (!std.mem.eql(u8, fields[2], "yes") and !std.mem.eql(u8, fields[2], "no")) @compileError("test_areas.tsv: tier is yes or no");
            const split_yes = std.mem.eql(u8, fields[3], "yes");
            if (!split_yes and !std.mem.eql(u8, fields[3], "no")) @compileError("test_areas.tsv: split is yes or no");
            const seconds = std.fmt.parseInt(u16, fields[4], 10) catch
                if (!split_yes and std.mem.eql(u8, fields[4], "-")) 0 else @compileError("test_areas.tsv: seconds is a count or -");
            list = list ++ &[_]Area{.{
                .step = fields[0],
                .kind = kind,
                .tier = std.mem.eql(u8, fields[2], "yes"),
                .split = split_yes,
                .seconds = seconds,
                .model = fields[5],
                .root = fields[6],
                .filters = fields[7],
                .about = fields[8],
            }};
        }
        return list;
    }
}

const test_areas = parseAreas(test_areas_src);

/// The filters column (`a|b`, `-` none) as -Dtest-filter's list, comptime.
fn splitFilters(comptime raw: []const u8) []const []const u8 {
    @setEvalBranchQuota(10_000);
    comptime {
        if (std.mem.eql(u8, raw, "-")) return &.{};
        var list: []const []const u8 = &.{};
        var it = std.mem.splitScalar(u8, raw, '|');
        while (it.next()) |f| list = list ++ &[_][]const u8{f};
        return list;
    }
}
/// `zig/src/image/`'s C (spec 019 WS-07): stb_image's JPEG decoder (`stb_image.c`) and libwebp's decoder, whose
/// sources are the `.c` files its `MANIFEST.sha256` lists. One static library, linked where `src/image/` is.
fn imageLibrary(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    module.addIncludePath(b.path("zig/vendor/stb"));
    module.addIncludePath(b.path("zig/vendor/libwebp"));
    module.addCSourceFile(.{ .file = b.path("zig/src/image/stb_image.c"), .flags = &.{"-std=c99"} });
    const listed = b.root.join(b.allocator, "zig/vendor/libwebp/MANIFEST.sha256") catch @panic("OOM");
    const manifest = listed.root_dir.handle.readFileAlloc(b.graph.io, listed.sub_path, b.allocator, .limited(1 << 20)) catch @panic("zig/vendor/libwebp/MANIFEST.sha256 is missing");
    var sources: std.ArrayList([]const u8) = .empty;
    var lines = std.mem.tokenizeScalar(u8, manifest, '\n');
    while (lines.next()) |line| {
        const path = std.mem.trimStart(u8, line[std.mem.indexOfScalar(u8, line, ' ') orelse continue ..], " ");
        if (std.mem.endsWith(u8, path, ".c")) sources.append(b.allocator, path) catch @panic("OOM");
    }
    module.addCSourceFiles(.{ .root = b.path("zig/vendor/libwebp"), .files = sources.items, .flags = &.{"-std=c99"} });
    return b.addLibrary(.{ .name = "zsimage", .linkage = .static, .root_module = module });
}

/// The original host audio decoders, from dr_libs and stb_vorbis.
fn audioLibrary(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const module = b.createModule(.{ .target = target, .optimize = optimize, .link_libc = true });
    module.addIncludePath(b.path("zig/vendor/dr_libs"));
    module.addIncludePath(b.path("zig/vendor/stb"));
    module.addCSourceFiles(.{ .files = &.{ "zig/src/audio/dr_libs.c", "zig/src/audio/vorbis.c" }, .flags = &.{"-std=c99"} });
    return b.addLibrary(.{ .name = "zsaudio", .linkage = .static, .root_module = module });
}

/// Cargo builds the additional audio codecs with the versions pinned in Cargo.lock.
fn audioCodecs(b: *std.Build) std.Build.LazyPath {
    const cargo = b.addSystemCommand(&.{ "cargo", "build", "--quiet", "--locked", "--release", "--manifest-path" });
    cargo.addFileArg(b.path("zig/src/audio/Cargo.toml"));
    cargo.addFileInput(b.path("zig/src/audio/Cargo.lock"));
    cargo.addFileInput(b.path("zig/src/audio/codecs.rs"));
    cargo.addArg("--target-dir");
    const output = cargo.addOutputDirectoryArg2("audio-codecs", .{});
    return output.path(b, "release/libzs_audio_codecs.a");
}

/// TensorFold's host as modules (specs 059, 072): what its server is built from. The one definition of
/// them is `tools/host_descriptor.zig` (spec 168): each module's name, root below `vendored`, imports and
/// stub, the five shared imports every `.shared` module is handed, and the files the importer takes — the
/// same lists `tools/vendor_import.sh` reads. What differs from upstream has its row in that directory's
/// `PATCHES.md`, which `check-specs` holds.
const Host = struct {
    const descriptor = @import("tools/host_descriptor.zig");
    /// TensorFold 1.3.4's files (`UPSTREAM` there names the commit).
    const vendored = descriptor.vendored;

    lanes: *std.Build.Module,
    api: *std.Build.Module,
    tokenizer: *std.Build.Module,
    template: *std.Build.Module,
    json: *std.Build.Module,
    server: *std.Build.Module,
    /// `zig/src/core/picture/`: PNG to RGB, the orientation tags, alpha over white (spec 019 WS-07)
    picture: *std.Build.Module,

    b: *std.Build,
    target: std.Build.ResolvedTarget,
    made: std.StringArrayHashMapUnmanaged(*std.Build.Module),

    fn init(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) Host {
        var host: Host = .{
            .lanes = undefined,
            .api = undefined,
            .tokenizer = undefined,
            .template = undefined,
            .json = undefined,
            .server = undefined,
            .picture = undefined,
            .b = b,
            .target = target,
            .made = .empty,
        };
        // the serving modules and what they import, in descriptor order; resolution is memoized, so a
        // shared or helper module is made once at the caller's optimize
        inline for (descriptor.modules) |entry| {
            if (entry.role == .host or entry.role == .helper) _ = host.named(entry.name, optimize);
        }
        host.lanes = host.made.get("lanes").?;
        host.api = host.made.get("engine_api").?;
        host.tokenizer = host.made.get("tokenizer").?;
        host.template = host.made.get("template").?;
        host.json = host.made.get("json").?;
        host.server = host.made.get("server").?;
        host.picture = host.made.get("picture").?;
        return host;
    }

    /// The module the descriptor names, made once at `optimize` and memoized: its own `imports` first, the
    /// five shared names beside them where the entry is `.shared`. A name the descriptor does not define
    /// fails here; a name it defines wrongly fails at `tools/host_descriptor.zig`'s comptime check first.
    fn named(host: *Host, name: []const u8, optimize: std.builtin.OptimizeMode) *std.Build.Module {
        if (host.made.get(name)) |existing| return existing;
        const entry = descriptor.find(name) orelse
            std.debug.panic("tools/host_descriptor.zig defines no module named \"{s}\"", .{name});
        var import_list: std.ArrayList(std.Build.Module.Import) = .empty;
        if (entry.shared) {
            // runtime lists: descriptor.find's slice is chosen at build time, not comptime
            for (descriptor.shared) |shared_name| {
                import_list.append(host.b.allocator, .{ .name = shared_name, .module = host.named(shared_name, optimize) }) catch |err|
                    std.debug.panic("host module {s}: {t}", .{ name, err });
            }
        }
        for (entry.imports) |import_name| {
            import_list.append(host.b.allocator, .{ .name = import_name, .module = host.named(import_name, optimize) }) catch |err|
                std.debug.panic("host module {s}: {t}", .{ name, err });
        }
        const root: std.Build.LazyPath = if (entry.stub) |text|
            host.b.addWriteFiles().add(host.b.fmt("{s}.zig", .{name}), text)
        else
            host.b.path(host.b.fmt("{s}{s}", .{ vendored, entry.root.? }));
        const made = host.b.createModule(.{
            .root_source_file = root,
            .target = host.target,
            .optimize = optimize,
            .link_libc = entry.libc,
            .imports = import_list.items,
        });
        host.made.put(host.b.allocator, name, made) catch |err|
            std.debug.panic("host module {s}: {t}", .{ name, err });
        return made;
    }

    /// `test-golden`: TensorFold's recorded replies against its server over its scripted engine, as its
    /// own build runs them on a Mac (its Linux build has no such step). About 80 s, no GPU. It is run after
    /// an import of upstream and after each local patch (`zig/vendor/tensorfold/AGENTS.md`).
    fn golden(host: *Host, b: *std.Build) void {
        // test-golden's two programs are descriptor modules too, compiled Debug (upstream's
        // `zig/build/golden.zig`), their imports — server, engine_api, tokenizer, template — the host's own
        const fake_serve = b.addExecutable(.{ .name = "fake_serve", .root_module = host.named("fake_serve", .Debug) });
        const golden_check = b.addExecutable(.{ .name = "golden_check", .root_module = host.named("golden_check", .Debug) });
        const step = b.step("test-golden", "TensorFold's recorded replies against its server over its scripted engine (no GPU)");
        // since 1.1.1 a second plan, `security-cases.json`: API keys, Host and Origin (upstream's `zig/build/golden.zig`)
        for ([_][]const u8{ "cases.json", "security-cases.json" }) |plan| {
            const run = b.addRunArtifact(golden_check);
            run.addArtifactArg(fake_serve);
            run.addFileArg(b.path(b.fmt("{s}zig/tests/server/golden/{s}", .{ vendored, plan })));
            run.addDirectoryArg(b.path(vendored ++ "zig/tests/server/golden"));
            run.addDirectoryArg(b.path(vendored ++ "zig/tests/server/fixtures"));
            run.has_side_effects = true;
            step.dependOn(&run.step);
        }
    }

    /// The xgrammar release verified with the vendored `libtfgrammar` C++ bridge (spec 174).
    const xgrammar_tag = "v0.2.9";

    /// `grammar`: structured output's library since 1.0.12, `libtfgrammar` (the vendored `tf_grammar.cc` and
    /// `tf_budget.cc` over an xgrammar checkout, `-Dxgrammar=DIR`), into `zig-out/lib`, where the host's
    /// `lanes/grammar.zig` looks beside `zig-out/bin` before the loader's path. As upstream's
    /// `zig/build/grammar.zig` builds it.
    fn grammar(b: *std.Build, target: std.Build.ResolvedTarget) void {
        const step = b.step("grammar", "libtfgrammar (structured output through xgrammar) from an xgrammar " ++ xgrammar_tag ++ " checkout: -Dxgrammar=DIR");
        const clone = "git clone --branch " ++ xgrammar_tag ++ " --recurse-submodules https://github.com/mlc-ai/xgrammar";
        const dir = b.option([]const u8, "xgrammar", "An xgrammar " ++ xgrammar_tag ++ " checkout with its submodules (zig build grammar)") orelse {
            step.dependOn(&b.addFail("zig build grammar needs -Dxgrammar=DIR: " ++ clone).step);
            return;
        };
        var code: u8 = 0;
        const tag = b.runAllowFail(&.{ "git", "-C", dir, "describe", "--tags", "--exact-match" }, &code, .ignore) catch "";
        if (!std.mem.eql(u8, std.mem.trim(u8, tag, "\r\n"), xgrammar_tag)) {
            step.dependOn(&b.addFail(b.fmt("-Dxgrammar={s} is not an xgrammar " ++ xgrammar_tag ++ " checkout: " ++ clone, .{dir})).step);
            return;
        }
        const root: std.Build.LazyPath = .{ .cwd_relative = dir };
        const module = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libcpp = true, .strip = true });
        const map = b.fmt("-ffile-prefix-map={s}=xgrammar", .{dir});
        const flags = [_][]const u8{ "-std=c++17", "-DXGRAMMAR_ENABLE_CPPTRACE=0", "-fexceptions", "-Wno-everything", map };
        module.addIncludePath(root.path(b, "include"));
        module.addIncludePath(root.path(b, "3rdparty/picojson"));
        module.addIncludePath(root.path(b, "3rdparty/dlpack/include"));
        module.addIncludePath(root.path(b, "cpp")); // tf_budget.cc reads xgrammar's internals (earley_parser.h)
        // xgrammar's library as its CMakeLists globs it: every .cc under cpp/ but the Python bindings
        var sources: std.ArrayList([]const u8) = .empty;
        const io = b.graph.io;
        var cpp = std.Io.Dir.cwd().openDir(io, b.pathJoin(&.{ dir, "cpp" }), .{ .iterate = true }) catch @panic("an xgrammar checkout has cpp/");
        defer cpp.close(io);
        var walk = cpp.walk(b.allocator) catch @panic("OOM");
        defer walk.deinit();
        while (walk.next(io) catch null) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".cc") or std.mem.startsWith(u8, entry.path, "tvm_ffi")) continue;
            sources.append(b.allocator, b.pathJoin(&.{ "cpp", entry.path })) catch @panic("OOM");
        }
        std.mem.sort([]const u8, sources.items, {}, struct {
            fn less(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.less);
        module.addCSourceFiles(.{ .root = root, .files = sources.items, .flags = &flags });
        module.addCSourceFile(.{ .file = b.path(vendored ++ "zig/src/grammar/tf_grammar.cc"), .flags = &flags });
        module.addCSourceFile(.{ .file = b.path(vendored ++ "zig/src/grammar/tf_budget.cc"), .flags = &flags });
        const library = b.addLibrary(.{ .name = "tfgrammar", .linkage = .dynamic, .root_module = module });
        step.dependOn(&b.addInstallArtifact(library, .{ .dest_dir = .{ .override = .{ .custom = "lib" } } }).step);
    }
};

pub fn build(b: *std.Build) void {
    const target = steady(b, b.standardTargetOptions(.{}));
    // `fast` unless asked: what is served and timed (a GLM load 37 s fast against 47 safe at one commit, 2026-10-20).
    // `tools/check.sh` builds its tests `safe`, the safety checks on; a Debug build packs a checkpoint in minutes.
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast (default) or small") orelse .fast;

    const cuda_home = b.option([]const u8, "cuda", "CUDA toolkit root (default /usr/local/cuda)") orelse "/usr/local/cuda";
    // DGX Spark's GB10 is compute capability 12.1; the `a` suffix enables its arch-specific instructions.
    const cuda_arch = b.option([]const u8, "cuda-arch", "nvcc target (default 121a)") orelse "121a";
    // Two ranks (spec 045): NCCL is linked only when named, so a build for one device needs none of it.
    const nccl_prefix = b.option([]const u8, "nccl", "NCCL's prefix, /usr for the apt packages: links it for two ranks (default: not linked)");

    // The stamp `zspark version` prints (spec 053): a command that runs at every build and writes the commit,
    // the dirty flag, the calendar version of build.zig.zon (spec 170) and the build mode's name, so the
    // executable is compiled again only when one of them changed. It is a step and not a value read here: a
    // configuration is cached, and would hold an earlier command's commit. The test binaries take fixed
    // constants instead, so that a commit alone compiles no test again.
    const optimize_name = switch (optimize) {
        .Debug => "debug",
        .ReleaseSafe => "safe",
        .ReleaseFast => "fast",
        .ReleaseSmall => "small",
    };
    const stamp_cmd = b.addSystemCommand(&.{ "sh", "-c", stamp_script });
    stamp_cmd.addArg("stamp"); // $0; the script's $1 and $2 are the version and the mode
    stamp_cmd.addArg(zon.version);
    stamp_cmd.addArg(optimize_name);
    stamp_cmd.setCwd(b.path("."));
    stamp_cmd.has_side_effects = true;
    const stamped = stamp_cmd.captureStdOut(.{ .basename = "stamp.zig" });
    const unstamped = b.addWriteFiles().add("stamp.zig", b.fmt("pub const commit = \"a test build\";\npub const dirty = true;\npub const version = \"{s}\";\npub const optimize = \"{s}\";\n", .{ zon.version, optimize_name }));

    // The program's module: as built, and as tested in full; the whole-model tests alone (`tier.whole`, read by the
    // tests that load a whole checkpoint); then the areas from roots of their own, whose whole-model tests skip. The tier is compiled in and not an environment variable set on the run step: a
    // step's environment is fixed when the build is configured, and a configuration is cached, so the tests
    // would see the ZSPARK_* variables of an earlier command. The linked areas' roots are the inventory's;
    // the program, the full tier and the NCCL analysis keep their roots here.
    const Linked = struct { root: []const u8, whole: bool, stamp: std.Build.LazyPath, nccl: bool = false, area: ?[]const u8 = null };
    const gpu_areas = comptime blk: {
        var n = 0;
        for (test_areas) |area| {
            if (area.kind != .host) n += 1;
        }
        break :blk n;
    };
    var linked: [2 + gpu_areas + 1]Linked = undefined;
    linked[0] = .{ .root = "zig/src/main.zig", .whole = true, .stamp = stamped };
    linked[1] = .{ .root = "zig/tests.zig", .whole = true, .stamp = unstamped };
    {
        var at: usize = 2;
        for (test_areas) |area| {
            if (area.kind == .host) continue;
            linked[at] = .{ .root = area.root, .whole = area.kind == .whole, .stamp = unstamped, .area = area.step };
            at += 1;
        }
        // The program as `-Dnccl` builds it, analysed only (no NCCL linked, no binary): `compile` sees the two-rank
        // code a one-device build leaves out (spec 134's split broke it unseen, 2026-10-25).
        linked[at] = .{ .root = "zig/src/main.zig", .whole = true, .stamp = unstamped, .nccl = true };
    }
    // TensorFold's host, which `zspark serve` stands behind (spec 059).
    var tensorfold_host: Host = .init(b, target, optimize);
    const image_c = imageLibrary(b, target, optimize);
    const audio_c = audioLibrary(b, target, optimize);
    const audio_codecs = audioCodecs(b);
    var modules: [linked.len]*std.Build.Module = undefined;
    for (&modules, linked) |*module, each| {
        const tier = b.addOptions();
        tier.addOption(bool, "whole", each.whole);
        module.* = b.createModule(.{
            .root_source_file = b.path(each.root),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        module.*.addOptions("tier", tier);
        const linked_options = b.addOptions();
        linked_options.addOption(bool, "nccl", nccl_prefix != null or each.nccl);
        module.*.addOptions("linked", linked_options);
        if (nccl_prefix) |prefix| {
            for ([_][]const u8{ "lib/aarch64-linux-gnu", "lib" }) |sub_path| {
                module.*.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, sub_path }) });
                module.*.addRPath(.{ .cwd_relative = b.pathJoin(&.{ prefix, sub_path }) });
            }
            module.*.linkSystemLibrary("nccl", .{});
        }
        module.*.addImport("engine_api", tensorfold_host.api);
        module.*.addImport("lanes", tensorfold_host.lanes);
        module.*.addImport("server", tensorfold_host.server);
        module.*.addImport("template", tensorfold_host.template);
        module.*.addImport("tokenizer", tensorfold_host.tokenizer);
        module.*.addImport("picture", tensorfold_host.picture);
        module.*.linkLibrary(image_c);
        module.*.linkLibrary(audio_c);
        module.*.addObjectFile(audio_codecs);
        module.*.linkSystemLibrary("unwind", .{});
        module.*.addAnonymousImport("stamp", .{ .root_source_file = each.stamp });
        withFixtures(b, module.*);
        module.*.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda_home, "lib64" }) });
        module.*.addRPath(.{ .cwd_relative = b.pathJoin(&.{ cuda_home, "lib64" }) });
        module.*.linkSystemLibrary("cudart", .{});
        module.*.linkSystemLibrary("stdc++", .{});
    }
    const exe = b.addExecutable(.{ .name = "zspark", .root_module = modules[0] });

    const nvcc = b.pathJoin(&.{ cuda_home, "bin", "nvcc" });
    for (kernels ++ test_kernels, 0..) |kernel, at| {
        const stem = std.fs.path.stem(kernel.source);
        const cmd = b.addSystemCommand(&.{ nvcc, "-c", "-O3", "-std=c++17", "-Xcompiler", "-fPIC" });
        cmd.addArg(b.fmt("-gencode=arch=compute_{s},code=sm_{s}", .{ cuda_arch, cuda_arch }));
        cmd.addArgs(kernel.flags);
        if (kernel.cutlass) {
            cmd.addArg("-I");
            cmd.addDirectoryArg(b.dependency("cutlass", .{}).path("include"));
        }
        cmd.addArg("-MD");
        cmd.addArg("-MF");
        _ = cmd.addDepFileOutputArg(b.fmt("{s}.d", .{stem}));
        cmd.addArg("-o");
        const object = cmd.addOutputFileArg(b.fmt("{s}.o", .{stem}));
        cmd.addFileArg(b.path(kernel.source));
        for (if (at < kernels.len) modules[0..] else modules[1..]) |module| module.addObjectFile(object);
    }

    // `zspark-router` (spec 088): one address over whole copies of `zspark serve`. No checkpoint, no CUDA,
    // nothing of the vendored host: its own root beside the router's module, stamped like `zspark` is.
    const router_module = b.createModule(.{
        .root_source_file = b.path("zig/src/router/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    router_module.addAnonymousImport("stamp", .{ .root_source_file = stamped });
    const router_exe = b.addExecutable(.{ .name = "zspark-router", .root_module = router_module });

    b.installArtifact(exe);
    b.installArtifact(router_exe);

    const run_step = b.step("run", "Run zspark");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    // -Dtest-filter=TEXT runs the tests whose names hold TEXT: a test on a whole checkpoint without the others.
    // Given again, a test runs whose name holds any of them (`check.sh qwen4-files`, spec 103).
    const test_filter = b.option([]const []const u8, "test-filter", "Run only the tests whose names contain this (repeated: any of them)");
    const filters: []const []const u8 = test_filter orelse &.{};
    // Two tiers (spec 022), each one binary. `test` is the full one: with the ZSPARK_* variables naming
    // checkpoints and references it loads each whole model. `test-host` links no kernel and needs neither the
    // GPU nor the CUDA toolkit. Then a step an area (spec 053), for the edit that touches one: a binary of that
    // area's tests alone, built with `tier.whole` false, so a test that loads a whole checkpoint skips itself.
    // `test-text` and `test-server` are the host's; `test-weights` (the device code every model shares) and the
    // models' need the GPU. `test-whole` is those whole-model tests alone (spec 133): the host tier, the GPU's
    // areas and it hold the full tier's tests between them (the quick tier, the full one's tests with
    // `tier.whole` false, is gone: a stage paid for it and the full tier both).
    // An area under a directory takes the directory as its filter (a test's full name starts with its file's
    // path, `server.http.test.NAME`), so the tests of other areas its root reaches are left to their own steps
    // and what only they call is not compiled; -Dtest-filter replaces it.
    const Step = struct { name: []const u8, about: []const u8, module: *std.Build.Module, area: []const []const u8 = &.{}, tier: bool = false };
    // The tests gated on `tier.whole`: each loads a whole checkpoint. The whole step's filters in
    // tools/test_areas.tsv hold every one (`tools/check.sh sets` fails a gated test missing from it).
    const host = struct {
        fn module(owner: *std.Build, root: []const u8, of: Host, c: [2]*std.Build.Step.Compile, codecs: std.Build.LazyPath, t: std.Build.ResolvedTarget, o: std.builtin.OptimizeMode, stamp: std.Build.LazyPath) *std.Build.Module {
            const made = owner.createModule(.{ .root_source_file = owner.path(root), .target = t, .optimize = o });
            for (c) |library| made.linkLibrary(library); // `src/image/`'s C and `src/audio/`'s
            made.addObjectFile(codecs);
            made.linkSystemLibrary("unwind", .{});
            // the host itself, for `tests/server/served_test.zig`: a model behind it, asked over a socket (spec 072)
            made.addImport("engine_api", of.api);
            made.addImport("lanes", of.lanes);
            made.addImport("server", of.server);
            made.addImport("template", of.template);
            made.addImport("tokenizer", of.tokenizer);
            made.addImport("picture", of.picture);
            withFixtures(owner, made);
            // the build metadata `src/cli.zig` and `src/server/command.zig` read (the tests' fixed stamp)
            made.addAnonymousImport("stamp", .{ .root_source_file = stamp });
            return made;
        }
    };
    const startup_driver = b.addExecutable(.{
        .name = "startup_serve",
        .root_module = host.module(b, "zig/startup_serve.zig", tensorfold_host, .{ image_c, audio_c }, audio_codecs, target, optimize, unstamped),
    });
    startup_driver.root_module.link_libc = true;
    const startup_options = b.addOptions();
    startup_options.addOptionPath("exe", startup_driver.getEmittedBin());

    var steps: [1 + test_areas.len]Step = undefined;
    steps[0] = .{ .name = "test", .about = "The full tier: every test, whole models where ZSPARK_* name them (needs the GPU)", .module = modules[1], .tier = true };
    inline for (test_areas, 1..) |area, at| {
        const module = switch (area.kind) {
            .host => host.module(b, area.root, tensorfold_host, .{ image_c, audio_c }, audio_codecs, target, optimize, unstamped),
            else => m: {
                for (&modules, &linked) |mod, each| {
                    if (each.area) |s| {
                        if (std.mem.eql(u8, s, area.step)) break :m mod;
                    }
                }
                @panic("test_areas.tsv: an area has no linked module");
            },
        };
        steps[at] = .{
            .name = b.fmt("test-{s}", .{area.step}),
            .about = area.about,
            .module = module,
            .area = comptime splitFilters(area.filters),
            .tier = area.tier,
        };
    }
    // `compile` builds the program and the full tier's test binary and runs none (spec 069): every file of
    // `src/` is compiled, no GPU and no lock on it is needed, so it is the check while a refactor is edited, and
    // several worktrees run it at once.
    const compile = b.step("compile", "Compile zspark and every test without running one (no GPU)");
    const runner: std.Build.Step.Compile.TestRunner = .{ .path = b.path("tools/test_runner.zig"), .mode = .server };
    compile.dependOn(&exe.step);
    compile.dependOn(&router_exe.step);
    compile.dependOn(&startup_driver.step);
    compile.dependOn(&b.addExecutable(.{ .name = "zspark-nccl-check", .root_module = modules[linked.len - 1] }).step);
    // Every test binary is built with `tools/test_runner.zig` (spec 133): Zig's runner, and each test's
    // milliseconds where ZSPARK_TEST_TIMES names a directory. Each step has a `-build` twin that compiles its
    // binary and runs nothing, so `tools/check.sh` compiles under the shared machine lock and takes the GPU's
    // only for the run, which then finds the binary in the cache (the same -D options, the same binary).
    var tiers: [steps.len]*std.Build.Step = undefined;
    for (&tiers, steps) |*tier, each| {
        tier.* = b.step(each.name, each.about);
        const binary = b.addTest(.{
            .root_module = each.module,
            .filters = if (test_filter == null and each.area.len != 0) each.area else filters,
            .test_runner = runner,
        });
        if (std.mem.eql(u8, each.name, "test")) compile.dependOn(&binary.step);
        b.step(b.fmt("{s}-build", .{each.name}), b.fmt("Compile {s}'s binary, run nothing", .{each.name})).dependOn(&binary.step);
        const run = b.addRunArtifact(binary);
        if (each.tier or std.mem.eql(u8, each.name, "test-server"))
            each.module.addOptions("startup_fixture", startup_options);
        // The tests read the GPU, checkpoints and ZSPARK_* variables (the host's, ZSPARK_FLASH_NEXT_FP4_LOCAL for the
        // checkpoint's tokenizer and template), none of which the cache sees: always run.
        run.has_side_effects = true;
        tier.*.dependOn(&run.step);
    }

    tensorfold_host.golden(b);
    Host.grammar(b, target);

    // `bench-held-turn` (spec 072): what the host costs a held turn before the engine has the prompt, over the
    // checkpoint's tokenizer and template named after `--`. No GPU, no weights; `compile` builds it.
    const held_turn = b.addExecutable(.{ .name = "held_turn_bench", .root_module = b.createModule(.{
        .root_source_file = b.path("tools/held_turn_bench.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "server", .module = tensorfold_host.server }, .{ .name = "engine_api", .module = tensorfold_host.api }, .{ .name = "tokenizer", .module = tensorfold_host.tokenizer } },
    }) });
    held_turn.root_module.addAnonymousImport("segments", .{ .root_source_file = b.path("zig/src/server/segments.zig"), .imports = &.{ .{ .name = "server", .module = tensorfold_host.server }, .{ .name = "tokenizer", .module = tensorfold_host.tokenizer } } });
    compile.dependOn(&held_turn.step);
    const held_turn_run = b.addRunArtifact(held_turn);
    held_turn_run.addPassthruArgs();
    held_turn_run.has_side_effects = true;
    b.step("bench-held-turn", "The host's milliseconds before a held turn reaches the engine: -- CKPT_DIR (no GPU)").dependOn(&held_turn_run.step);

    // The tracked specs' validator: no GPU and no CUDA toolkit, so CI can run it.
    const specs_module = b.createModule(.{
        .root_source_file = b.path("tools/check_specs.zig"),
        .target = steady(b, b.graph.host),
        .optimize = .Debug,
    });
    const check_specs = b.addRunArtifact(b.addExecutable(.{ .name = "check-specs", .root_module = specs_module }));
    check_specs.setCwd(b.path("."));
    const check_specs_step = b.step("check-specs", "Validate specs/ (lifecycle, IDs, index) and the size budgets of guidance and sources");
    check_specs_step.dependOn(&check_specs.step);
    const specs_tests = b.addRunArtifact(b.addTest(.{ .root_module = specs_module, .test_runner = runner }));
    specs_tests.setCwd(b.path("."));
    const specs_tests_step = b.step("test-specs", "Run the spec validator's tests");
    specs_tests_step.dependOn(&specs_tests.step);

    // `zspark-cache` (spec 168): build and download cache inspection and bounded pruning, a host-only
    // installed CLI; `compile` builds it and its test binary.
    const cache_module = b.createModule(.{
        .root_source_file = b.path("tools/cache.zig"),
        .target = steady(b, b.graph.host),
        .optimize = optimize,
    });
    const zspark_cache = b.addExecutable(.{ .name = "zspark-cache", .root_module = cache_module });
    b.installArtifact(zspark_cache);
    compile.dependOn(&zspark_cache.step);
    const cache_run = b.addRunArtifact(zspark_cache);
    cache_run.addPassthruArgs();
    cache_run.has_side_effects = true;
    b.step("cache", "Cache inspection and bounded pruning (no GPU)").dependOn(&cache_run.step);
    const cache_test_module = b.createModule(.{
        .root_source_file = b.path("tools/cache_test.zig"),
        .target = steady(b, b.graph.host),
        .optimize = optimize,
    });
    const cache_test = b.addTest(.{ .root_module = cache_test_module, .test_runner = runner });
    compile.dependOn(&cache_test.step);
    const cache_tests = b.addRunArtifact(cache_test);
    // The custom runner reads list/timing environment afresh, like the tier runners above.
    cache_tests.has_side_effects = true;
    b.step("test-cache", "Run zspark-cache's tests").dependOn(&cache_tests.step);

    const descriptor_binary = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/host_descriptor.zig"),
            .target = steady(b, b.graph.host),
            .optimize = .Debug,
        }),
        .test_runner = runner,
    });
    compile.dependOn(&descriptor_binary.step);
    const descriptor_tests = b.addRunArtifact(descriptor_binary);
    descriptor_tests.has_side_effects = true;
    b.step("test-host-descriptor", "Reject malformed host module graphs").dependOn(&descriptor_tests.step);
    const vendor_tests = b.addSystemCommand(&.{ "bash", "tools/vendor_import_test.sh" });
    vendor_tests.setCwd(b.path("."));
    vendor_tests.has_side_effects = true;
    b.step("test-vendor", "Exercise vendor publication and replay in temporary repositories").dependOn(&vendor_tests.step);
    // The test-area inventory's contract (spec 171): the load's refusals, the scope's shared and scoped
    // paths, curated coverage and failure propagation, on temporary repositories — no GPU.
    const areas_tests = b.addSystemCommand(&.{ "bash", "tools/check_areas_test.sh" });
    areas_tests.setCwd(b.path("."));
    areas_tests.has_side_effects = true;
    b.step("test-areas", "Exercise the test-area inventory and curated coverage in temporary repositories").dependOn(&areas_tests.step);

    // `fixture-nemotron` (spec 151): the nemotron forward fixture's small synthetic checkpoint written as a
    // real directory `zspark generate DIR` opens: the three config files, every tensor in one shard, and the
    // tokenizer files copied from the published checkpoint's public files. No GPU; `compile` builds it.
    const nemotron_fixture = b.addExecutable(.{ .name = "nemotron-fixture", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/nemotron_fixture.zig"),
        .target = steady(b, b.graph.host),
        .optimize = optimize,
    }) });
    compile.dependOn(&nemotron_fixture.step);
    const nemotron_fixture_run = b.addRunArtifact(nemotron_fixture);
    nemotron_fixture_run.addPassthruArgs();
    nemotron_fixture_run.has_side_effects = true;
    b.step("fixture-nemotron", "Write the small synthetic Nemotron checkpoint: -- DIR [HF_DIR] (no GPU)").dependOn(&nemotron_fixture_run.step);

    // The tiers run the spec and tool steps too; an area's step does not.
    for (tiers, steps) |tier, each| {
        if (!each.tier) continue;
        tier.dependOn(&check_specs.step);
        tier.dependOn(&specs_tests.step);
        tier.dependOn(&cache_tests.step);
        tier.dependOn(&descriptor_tests.step);
        tier.dependOn(&areas_tests.step);
    }
}

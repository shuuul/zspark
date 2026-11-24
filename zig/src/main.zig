const std = @import("std");
const cuda = @import("cuda/cuda.zig");
const units = @import("cuda/units.zig");
const cli = @import("cli.zig");
const dispatch = @import("decode/dispatch.zig");
// two ranks of the tests' random GLM: no checkpoint names its model, so the command is found here
const ranks_check = @import("models/glm5/checks/ranks_check.zig");
const qwen4_ranks_check = @import("models/qwen4/checks/ranks_check.zig");

/// The models, each with the commands it has (`dispatch.zig`): a command that takes a checkpoint directory is
/// its model's own, found by the directory's `model_type`. The one place that names a model.
const models = [_]dispatch.Model{
    @import("models/qwen4/commands.zig").model,
    @import("models/glm5/commands.zig").model,
    @import("models/glm5/commands.zig").drafter,
    @import("models/unlimited_ocr/commands.zig").model,
    @import("models/qwen3_asr/commands.zig").model,
    @import("models/nemotron/commands.zig").model,
};

/// The DGX Spark's GB10; kernels are built for this compute capability alone.
const required_capability = .{ .major = 12, .minor = 1 };

const usage =
    \\usage: zspark <command>
    \\
    \\  probe             report the GPU and launch a test kernel
    \\  info DIR [all|rank=R/2]  a Flash Next checkpoint's config and where its weights are; all: every missing tensor; rank: one rank's slices
    \\  chat DIR [OPTION..] [MESSAGE]  answer one request as text; `zspark chat` alone lists the options
    \\  serve DIR [TensorFold's serve flags] [--kv-dtype bf16|fp8|fp8-nvfp4] [--serial] [--no-copies] [--lazy-graphs]
    \\            [--depth N] [--reply-stride N] [--max-num-batched-tokens N]
    \\                    the checkpoint kept loaded behind TensorFold's HTTP server (chat completions,
    \\                    Responses, Messages), whose flags these are: --host, --port (127.0.0.1:8888),
    \\                    --name, --context (262144), --parallel (4), --max-tokens (the context's room), --no-drafts and the
    \\                    rest of `tensorfold serve`. --parallel 1 is one reply on the contiguous caches.
    \\                    --reply-stride: a state is kept each N tokens of a reply (256; 0: none), for a later
    \\                    prompt that parts inside it. --kv-dtype: bf16 unless given, but FP8 on an NVFP4
    \\                    checkpoint above --parallel 1 on one Spark, where one stream and several then hold
    \\                    the same caches (two ranks stay bf16); the drafting head's own cache is bf16 whatever --kv-dtype says.
    \\                    --max-num-batched-tokens: the rows a prompt's window runs and the model's buffers
    \\                    are made for, 2048 to 4096 (default 4096; the pool's rounds 2048, on two ranks
    \\                    the flag's where given); 4096 takes a long prompt's experts in half the
    \\                    readings for twice the windows' memory
    \\  tokens DIR TEXT   the ids of TEXT and each one's text, without the chat template
    \\  compare DIR REF [free] [rows[=POINT]] [perturb=SIZE[@LAYER[:ROW]]] [seeds=N]
    \\                    the forward against TensorFold's reference tensors, point by point; free: on its own
    \\                    inputs; rows: POINT (h_in) by row and layer too; perturb: free runs with noise of SIZE
    \\                    of a row's rms added to LAYER's h_in (every row, or ROW), against the run without it
    \\  generate DIR N [rows=R] [rounds|chains] [copies] [context=C] [depth=D] [confidence=P] [follow=FILE] [sample=SEED,T,K,P] [cases=FILE] [score=FILE] [KV] [serial|eager|lazy] ID..  run the prompt ids through the whole model, then
    \\                       decode N tokens greedily, with the MTP head's drafts (rows=R: buffers for windows of R
    \\                       rows, 16 unless given, as a session's for prompts are 512; serial: one token a step; eager:
    \\                       that with one launch a kernel, not the step's captured graph; lazy: a graph
    \\                       captured when a step first needs it, not at load; KV: fp8 or fp8-nvfp4 caches;
    \\                       context=C: caches for C tokens, a multiple of 512, 4096 unless given; copies: drafts copied
    \\                       from the context where the head begins the same;
    \\                       cases=FILE: a reply for each line `N [OPTION..] [name=LABEL] [plain] ID..` of FILE, on one load,
    \\                       and `loaded in` after the first reply's graphs; plain: one token a step, the head not asked;
    \\                       score=FILE: those ids scored on this load, as `zspark score` scores them;
    \\                       score-each=FILE: with every token's own too)
    \\  score DIR [KV] [each] [perturb=SIZE[@LAYER[:ROW]]] [seeds=N] ID..  run token ids through the whole model
    \\                    and print how well each token was predicted from those before it: the mean negative
    \\                    log-likelihood (nats) and the share ranked first; each: every token's own too, to pair
    \\                    two runs; perturb: the same again under a tap that adds nothing, then N runs (1) with
    \\                    noise of SIZE of a row's rms added to LAYER's h_in, each against the tapped run: the
    \\                    score's own spread
    \\  ranks-check random [model=flash-next] rank=0|1 at=ADDR:PORT  a rank a process over NCCL against both
    \\                    in one, bit for bit: GLM's random model, or Flash Next's (kv=bf16|fp8|fp8-nvfp4:
    \\                    with an attention layer, its caches so stored)
    \\  ranks-check DIR reference=REF [rank= at=]  GLM against TensorFold's export, its serial steps and replies
    \\  ranks-check DIR [layers=L] rank=0|1 at=ADDR:PORT  Flash Next's first L layers over NCCL against loopback
    \\  stack FLASH_DIR [--ocr DIR] [--asr DIR] [--peer ADDR] [FLAG..] [-- SERVE_FLAG..]
    \\                    Flash Next, Unlimited-OCR-2 and Qwen3-ASR, each its own `serve`, behind the router's one
    \\                    address, on one Spark or two (--peer: Flash Next as two ranks); `zspark stack` alone
    \\                    lists the flags
    \\  version           print the version
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var stdout_buffer: [1024]u8 = undefined;
    // Streaming, so stdout and stderr redirected to one file follow each other instead of overwriting.
    var stdout_file: std.Io.File.Writer = .initStreaming(.stdout(), init.io, &stdout_buffer);
    const stdout = &stdout_file.interface;
    defer stdout.flush() catch {};

    const command = if (args.len > 1) args[1] else "";
    cuda.account.trace = init.minimal.environ.getPosix("ZSPARK_ALLOCATIONS") != null;
    // freed device buffers held back and read again at the end: a write through a stale pointer panics
    cuda.quarantine.on = init.minimal.environ.getPosix("ZSPARK_QUARANTINE") != null;
    defer if (cuda.quarantine.on) {
        cuda.quarantine.drain() catch |err| switch (err) {
            error.Overwritten => std.debug.panic("ZSPARK_QUARANTINE: {f}", .{cuda.quarantine.found.?}),
            else => std.debug.panic("ZSPARK_QUARANTINE: {t}", .{err}),
        };
        std.debug.print("zspark: ZSPARK_QUARANTINE: {d} freed buffers read back, none written after its free\n", .{cuda.quarantine.checked});
    };
    // where a load's time goes, a line to stderr (`cuda.phases`)
    cuda.phases.on = init.minimal.environ.getPosix("ZSPARK_LOAD_PHASES") != null;
    cuda.phases.started = cuda.monotonic();
    // Flash Next's DeltaNet state read and written in rows of four floats (spec 073), the default; 0 keeps the
    // float at a time, for a comparison
    if (init.minimal.environ.getPosix("ZSPARK_GDN_QUADS")) |text| {
        @import("models/qwen4/blocks/gdn.zig").setQuads(!std.mem.eql(u8, text, "0"));
    }
    // Flash Next's write-back and the norm after it as one launch (spec 073), the default; 0 keeps the two
    // launches, for a comparison
    if (init.minimal.environ.getPosix("ZSPARK_HC_JOINED")) |text| {
        @import("models/qwen4/blocks/hc.zig").joined = !std.mem.eql(u8, text, "0");
    }
    // Flash Next's read-out at a prompt window's rows in three launches, the normed rows never written (spec
    // 148 WS-03), the default; 0 keeps the five, for a comparison
    if (init.minimal.environ.getPosix("ZSPARK_HC_FUSED")) |text| {
        @import("models/qwen4/blocks/hc.zig").fused = !std.mem.eql(u8, text, "0");
    }
    // a prompt window's attention a row a block from this many rows (spec 105 WS-05), the same bits; 0 never
    if (init.minimal.environ.getPosix("ZSPARK_ATTN_WINDOW")) |text| {
        @import("models/qwen4/blocks/attention.zig").window_rows = std.fmt.parseInt(usize, text, 10) catch 0;
    }
    // GLM's kernels by their other forms (spec 073 WS-05), each the base's bits and the default since the round
    // of both ranks (WS-06: 4 to 7% of decode, a sixth off a prompt's first token); 0 keeps the base. `pre`'s
    // gates and Sinkhorn on a warp's lanes, `mix`'s values loaded eight at a time
    {
        const glm_hc = @import("models/glm5/layers/hc.zig");
        glm_hc.lanes = !envIs(init, "ZSPARK_GLM_HC_LANES", "0");
        glm_hc.octs = !envIs(init, "ZSPARK_GLM_HC_OCTS", "0");
        glm_hc.setForms();
    }
    // the KDA chain and replay moving the state four floats at a time
    @import("models/glm5/layers/kda.zig").setQuads(!envIs(init, "ZSPARK_GLM_KDA_QUADS", "0"));
    // DSA's expand asking for kv_b's rows this many steps ahead of its loads, 2 unset
    @import("models/glm5/layers/dsa.zig").setAsk(if (init.minimal.environ.getPosix("ZSPARK_GLM_MLA_ASK")) |text|
        std.fmt.parseInt(i32, text, 10) catch 0
    else
        2);
    // DSA's chunks over an FP8 cache with FP8 queries and weights, as vLLM's (spec 114 WS-05); 0 bf16 queries
    @import("models/glm5/layers/dsa.zig").setQ8(!envIs(init, "ZSPARK_GLM_MLA_Q8", "0"));
    // a prompt chunk's indexer rows split across the ranks from this position on (spec 152 WS-10), 2,048
    // unset; 0 never
    if (init.minimal.environ.getPosix("ZSPARK_GLM_INDEX_SPLIT")) |text| {
        @import("models/glm5/layers/index.zig").split_from = std.fmt.parseInt(usize, text, 10) catch 0;
    }
    // Flash Next's commit shifting its layers' conv windows in one launch (spec 073 WS-07); 0 keeps a launch a
    // layer
    if (init.minimal.environ.getPosix("ZSPARK_SHIFT_ONCE")) |text| {
        @import("models/qwen4/blocks/state.zig").once = !std.mem.eql(u8, text, "0");
    }
    // the half-precision GEMMs on the tile their rows and weight pick (spec 073), unless 0: the same bits; 2
    // is 1 with the deep tile on a narrow input too, for a comparison
    if (init.minimal.environ.getPosix("ZSPARK_GEMM_TILES")) |text| {
        const bf16 = @import("weights/bf16.zig");
        const asked = std.fmt.parseInt(usize, text, 10) catch 0;
        bf16.tiles_by_shape = asked != 0;
        if (asked == 2) bf16.deep_from = 0;
    }
    // the NVFP4 experts' units in whole rounds over the warps (spec 073), unless 0: the same bits
    if (init.minimal.environ.getPosix("ZSPARK_EVEN_EXPERTS")) |text| {
        @import("weights/experts.zig").even_units = (std.fmt.parseInt(usize, text, 10) catch 0) != 0;
    }
    // block-FP8 linears by the tile-scale kernel up to this many rows, 16 at most and by default (spec 070); 0
    // keeps the lane matmul, for a comparison
    if (init.minimal.environ.getPosix("ZSPARK_FP8_TILES")) |text| {
        @import("weights/nvfp4.zig").tile_rows = @min(std.fmt.parseInt(usize, text, 10) catch 0, 16);
    }
    // Flash Next's drafting on what the checkpoint stores (spec 071): 1 leaves out the two requantizations
    // TensorFold makes at load for drafts, the draft head's rows and the MTP layer's experts; unset or 0, both
    if (init.minimal.environ.getPosix("ZSPARK_DRAFTS_STORED")) |text| {
        @import("models/qwen4/model.zig").drafts_stored = (std.fmt.parseInt(usize, text, 10) catch 0) != 0;
    }
    // Flash Next's prompt windows in the card's vLLM's activations (spec 143 WS-06): unset or `card`, `bf16` off
    @import("models/qwen4/model.zig").prompt_form = .card;
    if (init.minimal.environ.getPosix("ZSPARK_PROMPT_FORM")) |text| {
        @import("models/qwen4/model.zig").prompt_form = std.meta.stringToEnum(@import("models/qwen4/model.zig").PromptForm, text) orelse {
            std.log.err("ZSPARK_PROMPT_FORM={s}: card or bf16", .{text});
            return 2;
        };
    }
    // how two ranks hold Flash Next's MTP head (spec 090): `whole`, on rank 0; unset or `split`, half on each
    if (init.minimal.environ.getPosix("ZSPARK_RANKS_HEAD")) |text| {
        @import("models/qwen4/model.zig").ranks_head = std.meta.stringToEnum(@import("models/qwen4/model.zig").RanksHead, text) orelse {
            std.log.err("ZSPARK_RANKS_HEAD={s}: whole or split", .{text});
            return 2;
        };
    }
    // how Flash Next's DFlash2 drafter is held (spec 135 WS-04): `stored`, BF16 as the file has it; `mixed`, its
    // large linears and its proposing head in four bits made at load, as TensorFold's drafters
    if (init.minimal.environ.getPosix("ZSPARK_DRAFTER_STORAGE")) |text| {
        const storage = std.meta.stringToEnum(enum { stored, mixed }, text) orelse {
            std.log.err("ZSPARK_DRAFTER_STORAGE={s}: stored or mixed", .{text});
            return 2;
        };
        @import("models/qwen4/dflash2/rig.zig").drafter_stored = storage == .stored;
    }
    if (std.mem.eql(u8, command, "version")) {
        try stdout.print("zspark {s} {s}{s}\n", .{ cli.version, cli.stamp.commit, if (cli.stamp.dirty) " dirty" else "" });
        return 0;
    }
    // the launch of several `serve`s and the router (spec 139): host only, nothing of the device
    if (std.mem.eql(u8, command, "stack")) return @import("stack/command.zig").run(init, stdout, args[2..]);
    if (std.mem.eql(u8, command, "probe")) {
        const info = try cuda.probe();
        try stdout.print("{s}: compute capability {d}.{d}, {d:.1} of {d:.1} GiB free, kernel sum {d}\n", .{
            info.nameSlice(),
            info.major,
            info.minor,
            units.gib(info.free_bytes),
            units.gib(info.total_bytes),
            info.kernel_sum,
        });
        if (info.major != required_capability.major or info.minor != required_capability.minor) {
            std.log.err("zspark targets the DGX Spark (compute capability {d}.{d})", .{ required_capability.major, required_capability.minor });
            return 1;
        }
        return 0;
    }
    // a random model has no directory to dispatch by: `model=flash-next` picks Flash Next's, GLM's otherwise
    if (std.mem.eql(u8, command, "ranks-check") and args.len > 2 and std.mem.eql(u8, args[2], "random")) {
        var flash_next = false;
        for (args[3..]) |word| flash_next = flash_next or std.mem.eql(u8, word, "model=flash-next");
        const code = if (flash_next) try qwen4_ranks_check.run(init, stdout, args[2..]) else try ranks_check.run(init, stdout, args[2..]);
        if (code == 2) try stdout.writeAll(if (flash_next) qwen4_ranks_check.usage else ranks_check.usage);
        return code;
    }
    if (std.mem.eql(u8, command, "tokens") and args.len > 3) {
        return tokenize(init, stdout, args[2], args[3]);
    }
    // What this process took of the switches, a line a run: the proof of a comparison's side (spec 073 WS-06: a
    // shell that did not split a variable gave one switch the others' text, hence the quotes). Paths are left out
    {
        var any = false;
        for (init.minimal.environ.block.view().slice) |entry| {
            const text = std.mem.span(entry);
            const cut = std.mem.indexOfScalar(u8, text, '=') orelse continue;
            if (!std.mem.startsWith(u8, text, "ZSPARK_") or std.mem.startsWith(u8, text[cut + 1 ..], "/")) continue;
            std.debug.print("{s}{s}=\"{s}\"", .{ if (any) " " else "zspark: switches: ", text[0..cut], text[cut + 1 ..] });
            any = true;
        }
        std.debug.print("{s}\n", .{if (any) "" else "zspark: switches: none"});
    }
    // a model's command: the directory's `model_type` says whose
    const words = if (args.len > 2) args[2..] else args[0..0];
    const arena = init.arena.allocator();
    const model_type = dispatch.dirless(&models, command, words) orelse if (words.len > 0) dispatch.modelType(arena, init.io, .cwd(), words[0]) else null;
    switch (dispatch.pick(&models, command, model_type)) {
        .none => {},
        .command => |found| {
            if (words.len >= found.least) return found.run(init, stdout, words);
            if (found.usage) |text| {
                try stdout.writeAll(text);
                return if (words.len == 0) 0 else 2;
            }
        },
        .unknown => |found| {
            std.log.err("{s}: {s}", .{ words[0], try dispatch.unknownText(arena, &models, found) });
            return 1;
        },
        .lacking => |model| {
            std.log.err("{s}: {s}", .{ words[0], try dispatch.lackingText(arena, model, model_type.?, command) });
            return 1;
        },
    }
    try stdout.writeAll(usage);
    return if (command.len == 0) 0 else 2;
}

/// A text's ids with the tokenizer alone, one a line beside the text it stands for.
fn tokenize(init: std.process.Init, stdout: *std.Io.Writer, path: []const u8, text: []const u8) !u8 {
    const tokenizer = @import("text/tokenizer.zig");
    var diagnostic: tokenizer.Diagnostic = .{};
    var t = tokenizer.Tokenizer.load(init.gpa, init.io, .cwd(), path, &diagnostic) catch |err| {
        std.log.err("{s}: {s}", .{ path, if (diagnostic.len > 0) diagnostic.message() else @errorName(err) });
        return 1;
    };
    defer t.deinit(init.gpa);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(init.gpa);
    try t.encode(init.gpa, text, &ids);
    for (ids.items) |id| try stdout.print("{d}\t{f}\n", .{ id, std.zig.fmtString(t.tokenBytes(id)) });
    try stdout.print("{d} ids\n", .{ids.items.len});
    return 0;
}

/// Whether the environment holds `name` with exactly `value`.
fn envIs(init: std.process.Init, name: []const u8, value: []const u8) bool {
    return if (init.minimal.environ.getPosix(name)) |text| std.mem.eql(u8, text, value) else false;
}

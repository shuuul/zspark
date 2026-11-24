//! What a session is told when it is made and what a reply gives back: `Options`, the rows a prompt's window
//! runs, `Reply`, and the depths a reply drafts to. `session.zig` names each of them as its own.

const request_file = @import("../../../server/request.zig");
const copied = @import("../../../decode/copied.zig");
const recall = @import("../../../decode/recall.zig");
const draft = @import("../draft.zig");
const mtp = @import("../mtp.zig");
const rig = @import("../dflash2/rig.zig");
const qwen4_config = @import("../config.zig");
const masks_file = @import("../../../decode/masks.zig");
const marks_file = @import("../../../decode/marks.zig");

pub const Options = struct {
    /// The most rows a reply's window runs: a token and its drafts.
    window: usize = draft.forward_max_rows,
    /// The most rows a prompt's window runs, where the model was loaded for as many (`Model.max_rows`).
    prompt_rows: usize = prompt_rows,
    /// Positions the caches hold: a prompt and its reply together.
    capacity: usize = 4096,
    /// One launch a kernel instead of the step's captured graph.
    eager: bool = false,
    /// A reply's drafts are what followed an earlier occurrence of its tail where the head begins the same
    /// (`decode/copied.zig`, spec 060); false: the head's alone.
    copies: bool = true,
    /// `serve --recall`'s record (`decode/recall.zig`, spec 106), the caller's: the copies are asked of it
    /// too, and it takes each prompt and reply. Null: off, as by default.
    recall: ?*recall.Store = null,
    /// Snapshots of the state kept for later prompts to start from; 0: only the state as a reply leaves it.
    /// Past this many one goes, by `thinned`: they stay dense near the sequence's end, where the next prompt
    /// most likely parts from it, and thin out toward its start.
    snapshots: usize = 8,
    /// Sequences the session left that are kept to be taken up again (`parked.zig`, spec 064): a second
    /// conversation in turn with the first, each then running its new tokens. Each holds a state and its
    /// caches' rows above where it parted, 30.8 kB a token on Flash Next in bf16. Those that are the last to
    /// hold their harness's state are not of this count (`parked.Lot.last`, spec 108): `parked_bytes` and
    /// `parked.harnesses` bound them. 0: none.
    parked: usize = 4,
    /// Device bytes they hold at most (`--prompt-cache-gib`; the pool's `default_states_bytes` unless given).
    parked_bytes: usize = 6816 << 20,
    /// A snapshot is taken where a prompt's window ends too, when the window crossed a multiple of this many
    /// positions and no message starts in it, so a prompt that parts early in a long message runs few of the
    /// tokens they share; 0: none.
    stride: usize = 512,
    /// A message's start stands for the starts and marks less than this many tokens after it (the start of
    /// the reply's header after a short message, the mark where a prompt parted), so a short message has one
    /// snapshot and not three. Not for the mark before the prompt's end (`tail`), which is kept however near
    /// a start it lies: a window's cost is not as fixed as this rule took it to be. Traced on NVFP4 (spec
    /// 024), a prompt sent again resumed the seven rows from its header's start in 71.5 ms, 67 of them the
    /// window: the linears' 40 ms are a window's whatever its rows, but the experts' 19.5 ms follow the rows
    /// (1.6 ms a row), so the six rows that mark saves were 10 to 12 ms, where a state costs 1.5 ms to
    /// restore and, written into buffers that exist, 1.5 to 1.9 ms to take (spec 036). The marks this still
    /// leaves out cost their rows at that price: up to 31 rows, which is why it is not wider.
    near: usize = 32,
    /// A snapshot is taken this far before a prompt's end, whatever stands near it (`near`): the same prompt
    /// sent again (a regeneration) resumes these rows, and the next turn's prompt holds this one up to there.
    /// Qwen's template ends a prompt with `<think>\n` and writes a past reply as `<think>\n\n</think>`, so the
    /// two part at the last token, and a state serves a prompt that agrees past its position (`resume`): two
    /// and not one, which would serve the prompt sent again alone. vLLM and SGLang hold a state within a
    /// block of a prompt's end likewise and run its tail again (spec 018's survey).
    tail: usize = 2,
    /// The states inside a prompt's last window get their DeltaNet part once the reply's first token is sent
    /// (`forward.settle`), where the window fits (`forward.defers`). Off, they are replayed inside the
    /// forward, before that token: for the tests that hold the two against each other, and a timing's base.
    settle: bool = true,
    /// While a reply is decoded a state is kept each time this many tokens stand past the newest state held
    /// (spec 036), so a prompt that parts late in a long reply runs this many of the tokens they share at
    /// most, not the reply: SGLang's `mamba_track_interval`. Counted from the newest state and not on the
    /// sequence's grid, as vLLM's and SGLang's are (theirs is what a block's hash or a tree's key can name;
    /// here a state is named by its position in the one sequence held): a reply shorter than this takes
    /// none, where a grid would cut a third of the 80-token ones. A window gives one at most, so a stride
    /// narrower than a window's kept rows is that window's (SGLang asserts its interval is no less than the
    /// drafts). 0: none.
    reply_stride: usize = 256,
    /// How many of those are held, the newest, beside `snapshots`: the two a running request holds in vLLM
    /// (`MambaSpec.max_memory_usage_bytes`) and SGLang (its ping-pong buffer). Two and not one: a prompt
    /// that parts at or below the newest (a stop string found right after it was taken) starts from the
    /// one before, a stride lower, which is what SGLang's second slot is for (`_select_finished_checkpoint`).
    /// They are their own budget, so no state `plan` keeps goes for a reply's.
    reply_marks: usize = 2,
    /// DFlash2 in the MTP head's place (`dflash2/rig.zig`): opened and held to the model before it loads.
    drafter: ?*const rig.Opened = null,
    /// The drafter's linears BF16 as the file has them, not NVFP4 made at load.
    drafter_stored: bool = false,

    /// What `marks.Planner` reads of these.
    pub fn marking(self: Options) marks_file.Options {
        return .{ .budget = self.snapshots, .stride = self.stride, .near = self.near, .tail = self.tail };
    }

    /// The rows a prompt's window runs under these options on a model loaded for `max_rows`.
    pub fn promptWindow(self: Options, max_rows: usize) usize {
        return @min(max_rows, @max(self.window, self.prompt_rows));
    }
};

/// Rows a prompt's window runs in a round of the pool and in a pair on two ranks, and the least
/// `--max-num-batched-tokens` takes: the pool's rounds are loaded for these (`hall.rowsFor`), so a lone stream's
/// window below them would not hold a round's. A row's bits do not depend on the window's width, so the width is
/// chosen by time alone. On NVFP4 (spec 031, 16257 tokens) windows of 512, 1024 and 2048 rows gave about 1060,
/// 1200 and 1290 tok/s through the same kernels: an expert's weights are read once for the rows a window sends
/// it, which are 10 at 512 rows and 40 at 2048.
pub const prompt_rows = 2048;

/// The rows of a round of the pool and of a pair's prompt windows for a model of `config`.
pub fn promptRows(_: qwen4_config.Config) usize {
    return prompt_rows;
}

/// Rows a lone stream's prompt window runs on one rank unless `--max-num-batched-tokens` says otherwise (`chat`,
/// `serve`'s session and the pool's lone stream), and what the model is loaded for: the owner, 2026-10-28 (spec
/// 148 WS-02), vLLM's card's `--max-num-batched-tokens 4096`. On FP4-Local a 16k cold prompt's routed experts are
/// read half as often as in windows of 2,048, for about 2.6 GiB more of the session's windows.
pub const lone_prompt_rows = 4096;

/// The rows to load a model of `config` for on one rank, so that a lone stream's prompts run in windows of them.
pub fn lonePromptRows(_: qwen4_config.Config) usize {
    return lone_prompt_rows;
}

/// The widest prompt window `--max-num-batched-tokens` may ask (spec 105 WS-07); no wider window was measured.
pub const most_prompt_rows = 4096;

/// Why `asked` rows (`--max-num-batched-tokens`) are not a prompt window for a model of `config`; null where
/// they are. A row's bits do not depend on the window, so the width is time and memory alone. Below
/// `promptRows` the pool's round would not hold its replies' windows beside a chunk (`hall.rowsFor`), so that is
/// the least.
pub fn refusedRows(config: qwen4_config.Config, asked: usize) ?[]const u8 {
    if (asked < promptRows(config) or asked > most_prompt_rows) return "rows from 2048 to 4096";
    return null;
}

/// Buckets of context `chat` and `serve` hand `Session.prepare` while they load.
pub const prepared_buckets = 1;

pub const Reply = struct {
    finish: request_file.Finish,
    /// The token that ended the reply, where `finish` is `.stop`: one of the reply's end tokens.
    end: ?i32 = null,
    /// The reply's tokens, the think block's closing tag and the reasoning before it included.
    tokens: usize,
    reasoning_tokens: usize,
    stats: draft.Stats,
    /// The copies of its context the reply was offered (`decode/copied.zig`); all zero where it was offered none.
    copies: copied.Stats = .{},
    /// The head's levels in its rounds (`mtp.Paying.Counts`).
    levels: mtp.Paying.Counts = .{},
    /// The drafts the reply's first window carried.
    opening: usize = 0,
    /// Of `tokens`, those sent before the window that verified them had run: drafts a constraint forces.
    ahead: usize = 0,
    /// The prompt's tokens that were not run: the state held them.
    cached: usize,
    /// The windows the rest of the prompt ran as.
    windows: usize,
    /// The last of them was launched as a captured graph: a window of a reply's rows, held already.
    graphed: bool = false,
    /// The snapshots of the prompt's last window whose DeltaNet states waited for the first token to be sent.
    settled: usize = 0,
    /// The snapshots taken in them, and in the reply: where its reasoning ended, and on its stride.
    taken: usize,
    /// A digest of every draft the head gave, in order: two replies with the same one were drafted alike.
    drafts: u64,
    /// Seconds in the prompt's windows, and from the reply's first token to its last.
    prompt_seconds: f64,
    seconds: f64,
    /// Under a constraint: the rows masked, the drafts it gave, the drafts cut and the time the masks took. Zeros without one.
    masks: masks_file.Stats = .{},
};

/// Drafts a window at most unless a caller says: all a window takes. TensorFold's `DEPTH` for this family is
/// 6, with its rule that ends a chain by its last draft's probability alone: under that rule deeper chains
/// were verified for nothing where the head lost its way or the reply ended (spec 020: 15 was up to 27%
/// slower on a short reply and 7% faster on one long code reply). The head now ends a chain by the
/// probability of all of it and at an end token (`mtp.Head.chain`, `ends`), and the depth that pays is the
/// window's (spec 028).
pub const default_depth = draft.forward_max_rows - 1;

/// Drafts at most in the first window of a reply that is offered tools and may call one or not
/// (`openingFor`); the owner's choice, 2026-10-06. A call's opening is short and the head drafts well past
/// it, so at the window's depth that first window ran 16 rows before a client was sent the first of the
/// call: its first delta came 0.02 to 0.04 s later than under the six drafts a window took before spec 028.
/// With six the first delta is 0.019 s sooner on NVFP4 and 0.010 s on the pack, and the call's end no later
/// (spec 037 WS-02, a client's times). Every later window takes `depth`, and the cap changes how many drafts
/// a window verifies, never a token. A window of seven rows is one the default depth already captures, so
/// no graph is added.
pub const default_opening_depth = 6;

/// The most drafts the first window of a request's reply takes, of `depth`: `default_opening_depth` where
/// tools are `offered` and the call is free. Where the reply is `constrained` (a required call, a schema) the
/// forced drafts go to the client before their window, and without tools no call opens the reply: both keep
/// `depth`.
pub fn openingFor(offered: bool, constrained: bool, depth: usize) usize {
    return if (offered and !constrained) @min(depth, default_opening_depth) else depth;
}

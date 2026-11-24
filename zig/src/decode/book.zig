//! Which states a sequence holds and where a prompt goes on from them (spec 094; `decode/` since spec 112
//! WS-04): positions alone. Flash Next's two ranks keep their account here, rank 0 deciding and telling rank 1
//! each state taken and each one dropped (`ranks_forward.zig`'s `Op`), so both hold the same positions. The
//! rules are one rank's (`session.zig`'s `resume`, `plan`, `keepWithin` and `keepStride`, over
//! `marks.Planner`), without the sequences left and the states on disk; which of them a window holds is
//! `inside`. Host only: whoever calls takes and drops the states named.
//!
//! A session whose states are lists of its own (Flash Next's one rank, GLM's) loads their positions before it
//! asks (`clear`, `put`), and the Book holds the rules both models keep their states by: where a prompt goes
//! on from, which a plan keeps and drops (`marks.Planner`), the state that closes a reply's reasoning, a reply's
//! on its stride and which of those goes. GLM's rules where they differ are `Options`; what a state is, and
//! whether it is taken ahead of its window or replayed after it, stay the model's.

const std = @import("std");
const Allocator = std.mem.Allocator;
const marks_file = @import("marks.zig");

/// The positions of `at` (ascending) that a window of `rows` rows run from `from` holds, past its first row and
/// up to its last, as rows into it (1 to `rows`): the states a window wider than a layer keeps is asked for
/// before it runs (Flash Next's two ranks, `pair.zig`'s `askWithin`). Into `out`; null where they do not fit.
pub fn inside(at: []const usize, from: usize, rows: usize, out: []u32) ?[]u32 {
    var n: usize = 0;
    for (at) |pos| {
        if (pos <= from or pos > from + rows) continue;
        if (n == out.len) return null;
        out[n] = @intCast(pos - from);
        n += 1;
    }
    return out[0..n];
}

pub const Options = struct {
    /// A prompt's marks, as `session.Options` has them. A budget of 0 keeps no state at all.
    marks: marks_file.Options = .{},
    /// `session.Options.reply_stride` and `reply_marks`.
    reply_stride: usize = 256,
    reply_marks: usize = 2,
    /// A state at p serves a prompt that agrees with the sequence through token p, the head's entry for p - 1
    /// absorbed with it (Flash Next's); false: through token p - 1, and below the prompt's end (GLM's, whose
    /// state at p holds no entry of token p).
    through: bool = true,
    /// A reply's state at the prompt's position joins the marks wherever the state stands there, gone back to
    /// or gone on from the caches (GLM's); false: only where `start` went back to it.
    joins_held: bool = false,
    /// And it joins where a message starts there, standing for the marks less than `near` after it (GLM's: a
    /// reply's end is where the next turn's message starts, which an edit of that message goes on from).
    start_joins: bool = false,
    /// A mark at the prompt's last window's start, where it is neither the state's position nor a message's
    /// start (GLM's: an edit of the prompt's last tokens goes on from it in one window).
    last_window: bool = false,
    /// A reply's states on a grid, one each `reply_stride` tokens from the prompt's end, at the end of the
    /// round that reached it (`gridFrom`, `onGrid`), and one at the reply's end (GLM's); false: one each time
    /// a window's kept rows reach `reply_stride` past the newest state held, at that token (`stride`).
    grid: bool = false,
};

/// Where the state stands for a prompt: as it is, at a state held, or empty.
pub const Start = union(enum) { stay, back: usize, reset };

pub const Resumed = struct {
    /// How many tokens the prompt shares with the sequence the state ran, 0 where that is not trusted.
    shared: usize,
    to: Start,
    /// The states that go first, by position: no state of the prompt's.
    drops: []const usize,
};

/// A state on the reply's stride: where, and the one that goes for it.
pub const Stride = struct { at: usize, drop: ?usize };

/// A state gone back to: where, and whether it is a reply's (one on its stride).
pub const Back = struct { pos: usize, stride: bool };

pub const Book = struct {
    options: Options,
    /// The states a prompt's plan holds, ascending: `[budget + 1]`, `marked` set.
    marks: []usize,
    marked: usize = 0,
    /// Those on a reply's stride, ascending: `[reply_marks]`, `strode` set. Each stands above every mark that
    /// was held when it was taken.
    strides: []usize,
    strode: usize = 0,
    /// `marks.Request.kept` of the prompt running.
    kept: ?usize = null,
    /// Whether the last `start` went back to the newest of `strides`: `plan` may hold it among the marks.
    from_stride: bool = false,
    /// A plan's positions (`marks.Planner.list`, `dropped`), and the positions `start`, `planned` name to go.
    list: []usize,
    dropped: []usize,
    gone: []usize,
    /// The positions the prompt running keeps the state at, ascending (`planned`).
    due: []const usize = &.{},
    /// Whether the last `plan` joined a reply's state to the marks.
    joined: bool = false,
    /// The marks held when the last `plan` began: past the budget where a session put more (GLM's from disk).
    begun: usize = 0,
    /// The next point of a reply's grid (`Options.grid`).
    grid_at: usize = 0,

    pub fn init(gpa: Allocator, options: Options) Allocator.Error!Book {
        const budget = options.marks.budget;
        const marks = try gpa.alloc(usize, budget + 1);
        errdefer gpa.free(marks);
        const strides = try gpa.alloc(usize, options.reply_marks);
        errdefer gpa.free(strides);
        const list = try gpa.alloc(usize, budget + 1);
        errdefer gpa.free(list);
        const dropped = try gpa.alloc(usize, budget + 1);
        errdefer gpa.free(dropped);
        const gone = try gpa.alloc(usize, budget + 1 + options.reply_marks);
        return .{ .options = options, .marks = marks, .strides = strides, .list = list, .dropped = dropped, .gone = gone };
    }

    pub fn deinit(self: *Book, gpa: Allocator) void {
        for ([_][]usize{ self.marks, self.strides, self.list, self.dropped, self.gone }) |one| gpa.free(one);
        self.* = undefined;
    }

    /// The positions a session holds are loaded again: `put` each after this.
    pub fn clear(self: *Book) void {
        self.marked = 0;
        self.strode = 0;
    }

    /// Room for `marks` and `strides` to be `put`, past the budgets `init` sized for (GLM's marks read from
    /// disk join those held past the budget).
    pub fn reserve(self: *Book, gpa: Allocator, marks: usize, strides: usize) Allocator.Error!void {
        const most = @max(marks, self.options.marks.budget) + 2;
        if (self.marks.len < most) {
            for ([_]*[]usize{ &self.marks, &self.list, &self.dropped }) |one| one.* = try gpa.realloc(one.*, most);
        }
        if (self.strides.len < strides + 1) self.strides = try gpa.realloc(self.strides, strides + 1);
        if (self.gone.len < self.marks.len + self.strides.len) self.gone = try gpa.realloc(self.gone, self.marks.len + self.strides.len);
    }

    /// A state held at `at`: a prompt's mark, or a reply's (`on_stride`); each kind ascending.
    pub fn put(self: *Book, at: usize, on_stride: bool) void {
        if (on_stride) {
            std.debug.assert(self.strode == 0 or self.strides[self.strode - 1] < at);
            self.strides[self.strode] = at;
            self.strode += 1;
        } else {
            std.debug.assert(self.marked == 0 or self.marks[self.marked - 1] < at);
            self.marks[self.marked] = at;
            self.marked += 1;
        }
    }

    pub fn count(self: *const Book) usize {
        return self.marked + self.strode;
    }

    /// The position of the newest state held, of either kind; 0 with none.
    pub fn newest(self: *const Book) usize {
        const a = if (self.marked > 0) self.marks[self.marked - 1] else 0;
        const b = if (self.strode > 0) self.strides[self.strode - 1] else 0;
        return @max(a, b);
    }

    /// Where `prompt` goes on from, the state standing at `pos` after the sequence `seen` (its tokens, then the
    /// token chosen after them), which is the state's and the head's where `trusted`. As one rank's `resume`: a
    /// state at position p serves a prompt that agrees with `seen` through token p, since the head's entry for
    /// p - 1 was absorbed with that token. So the state as it stands serves a prompt that holds all of `seen`,
    /// else the highest state held that the prompt agrees past, else none; and every state at or past where
    /// the prompt parts goes, so that none is found later at a position run again with other tokens.
    pub fn start(self: *Book, seen: []const i32, prompt: []const i32, trusted: bool, pos: usize, kept: ?usize) Resumed {
        const shared = if (trusted) std.mem.indexOfDiff(i32, seen, prompt) orelse @min(seen.len, prompt.len) else 0;
        self.kept = kept;
        self.due = &.{};
        var going: usize = 0;
        while (self.strode > 0 and self.strides[self.strode - 1] >= shared) {
            self.strode -= 1;
            self.gone[going] = self.strides[self.strode];
            going += 1;
        }
        while (self.marked > 0 and self.marks[self.marked - 1] >= shared) {
            self.marked -= 1;
            self.gone[going] = self.marks[self.marked];
            going += 1;
        }
        self.from_stride = false;
        const to: Start = to: {
            if (trusted and pos < shared) break :to .stay;
            const back = self.under(std.math.maxInt(usize)) orelse break :to .reset;
            self.from_stride = back.stride;
            break :to .{ .back = back.pos };
        };
        return .{ .shared = shared, .to = to, .drops = self.gone[0..going] };
    }

    /// Below which position a state serves a prompt `len` tokens long that shares `shared` with the sequence
    /// held (`Options.through`).
    pub fn reach(self: *const Book, shared: usize, len: usize) usize {
        return if (self.options.through) shared else @min(shared + 1, len);
    }

    /// The newest state held below `limit`, of either kind (a reply's where both stand there, as `start` and one
    /// rank's resume chose), none at position 0; changes nothing.
    pub fn under(self: *const Book, limit: usize) ?Back {
        var best: ?Back = null;
        for (self.marks[0..self.marked]) |at| if (at > 0 and at < limit) {
            best = .{ .pos = at, .stride = false };
        };
        for (self.strides[0..self.strode]) |at| if (at > 0 and at < limit and (best == null or at >= best.?.pos)) {
            best = .{ .pos = at, .stride = true };
        };
        return best;
    }

    /// The plan of the prompt in `request`, about to run from `request.pos`: the caller runs its windows
    /// through it (`window`, by `Decoder.promptRows`) and hands it to `planned`. A state on a reply's stride
    /// that the state was put back to and that stands at one of the prompt's two marks joins the marks, as one
    /// rank's does (`session.zig`'s `plan`; `joined`), or with `joins_held` and `start_joins` any held there.
    pub fn plan(self: *Book, request: marks_file.Request) marks_file.Planner {
        std.debug.assert(self.options.marks.budget > 0);
        @memcpy(self.list[0..self.marked], self.marks[0..self.marked]);
        self.begun = self.marked;
        var planner: marks_file.Planner = .begin(self.options.marks, request, self.list, self.marked, 0, self.dropped);
        const held = self.strode > 0 and self.strides[self.strode - 1] == request.pos;
        const starts = self.options.start_joins and std.mem.indexOfScalar(usize, request.starts, request.pos) != null;
        self.joined = false;
        if ((if (self.options.joins_held) held else self.from_stride) and (starts or planner.joins())) {
            std.debug.assert(held);
            self.strode -= 1;
            self.marks[self.marked] = request.pos;
            self.marked += 1;
            planner.join();
            // the start stands for the marks less than `near` after it, as one held does (`Planner.begin`)
            if (starts) planner.anchor = request.pos;
            self.joined = true;
        }
        self.from_stride = false;
        return planner;
    }

    /// The plan's window of `rows` rows from `planner.from`, the last window's start first (`last_window`).
    pub fn window(self: *const Book, planner: *marks_file.Planner, rows: usize) void {
        const request = planner.request;
        const from = planner.from;
        if (self.options.last_window and from + rows == request.len and from > request.pos and std.mem.indexOfScalar(usize, request.starts, from) == null)
            planner.add(from);
        planner.window(from + rows);
    }

    /// The plan's end: the positions of the states held that go, and `due` set to those the prompt takes.
    pub fn planned(self: *Book, planner: *const marks_file.Planner) []const usize {
        var going: usize = 0;
        for (planner.drops()) |drop| {
            self.gone[going] = self.marks[drop];
            going += 1;
            std.mem.copyForwards(usize, self.marks[drop .. self.marked - 1], self.marks[drop + 1 .. self.marked]);
            self.marked -= 1;
        }
        std.debug.assert(planner.held == self.marked and planner.count <= @max(self.options.marks.budget, self.begun));
        self.due = planner.taken();
        return self.gone[0..going];
    }

    /// A state was taken at `at`, above every state held: one of `due`, or a caller's own.
    pub fn took(self: *Book, at: usize) void {
        std.debug.assert(self.count() == 0 or self.newest() < at);
        self.marks[self.marked] = at;
        self.marked += 1;
    }

    /// The state at `at`, after the token that closes a reply's reasoning, joins the marks (`session.zig`'s
    /// `keepWithin`). Returns the position of the mark that goes for it where the budget is full, by
    /// `thinnedBut`: never this one.
    pub fn close(self: *Book, at: usize) ?usize {
        std.debug.assert(self.options.marks.budget > 0 and (self.count() == 0 or self.newest() < at));
        const full = self.marked == self.options.marks.budget;
        self.marks[self.marked] = at;
        self.marked += 1;
        if (!full) return null;
        const i = self.over(at + 1);
        const drop = self.marks[i];
        std.mem.copyForwards(usize, self.marks[i .. self.marked - 1], self.marks[i + 1 .. self.marked]);
        self.marked -= 1;
        return drop;
    }

    /// The index among the marks of the one that goes where one too many are held for a sequence `end`
    /// tokens long, by `thinned` (never the newest, nor the one at `kept` while another can go).
    pub fn over(self: *const Book, end: usize) usize {
        return marks_file.thinned(self.marks[0..self.marked], end, self.kept);
    }

    /// Whether a reply's window from `pos` that keeps `keep` rows takes a state on the stride: one each time
    /// the kept rows reach `reply_stride` tokens past the newest state held, at that token (`session_marks.zig`'s
    /// `Ends`). With `reply_marks` held the oldest goes.
    pub fn stride(self: *Book, pos: usize, keep: usize) ?Stride {
        std.debug.assert(!self.options.grid);
        if (self.options.reply_stride == 0 or self.options.reply_marks == 0) return null;
        const at = @max(self.newest() + self.options.reply_stride, pos + 1);
        if (at > pos + keep) return null;
        return .{ .at = at, .drop = self.reply(at) };
    }

    /// A reply's state at `at`: none goes where one is held there already (it is taken again); else with
    /// `reply_marks` held the oldest of them goes, its position returned.
    pub fn reply(self: *Book, at: usize) ?usize {
        std.debug.assert(self.options.reply_marks > 0);
        if (std.mem.indexOfScalar(usize, self.marks[0..self.marked], at) != null) return null;
        if (std.mem.indexOfScalar(usize, self.strides[0..self.strode], at) != null) return null;
        var drop: ?usize = null;
        if (self.strode >= self.options.reply_marks) {
            drop = self.strides[0];
            std.mem.copyForwards(usize, self.strides[0 .. self.strode - 1], self.strides[1..self.strode]);
            self.strode -= 1;
        }
        self.strides[self.strode] = at;
        self.strode += 1;
        return drop;
    }

    /// A reply's grid starts at `pos`, the prompt's end (`Options.grid`).
    pub fn gridFrom(self: *Book, pos: usize) void {
        std.debug.assert(self.options.grid);
        self.grid_at = pos + self.options.reply_stride;
    }

    /// Whether the state, now at `pos` at a round's end, takes the grid's next state: called until false, so
    /// a round that crossed two points takes the state twice (the second in the first's place).
    pub fn onGrid(self: *Book, pos: usize) bool {
        if (self.options.reply_stride == 0 or pos < self.grid_at) return false;
        self.grid_at += self.options.reply_stride;
        return true;
    }

    /// Every position held, ascending, into `out`: `[budget + 1 + reply_marks]` at least.
    pub fn positions(self: *const Book, out: []usize) []const usize {
        var n: usize = 0;
        var a: usize = 0;
        var b: usize = 0;
        while (a < self.marked or b < self.strode) {
            const mark = a < self.marked and (b == self.strode or self.marks[a] < self.strides[b]);
            out[n] = if (mark) self.marks[a] else self.strides[b];
            if (mark) a += 1 else b += 1;
            n += 1;
        }
        return out[0..n];
    }
};

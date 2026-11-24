//! Where a sequence keeps its states, and which goes when it holds one too many: spec 030's marks, the
//! arithmetic both models' sessions and Flash Next's pool keep a prompt's states by (Flash Next's since
//! spec 054, GLM's since spec 112 WS-06), each with its own `Options`. Host only: positions in, positions
//! out; whoever calls takes the states and frees the ones named.

const std = @import("std");

pub const Options = struct {
    /// States kept at most: past it one goes, by `thinned`.
    budget: usize = 8,
    /// A state where a window ends that crossed a multiple of this many positions with no message's start in
    /// it; 0: none.
    stride: usize = 512,
    /// A message's start stands for the starts and marks less than this many tokens after it.
    near: usize = 32,
    /// A state this far before the prompt's end, however near a start.
    tail: usize = 2,
};

/// What a prompt is planned from.
pub const Request = struct {
    /// Where the state the prompt goes on from stands.
    pos: usize,
    /// The prompt's length.
    len: usize,
    /// How many tokens the prompt shares with the sequence held before it.
    shared: usize,
    /// Where the prompt's messages start, ascending.
    starts: []const usize,
    /// Where the prefix this prompt shares with other conversations ends (a message's start, by the server's
    /// account of the system prompt and tools): the state there is never the one that goes (spec 064).
    kept: ?usize = null,
};

/// The positions a prompt keeps, a window at a time. The candidates, in order:
///
/// - where each message starts (`starts`): a later prompt parts from this one inside a message, edited, or at
///   one, the conversation cut back or gone on otherwise, and the state at that message's start is the nearest
///   that serves both;
/// - one token before where this prompt parted from the sequence held (`shared`), which a prompt after it may
///   share too, and `options.tail` before its end, which the next turn's will;
/// - where a window that crossed a multiple of `options.stride` ends, when no message starts in it: the inside
///   of a long message.
///
/// A candidate less than `options.near` after a message's start that is kept is left out, but the mark before
/// the prompt's end, which is always kept. Past `options.budget` one goes, by `thinned`, as if each had been
/// taken in turn: a state held is named in `dropped`, a candidate is never taken. So a prompt of two hundred
/// messages takes eight states.
pub const Planner = struct {
    options: Options,
    request: Request,
    /// The states held, then the positions planned, ascending: `[context + budget + 1]`, `count` set.
    list: []usize,
    count: usize,
    /// Of `list`, the states held: the first `held`.
    held: usize,
    /// Of the states held, the lowest `context` are another's (a pool chain's states are its own above its
    /// fork, and those below it are told here for `near` alone): never dropped, and outside the budget.
    context: usize,
    /// The states held that go, each an index into the list of those held as it stood when it went (so
    /// `orderedRemove` in this order): `[budget + 1]`, `gone` set.
    dropped: []usize,
    gone: usize = 0,
    /// The message's start the highest state kept stands for.
    anchor: ?usize = null,
    start: usize = 0,
    mark: usize = 0,
    marks: [2]usize,
    /// Where the next window runs from.
    from: usize,

    /// `list[0..held]` are the states held, ascending, each at or below `request.pos`.
    pub fn begin(options: Options, request: Request, list: []usize, held: usize, context: usize, dropped: []usize) Planner {
        std.debug.assert(context <= held and list.len >= context + options.budget + 1 and dropped.len >= options.budget + 1);
        var self: Planner = .{
            .options = options,
            .request = request,
            .list = list,
            .count = held,
            .held = held,
            .context = context,
            .dropped = dropped,
            .marks = undefined,
            .from = request.pos,
        };
        const pos = request.pos;
        // The message's start the highest state held stands for: its own position, or that of the state
        // below it when it lies less than `near` above that one, as the mark before an earlier prompt's end
        // does above its header's start. The marks after it are measured from that start.
        while (self.start < request.starts.len and request.starts[self.start] <= pos) : (self.start += 1) {
            const at = request.starts[self.start];
            if (held > 0 and at == list[held - 1]) self.anchor = at;
            if (held > 1 and at == list[held - 2] and list[held - 1] - at < options.near) self.anchor = at;
        }
        const part = request.shared -| 1;
        const turn = request.len -| options.tail;
        self.marks = .{ @min(part, turn), @max(part, turn) };
        return self;
    }

    /// Whether a state that stands at `request.pos` and is not among those held (a session's, on a reply's
    /// stride) is one this prompt would have kept had it run over it: it stands where the prompt parted or at
    /// the turn's mark, and no start kept stands for it.
    pub fn joins(self: *const Planner) bool {
        const pos = self.request.pos;
        const part = self.request.shared -| 1;
        const turn = self.request.len -| self.options.tail;
        return pos < self.request.len and (pos == part or pos == turn) and !(self.anchor != null and pos - self.anchor.? < self.options.near);
    }

    /// The state at `request.pos` joins those held, as if taken first.
    pub fn join(self: *Planner) void {
        std.debug.assert(self.count == self.held and (self.count == 0 or self.list[self.count - 1] < self.request.pos));
        self.held += 1;
        self.planned(self.request.pos);
        self.anchor = null;
    }

    /// The window from `from` to `stop`: what it keeps is added to `list`.
    pub fn window(self: *Planner, stop: usize) void {
        const list = self.list;
        const pos = self.request.pos;
        const len = self.request.len;
        const starts = self.request.starts;
        const near = self.options.near;
        const turn = len -| self.options.tail;
        const stride = self.options.stride;
        std.debug.assert(stop > self.from and stop <= len);
        // a message starts in this window
        var any = false;
        while (true) {
            while (self.mark < self.marks.len and self.marks[self.mark] <= pos) self.mark += 1;
            const a: usize = if (self.start < starts.len) starts[self.start] else len;
            const b: usize = if (self.mark < self.marks.len) self.marks[self.mark] else len;
            const at = @min(a, b);
            if (at > stop or at >= len) break;
            if (a == at) self.start += 1;
            if (b == at) self.mark += 1;
            if (a == at) any = true;
            if (self.count > 0 and at <= list[self.count - 1]) continue;
            // The mark before the prompt's end is kept however near a start (`session.Options.near` says
            // why), and the start goes on standing for what follows it.
            const ending = at == turn;
            const stands = if (self.anchor) |base| at - base < near else false;
            if (stands and !ending) continue;
            if (a == at) self.anchor = at else if (!stands) self.anchor = null;
            self.planned(at);
        }
        // Not the prompt's last window: the state at a prompt's end serves no prompt the live state does not.
        if (!any and stop < len and stride != 0 and stop / stride != self.from / stride and (self.count == 0 or list[self.count - 1] != stop)) {
            self.anchor = null;
            self.planned(stop);
        }
        self.from = stop;
    }

    /// The positions planned, ascending, each above `request.pos` and below the prompt's end.
    pub fn taken(self: *const Planner) []const usize {
        return self.list[self.held..self.count];
    }

    pub fn drops(self: *const Planner) []const usize {
        return self.dropped[0..self.gone];
    }

    /// A position its model keeps beside the candidates above, before the window that runs from it (GLM's,
    /// a prompt's last window's start: `glm5/session/marks.zig`); thinned as they are.
    pub fn add(self: *Planner, at: usize) void {
        std.debug.assert(at > self.request.pos and at < self.request.len and at == self.from);
        if (self.count > 0 and self.list[self.count - 1] == at) return;
        self.planned(at);
    }

    /// Adds `at` to the positions planned; one too many, and the one `thinned` names goes: a state held is
    /// named in `dropped`, a position planned is forgotten.
    fn planned(self: *Planner, at: usize) void {
        const list = self.list;
        list[self.count] = at;
        self.count += 1;
        const own = list[self.context..self.count];
        if (own.len <= self.options.budget) return;
        const drop = self.context + thinned(own, self.request.len, self.request.kept);
        if (drop < self.held) {
            self.dropped[self.gone] = drop;
            self.gone += 1;
            self.held -= 1;
        }
        std.mem.copyForwards(usize, list[drop .. self.count - 1], list[drop + 1 .. self.count]);
        self.count -= 1;
    }
};

/// Which of the states at `held`, ascending positions below `end`, goes when one too many are held for a
/// sequence `end` tokens long: the one whose going leaves the smallest gap for the tokens that would then run
/// after it, the gap between its neighbours over the distance from the lower of them to the end; never the
/// newest. A prompt that parts from the sequence at token x runs x less the state below it again, and all of
/// end - x whatever is held, so a gap far from the end costs little beside what must run anyway and a gap
/// near it is the whole of a request's time. What stays is spaced by its distance from the end: the last turns'
/// own marks, then ever fewer toward the start. (Until spec 026 the one nearest the one below it went, which
/// kept them evenly spread: adding a turn dropped the mark before the previous prompt's end, the one an edited
/// last message starts from, 468 tokens above a stride's.)
///
/// Never the state at `kept` while another can go: where a prefix shared with other conversations ends (a new
/// conversation starts from it however long this one has grown, so its worth is not this sequence's gap).
/// Where only it and the newest are held, the oldest goes.
pub fn thinned(held: []const usize, end: usize, kept: ?usize) usize {
    std.debug.assert(held.len >= 2 and held[held.len - 1] < end);
    var drop: ?usize = null;
    var least_gap: u128 = 0;
    var least_span: u128 = 0;
    for (held[0 .. held.len - 1], 0..) |pos, i| {
        if (kept != null and pos == kept.?) continue;
        const below = if (i == 0) 0 else held[i - 1];
        const gap: u128 = held[i + 1] - below;
        const span: u128 = end - below;
        // gap / span < least_gap / least_span
        if (drop == null or gap * least_span < least_gap * span) {
            drop = i;
            least_gap = gap;
            least_span = span;
        }
    }
    return drop orelse 0;
}

/// The message's start a state is kept at for the prefix the server found shared with other conversations
/// (`Request.shared_prefixes`: its cuts up to where the prompt and a probe in the first user message's place
/// part, spec 064): the highest start below the longest cut, which is that message's header. TensorFold keeps
/// a state at each cut; here a state stands at a message's start whatever the cut, and one is enough: a row's
/// bits do not depend on where a window began.
pub fn keptOf(starts: []const usize, cuts: []const u32) ?usize {
    const cut = std.mem.max(u32, if (cuts.len > 0) cuts else return null);
    var found: ?usize = null;
    for (starts) |at| {
        if (at >= 1 and at < cut) found = at;
    }
    return found;
}

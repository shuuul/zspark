//! A sequence's committed state at one position, to return to. The latents and the index keys stay in the
//! live cache: rows below the position stay valid, and a row never reads one at or past its own position.
//! Cutting the cache makes the pool that holds the cut unfinished (`dsa.Cache.cut`). Modelled on Flash Next's
//! `snapshot.zig`. Which marks are held, which goes, which one a prompt goes back to and whose buffers the next one
//! takes are the shared layer's (`decode/shelf.zig`, `decode/marks.zig`, `decode/book.zig`); where a mark is
//! taken is the session's.

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const kda = @import("../layers/kda.zig");
const model = @import("../layers/model.zig");

/// How many of a conversation's states a session holds of its prompts (`marks.zig`'s plan), and how many of its
/// replies, on their stride and at their end, beside them: Flash Next's 8 and 2.
pub const held_default = 8;
pub const reply_extra = 2;
/// A reply keeps a state each this many of its tokens.
pub const stride_default = 256;

/// One rank's recurrent state at `pos`. The attention rows are the cache's.
pub const Shot = struct {
    pos: usize,
    /// The KDA layers' conv window and the side of their state a window reads. Null where the model has none.
    kda: ?kda.Mark = null,

    pub fn deinit(self: *Shot) void {
        if (self.kda) |*kept| kept.deinit();
        self.* = undefined;
    }
};

/// `state` as it stands: its committed KDA state and conv window, in `spare`'s device buffers where one is
/// given (a dropped shot's, lent by the session's `decode/shelf.zig` `Shelf`): no allocation on a reply's way.
pub fn takeRank(state: model.State, spare: ?kda.Mark) cuda.Error!Shot {
    var shot: Shot = .{ .pos = state.pos };
    if (state.kda) |chains| {
        var kept: kda.Mark = spare orelse try .init(chains.layers, chains.heads);
        errdefer kept.deinit();
        try kept.take(chains);
        shot.kda = kept;
    }
    return shot;
}

/// `takeRank` of the state after the first `rows` rows of the window that has run on `state` and is not
/// committed (`kda.Mark.takeWithin`): the shot stands at `state.pos + rows`, and the cache's rows below it are
/// the window's once it commits. A window that ran chunked wrote the state into `spare` as it ran, which the
/// mark was asked for before it (`kda.State.ask`, spec 116): its conv window alone is left.
pub fn takeWithin(state: model.State, rows: usize, spare: ?kda.Mark) cuda.Error!Shot {
    var shot: Shot = .{ .pos = state.pos + rows };
    if (state.kda) |chains| {
        var kept: kda.Mark = spare orelse try .init(chains.layers, chains.heads);
        errdefer kept.deinit();
        if (chains.window.chunked) {
            std.debug.assert(chains.wrote(rows, kept));
            try kept.convWithin(chains, rows);
        } else try kept.takeWithin(chains, rows);
        shot.kda = kept;
    }
    return shot;
}

/// `state` is `shot` again. The cache is cut at the shot's position; rows below it stay.
pub fn restoreRank(shot: Shot, state: *model.State) cuda.Error!void {
    if (shot.kda) |kept| try kept.restore(state.kda.?);
    if (state.dsa) |*cache| try cache.cut(shot.pos);
    state.pos = shot.pos;
}

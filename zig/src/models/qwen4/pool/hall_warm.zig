//! Warm turns (`serve --warm-turns`, spec 106 WS-07; TensorFold 1.0.12's `warm_turns`): after a chat reply the
//! host sends the conversation again with the reply as its client got it and the next user turn's header
//! (`server/warm.zig`), and the hall runs it as a reply of one token that nobody is sent (`Ask.warm`). What it
//! is for are the states its prompt keeps on the way (`hall_keep.zig`, the session's marks): the next turn
//! goes on from the one two tokens before its end (`marks.Options.tail`), and runs its new message alone,
//! where without the pass it would run the reply again, which the template renders otherwise than it was
//! decoded (its reasoning left out).
//!
//! A pass never stands beside a live stream, so no live request waits on it for more than the round that is
//! running when it comes:
//!
//! - It is declined while any other stream is in the hall (`admits`): its prompt would ride in their rounds.
//! - As the lone stream, a request that comes is told to wait (`Core.admit`) rather than open the pool beside
//!   it; the next round ends the pass unrun (`stepsAside`), and the request is the lone stream after it. A
//!   lone session runs a prompt whole before it can be asked to stop, so a pass there runs at most one prompt
//!   window past what the session last ran, and a longer one is declined.
//! - In the pool, a pass is ended at the top of any round where another seat is live or a request waits
//!   (`yieldSeats`), before a prompt is chosen to fill: the oldest prompt would otherwise be the pass's.
//!
//! - Before it is queued, it waits `quiet_for` (TensorFold's `warm_quiet`, its Nemotron's 150 ms) for other
//!   work (`quiet`): a request that comes then, such as an agent's next call right after a reply, finds no
//!   pass in its way, and the pass is dropped unrun.
//!
//! Its one token's `.end` state is not kept (`Engine.retire`): it stands on a guess at the user's first token.
//! Where TensorFold departs: its lane host lets a pass queue behind requests and yields between a pass's
//! chunks; here a pass declined is dropped, since the next turn runs whole without it all the same, and a lone
//! pass yields only before its prompt.

const std = @import("std");
const hall_file = @import("hall.zig");
const hall_engine = @import("hall_engine.zig");
const Ask = hall_file.Ask;
const Core = hall_file.Core;
const Out = hall_file.Out;
const sched = @import("../../../server/scheduler.zig");

/// How long a pass waits for other work before it is queued (TensorFold 1.0.12's Nemotron `warm_quiet`).
pub const quiet_for: std.Io.Duration = .fromMilliseconds(150);
const quiet_step: std.Io.Duration = .fromMilliseconds(5);

/// Waits `quiet_for` while `scheduler` admits nothing, queues nothing and `live` (its pass's request) is still
/// wanted: true if all of it held, and the pass is queued; false drops it unrun.
pub fn quiet(io: std.Io, scheduler: *sched.Scheduler, live: anytype) bool {
    const before = scheduler.counts().admitted;
    var waited: i96 = 0;
    while (waited < quiet_for.nanoseconds) : (waited += quiet_step.nanoseconds) {
        const now = scheduler.counts();
        if (now.closing or now.queued > 0 or now.admitted != before or live.cancelled()) return false;
        io.sleep(quiet_step, .awake) catch return false;
    }
    const now = scheduler.counts();
    return !(now.closing or now.queued > 0 or now.admitted != before or live.cancelled());
}

/// Whether `ask`, a warm pass, may take `core` now: nothing else is in it, and on the lone session its prompt
/// runs in one window past the sequence the session last ran.
pub fn admits(core: *const Core, ask: Ask) error{Declined}!void {
    if (core.lone != null) return error.Declined;
    if (core.engine) |*engine| {
        for (engine.seats) |seat| if (seat != null) return error.Declined;
        return;
    }
    if (ask.prompt.len - sharedLen(core.sequence.items, ask.prompt) > core.options.session.prompt_rows) return error.Declined;
}

/// The tokens `a` and `b` begin with alike.
pub fn sharedLen(a: []const i32, b: []const i32) usize {
    return std.mem.indexOfDiff(i32, a, b) orelse @min(a.len, b.len);
}

/// A lone warm pass ends unrun when its client is gone or a request waits: true when it was ended.
pub fn stepsAside(warm: bool, slot: usize, out: Out) bool {
    if (!warm or !(out.cancelled(out.ctx, slot) or out.waiting(out.ctx))) return false;
    out.stop(slot, .cut);
    return true;
}

/// Ends each warm seat of `engine` that another live seat or a waiting request finds there, before the round
/// chooses what to run: it runs nothing more, and the scheduler retires it after the round.
pub fn yieldSeats(engine: *hall_engine.Engine, out: Out) void {
    const waits = out.waiting(out.ctx);
    for (engine.seats, 0..) |maybe, i| {
        const seat = maybe orelse continue;
        if (!seat.warm or seat.phase == .done) continue;
        const beside = for (engine.seats, 0..) |other, j| {
            if (j != i and other != null and other.?.phase != .done and !other.?.warm) break true;
        } else false;
        if (!beside and !waits and !out.cancelled(out.ctx, i)) continue;
        out.stop(i, .cut);
        engine.seats[i].?.phase = .done;
    }
}

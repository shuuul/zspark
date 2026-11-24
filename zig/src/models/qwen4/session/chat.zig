//! The text side of a reply: whether a prompt leaves the reply thinking (`thinks`), a request as the pool's
//! job takes it (`Prepared`, which `served.zig` fills for `hall_front.makeJob`), and a reply's tokens made
//! text (`Speaker`) on their way to a sink (`Sending`, which `Session.reply` fills): `zspark chat` writes its
//! reply through one, and `zspark serve` taps the tokens before it (`session.Tap`).

const std = @import("std");
const request_file = @import("../../../server/request.zig");
const grammar = @import("../../../text/grammar.zig");
const sampling = @import("../../../decode/sampling.zig");
const tokenizer = @import("../../../text/tokenizer.zig");
const session_file = @import("session.zig");
const ReplyOptions = session_file.ReplyOptions;
const Session = session_file.Session;

/// The `<think>` and `</think>` tokens of a checkpoint that has them.
pub const Think = request_file.Think;
pub const thinks = request_file.thinks;

/// A request ready to be decoded. `prompt` is the arena's; the constraint is the caller's to free.
pub const Prepared = struct {
    prompt: []i32,
    settings: sampling.Sampling,
    max_tokens: usize,
    /// The prompt leaves the reply thinking.
    thinking: bool,
    constraint: ?grammar.Constraint,
    /// Log probabilities, this many best tokens a row (`ReplyOptions.logprobs`).
    logprobs: ?u8 = null,
};

/// A reply's tokens counted and made text: the reasoning until its closing tag, then the content without the
/// blank lines the template puts after the tag. `counted` is the one place a session's reply counts a token.
pub const Speaker = struct {
    decoder: *tokenizer.StreamDecoder,
    /// `</think>` where the checkpoint has one.
    close: ?i32,
    /// The text is the reasoning's, until its closing tag.
    reasoning: bool,
    /// After the closing tag: the blank lines the template puts there are dropped.
    leading: bool = false,
    /// The reply's tokens so far.
    count: usize = 0,
    reasoning_tokens: usize = 0,

    /// The reply's tokens with `token`, or null where `token` is one of `eos`: it ends the reply and is none
    /// of its tokens, so it is neither counted nor said.
    pub fn counted(self: *Speaker, token: i32, eos: []const u32) ?usize {
        if (std.mem.indexOfScalar(u32, eos, @intCast(token)) != null) return null;
        self.count += 1;
        return self.count;
    }

    /// The text `token` adds, to `sink`; `counted` took the token.
    pub fn say(self: *Speaker, token: i32, sink: request_file.Sink) request_file.Sink.Signal {
        if (self.reasoning and self.close != null and token == self.close.?) {
            // The reasoning ends here; what the decoder still holds belongs to it.
            self.reasoning_tokens = self.count;
            const signal = sink.write(.reasoning, self.decoder.finish());
            self.reasoning = false;
            self.leading = true;
            return signal;
        }
        var piece = self.decoder.next(@intCast(token));
        if (self.leading) piece = std.mem.trimStart(u8, piece, "\n");
        if (piece.len == 0) return .more;
        self.leading = false;
        return sink.write(if (self.reasoning) .reasoning else .content, piece);
    }

    /// The reply ended: what the decoder still holds is written, unless the client left.
    pub fn end(self: *Speaker, cut: bool, sink: request_file.Sink) void {
        if (self.reasoning) self.reasoning_tokens = self.count;
        const rest = self.decoder.finish();
        if (rest.len > 0 and !cut) _ = sink.write(if (self.reasoning) .reasoning else .content, rest);
    }
};

/// A reply's tokens on their way to its sink, in the order the reply holds them.
pub const Sending = struct {
    session: *const Session,
    options: ReplyOptions,
    sink: request_file.Sink,
    /// It counts the reply's tokens (`Speaker.count`).
    speaker: Speaker,
    /// The end token the reply stopped at.
    end: ?i32 = null,

    /// The reply's next token: how the reply ends at it, or null while it goes on. `row` is its row for a reply
    /// that asks for log probabilities: the tap is given it before the token, and for the end token too.
    pub fn take(self: *Sending, next: i32, row: ?*const request_file.LogprobRow) ?request_file.Finish {
        if (self.speaker.count == self.options.max_tokens) return .length;
        if (row) |one| if (self.options.tap) |tap| if (tap.rowFn) |give| give(tap.context, one);
        const count = self.speaker.counted(next, self.options.eos orelse self.session.eos) orelse {
            self.end = next;
            return .stop;
        };
        if (self.options.record) |record| record[count - 1] = next;
        if (self.options.tap) |tap| {
            tap.take(next);
            return null;
        }
        return if (self.speaker.say(next, self.sink) == .stop) .cut else null;
    }
};

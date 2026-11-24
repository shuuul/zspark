//! What a request and a reply are called between a model and what stands before it: a reply's structure
//! (`Structure`, with the tools a call is held to: `Request`), where its text goes (`Sink`), how it ended
//! (`Finish`), what a refusal says (`Diagnostic`, `Error`) and a token's log probabilities (`LogprobRow`). `server/structured.zig` compiles a `Request`,
//! `server/scheduler.zig` carries the rest between its threads, and `models/qwen4/session/session.zig` writes a
//! reply's text to a `Sink` for `zspark chat`.

const std = @import("std");
const grammar = @import("../text/grammar.zig");

/// What a reply must be, where a request asks for more than text (spec 017). The engine compiles it for its
/// own tokenizer (`structured.Compiler`) and decodes under it; a schema it cannot hold is `BadRequest`.
pub const Structure = union(enum) {
    /// A JSON document and nothing else: `response_format` and what the other APIs map onto it.
    document: Document,
    /// One call or more to `Request.tools`, and nothing else: `tool_choice` "required" or a named function,
    /// whose tool is then the only one in `tools`. Each call's arguments are its tool's `parameters`' document.
    call: Call,

    pub const Document = struct {
        spec: grammar.Spec,
        /// The request's field, as the client named it: a refusal's first words.
        field: []const u8 = "response_format",
    };

    pub const Call = struct {
        /// One call and the end (`parallel_tool_calls: false`).
        single: bool = false,
    };
};

/// What of a request its reply's constraint is made from (`structured.Compiler.compile`).
pub const Request = struct {
    /// The tools offered (after `tool_choice`), or null.
    tools: ?std.json.Value = null,
    /// Null is a reply of free text: nothing constrains it and no grammar is asked anything.
    structure: ?Structure = null,
};

/// `<think>` and `</think>` as the tokenizer holds them.
pub const Think = struct { open: i32, close: i32 };

/// Whether `prompt` leaves the reply thinking: its last `<think>` is not closed.
pub fn thinks(think: ?Think, prompt: []const i32) bool {
    const tags = think orelse return false;
    var thinking = false;
    for (prompt) |id| {
        if (id == tags.open) thinking = true;
        if (id == tags.close) thinking = false;
    }
    return thinking;
}

pub const Kind = enum { reasoning, content };

/// Where the reply's text goes as it is decoded: whole characters, in order, the think block's tags left out.
pub const Sink = struct {
    context: *anyopaque,
    writeFn: *const fn (context: *anyopaque, kind: Kind, text: []const u8) Signal,

    /// `stop` ends the reply after this piece: a stop string was met, or the client has gone.
    pub const Signal = enum { more, stop };

    /// An empty piece writes nothing and is answered all the same: who asks whether a reply that is not coming
    /// yet is still wanted asks with one.
    pub fn write(self: Sink, kind: Kind, text: []const u8) Signal {
        return self.writeFn(self.context, kind, text);
    }
};

pub const Finish = enum {
    /// The model ended the reply.
    stop,
    /// `max_tokens` or the context ended it.
    length,
    /// The sink asked for the end.
    cut,
};

pub const Diagnostic = struct {
    buffer: [640]u8 = undefined,
    len: usize = 0,

    pub fn message(d: *const Diagnostic) []const u8 {
        return d.buffer[0..d.len];
    }

    pub fn set(d: *Diagnostic, comptime format: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&d.buffer);
        w.print(format, args) catch {};
        d.len = w.buffered().len;
    }
};

/// `BadRequest` is the client's (a 400 with the diagnostic's message: a prompt past the context, a schema that
/// cannot be enforced); `Failed` is ours (a 500).
pub const Error = error{ OutOfMemory, BadRequest, Failed };

/// A reply's token with its log probability and its row's best tokens (`decode/logprob.zig`).
pub const LogprobRow = @import("../decode/logprob.zig").Row;

//! zspark: speech to text, ``POST /v1/audio/transcriptions`` (spec 138): what the server hands an engine and what
//! the engine writes back. The engine decodes the file, makes the prompt and reads the reply's text (vLLM's
//! ``SupportsTranscription``: ``get_generation_prompt``, ``post_process_output``); the server reads the form and
//! words the answer.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// What a transcription may hold, fixed once the engine has loaded.
pub const TranscriptionInfo = struct {
    /// The longest audio taken, in seconds; longer is refused.
    max_seconds: f64 = 1200,
    /// Reply tokens when the request names none.
    max_tokens: u32 = 4096,
};

/// Caller-owned until finished, even on cancellation.
pub const TranscriptionRequest = struct {
    /// The uploaded file's bytes, any format; the engine says what it decodes.
    audio: []const u8,
    /// The form's ``language``, as sent; null: not sent.
    language: ?[]const u8 = null,
    /// The form's ``prompt``: the context the model is given.
    prompt: []const u8 = "",
    max_tokens: u32,
    /// Where the engine writes the answer, each slice from ``gpa`` (``TranscriptionResult.deinit``).
    result: *TranscriptionResult,
    gpa: Allocator,
};

pub const TranscriptionResult = struct {
    text: []const u8 = "",
    /// The language the reply named or the request forced; empty: none.
    language: []const u8 = "",
    /// The decoded audio's length.
    seconds: f64 = 0,
    prompt_tokens: u32 = 0,
    tokens: u32 = 0,
    /// The request cannot be served as sent (the file, its length, the language): a 400 with these words.
    refused: []const u8 = "",
    /// The refused field, where the refusal names one.
    param: []const u8 = "",

    pub fn deinit(r: *TranscriptionResult, gpa: Allocator) void {
        for ([_][]const u8{ r.text, r.language, r.refused, r.param }) |s| if (s.len > 0) gpa.free(s);
        r.* = .{};
    }
};

/// zspark: audio parts on chat (spec 150), as vLLM reads an ``audio_url`` or ``input_audio`` part for a model with
/// an audio tower: the host fetches a part's bytes, the engine decodes them (``prepare``) and says the rows they
/// take, and the host writes that many ``audio_token`` ids where the template wrote one (``server/audios.zig``).
pub const AudioOffer = struct {
    ctx: *anyopaque,
    /// A part's bytes, any format, to the engine's input; allocated in ``a``. An error the engine words
    /// (``explain``) is the client's 400 in those words.
    prepare: *const fn (ctx: *anyopaque, a: Allocator, bytes: []const u8) anyerror!PreparedAudio,
    explain: ?*const fn (err: anyerror) ?[]const u8 = null,
    /// The token the chat template writes once an audio part, expanded to the audio's rows.
    audio_token: u32,
    /// Audio parts one request may hold, and their seconds and rows in all.
    max_audios: u32,
    max_seconds: f64,
    max_tokens: u32,
    /// An audio part's encoded bytes at most.
    max_bytes: usize,
};

/// One audio part as the engine decoded it: its samples at the engine's rate and the prompt rows they take.
pub const PreparedAudio = struct { samples: []const f32, seconds: f64, tokens: u32 };

/// One audio of a request (``Request.audios``): its samples, and the prompt rows from ``at`` its features replace.
pub const AudioClip = struct { samples: []const f32, at: u32, tokens: u32 };

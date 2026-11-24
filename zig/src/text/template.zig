//! A checkpoint's chat template, rendered by TensorFold's engine (its `template` module, a dependency of the
//! build, spec 059): the engine `zspark serve`'s host renders a request with, so `chat`, the probes of
//! `frontend.zig` and a served prompt are one template's bytes. Until spec 059 WS-06 this file was a Jinja
//! subset of zspark's own.
//!
//! `Template` keeps the interface that subset had: parsed once from the source, rendered against a JSON object
//! whose members are the template's variables. TensorFold's `render` takes `messages`, `tools` and
//! `add_generation_prompt` apart from the rest, gives `tools` as none where it is absent and adds `documents`
//! as none; this file parts the object so. A failure has no line: the engine gives a message alone.
//!
//! The text helpers below (Python's whitespace and characters, its float's `repr`) are what `pyjson.zig` reads
//! a call's arguments with and `server/structured.zig` a tool's name.

const std = @import("std");
const Allocator = std.mem.Allocator;
const engine = @import("template");

pub const ParseError = error{ OutOfMemory, TemplateSyntax };
pub const RenderError = error{ OutOfMemory, TemplateRaised, TemplateRuntime };

/// Why a parse or a render failed. For `error.TemplateRaised` the message is the template's own.
pub const Diagnostic = struct {
    buffer: [512]u8 = undefined,
    len: usize = 0,

    pub fn message(d: *const Diagnostic) []const u8 {
        return d.buffer[0..d.len];
    }

    pub fn set(d: *Diagnostic, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&d.buffer);
        w.print(fmt, args) catch {};
        d.len = w.buffered().len;
    }
};

// ----------------------------------------------------------------------------------------------------------------
// Text helpers: Python's notion of whitespace and of a string's characters

/// `str.isspace()` for one code point, which is also what `\s` matches in jinja2's lexer.
pub fn isSpace(c: u21) bool {
    return switch (c) {
        0x09...0x0d, 0x1c...0x20, 0x85, 0xa0, 0x1680, 0x2000...0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000 => true,
        else => false,
    };
}

/// The length in bytes of the character at the start of `s`; a byte that starts no valid sequence counts alone.
pub fn charLen(s: []const u8) usize {
    const n = std.unicode.utf8ByteSequenceLength(s[0]) catch return 1;
    if (n > s.len) return 1;
    _ = std.unicode.utf8Decode(s[0..n]) catch return 1;
    return n;
}

pub fn charAt(s: []const u8) u21 {
    const n = charLen(s);
    return std.unicode.utf8Decode(s[0..n]) catch 0xfffd;
}

/// The start of the character that ends at `end`.
fn charBefore(s: []const u8, end: usize) usize {
    var start = end - 1;
    while (start > 0 and end - start < 4 and s[start] & 0xc0 == 0x80) start -= 1;
    return if (start + charLen(s[start..end]) == end) start else end - 1;
}

pub fn skipSpace(s: []const u8, from: usize) usize {
    var i = from;
    while (i < s.len) {
        if (!isSpace(charAt(s[i..]))) break;
        i += charLen(s[i..]);
    }
    return i;
}

/// `str.strip`, `lstrip` and `rstrip`: of whitespace, or of the characters in `chars`.
pub fn strip(s: []const u8, chars: ?[]const u8, left: bool, right: bool) []const u8 {
    var start: usize = 0;
    var end = s.len;
    if (left) while (start < end) {
        const n = charLen(s[start..end]);
        if (!stripped(s[start..][0..n], chars)) break;
        start += n;
    };
    if (right) while (end > start) {
        const at = @max(charBefore(s, end), start);
        if (!stripped(s[at..end], chars)) break;
        end = at;
    };
    return s[start..end];
}

fn stripped(char: []const u8, chars: ?[]const u8) bool {
    const set = chars orelse return isSpace(charAt(char));
    var i: usize = 0;
    while (i < set.len) {
        const n = charLen(set[i..]);
        if (std.mem.eql(u8, set[i..][0..n], char)) return true;
        i += n;
    }
    return false;
}

pub fn among(names: []const []const u8, name: []const u8) bool {
    for (names) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// Python's `repr(float)`: the shortest digits that read back to the same double, in positional notation while
/// the decimal exponent is in [-4, 16) and in scientific notation with a two-digit exponent outside it.
pub fn appendFloat(arena: Allocator, out: *std.ArrayList(u8), f: f64, nan: []const u8, inf: []const u8) Allocator.Error!void {
    if (std.math.isNan(f)) return out.appendSlice(arena, nan);
    if (std.math.isInf(f)) {
        if (f < 0) try out.append(arena, '-');
        return out.appendSlice(arena, inf);
    }
    var buf: [64]u8 = undefined;
    const sci = std.fmt.float.render(&buf, f, .{ .mode = .scientific }) catch unreachable;
    const e_at = std.mem.indexOfScalar(u8, sci, 'e').?;
    const exponent = std.fmt.parseInt(i32, sci[e_at + 1 ..], 10) catch unreachable;
    var digits_buf: [32]u8 = undefined;
    var count: usize = 0;
    for (sci[0..e_at]) |c| if (std.ascii.isDigit(c)) {
        digits_buf[count] = c;
        count += 1;
    };
    const digits = digits_buf[0..count];
    if (sci[0] == '-') try out.append(arena, '-');
    if (exponent < -4 or exponent >= 16) {
        try out.append(arena, digits[0]);
        if (digits.len > 1) {
            try out.append(arena, '.');
            try out.appendSlice(arena, digits[1..]);
        }
        try out.print(arena, "e{c}{d:0>2}", .{ @as(u8, if (exponent < 0) '-' else '+'), @abs(exponent) });
    } else if (exponent < 0) {
        try out.appendSlice(arena, "0.");
        try out.appendNTimes(arena, '0', @intCast(-exponent - 1));
        try out.appendSlice(arena, digits);
    } else {
        const whole: usize = @intCast(exponent + 1);
        if (digits.len <= whole) {
            try out.appendSlice(arena, digits);
            try out.appendNTimes(arena, '0', whole - digits.len);
            try out.appendSlice(arena, ".0");
        } else {
            try out.appendSlice(arena, digits[0..whole]);
            try out.append(arena, '.');
            try out.appendSlice(arena, digits[whole..]);
        }
    }
}

// ----------------------------------------------------------------------------------------------------------------
// The public interface

/// A parsed chat template. Parse once per checkpoint, render once per request.
pub const Template = struct {
    gpa: Allocator,
    compiled: engine.Compiled,

    /// Parses `source`, which need not outlive the call. On `error.TemplateSyntax` the `Diagnostic` says what
    /// was found; a filter or a test the engine does not have is refused here.
    pub fn parse(gpa: Allocator, source: []const u8, diag: *Diagnostic) ParseError!Template {
        var said: engine.Diag = .{};
        defer gpa.free(said.msg);
        const compiled = engine.compile(gpa, source, &said) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TemplateFailed => {
                diag.set("{s}", .{said.msg});
                return error.TemplateSyntax;
            },
        };
        return .{ .gpa = gpa, .compiled = compiled };
    }

    pub fn deinit(self: *Template) void {
        self.compiled.deinit();
    }

    /// Renders against `context`, a JSON object whose members are the template's variables (`messages`, `tools`,
    /// `add_generation_prompt`, `enable_thinking`, ...), into memory the caller frees with `gpa`.
    ///
    /// `error.TemplateRaised` is the template's own `raise_exception`, its message in `diag`;
    /// `error.TemplateRuntime` is anything else the engine fails a render for.
    pub fn renderAlloc(self: *const Template, gpa: Allocator, context: std.json.Value, diag: *Diagnostic) RenderError![]u8 {
        if (context != .object) {
            diag.set("the context must be a JSON object", .{});
            return error.TemplateRuntime;
        }
        const messages = context.object.get("messages") orelse std.json.Value.null;
        const tools = context.object.get("tools") orelse std.json.Value.null;
        const generation = if (context.object.get("add_generation_prompt")) |v| v == .bool and v.bool else false;
        var said: engine.Diag = .{};
        defer gpa.free(said.msg);
        return engine.render(gpa, &self.compiled, messages, tools, context, generation, &said) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TemplateFailed => {
                diag.set("{s}", .{said.msg});
                return if (said.raised) error.TemplateRaised else error.TemplateRuntime;
            },
        };
    }
};

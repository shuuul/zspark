//! What the commands share that names no model: the version and the build's stamp, the words of a command
//! line (`key=value`, token ids, a table of `--flag`s), the prompt a `chat` builds from them and the opening of
//! a checkpoint directory.

const std = @import("std");
const safetensors = @import("weights/safetensors.zig");
const text_frontend = @import("text/frontend.zig");
const text_messages = @import("text/messages.zig");
const sampling = @import("decode/sampling.zig");

/// The commit this binary was built from and whether the tree differed from it (`build.zig`'s stamp; spec 053):
/// what `tools/check.sh` reads before it measures, so that a stale binary is not taken for the head. The stamp
/// also carries the calendar version (`build.zig.zon`'s `version`, spec 170) and the build mode's name.
pub const stamp = @import("stamp");
/// The calendar version with the build mode after a `+` (`2026.11.12+fast`): one commit's safe and fast
/// binaries load and run at different speeds, so a measurement names which it took.
pub const version = stamp.version ++ "+" ++ stamp.optimize;

/// What follows `key` in `word` (`value("rows=64", "rows=")` is `64`), or null when the word is another.
pub fn value(word: []const u8, comptime key: []const u8) ?[]const u8 {
    return if (std.mem.startsWith(u8, word, key)) word[key.len..] else null;
}

/// The token ids `words` spell, one a word.
pub fn ids(arena: std.mem.Allocator, words: []const []const u8) ![]i32 {
    const tokens = try arena.alloc(i32, words.len);
    for (tokens, words) |*token, text| token.* = try std.fmt.parseInt(i32, text, 10);
    return tokens;
}

/// The token ids of the file at `path`, separated by blanks or lines.
pub fn readIds(io: std.Io, arena: std.mem.Allocator, path: []const u8) ![]i32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .unlimited);
    var list: std.ArrayList(i32) = .empty;
    var words = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (words.next()) |word| try list.append(arena, try std.fmt.parseInt(i32, word, 10));
    return list.items;
}

/// One `--flag` of a command: the field of its options that the flag sets. A `bool` field makes it a switch,
/// set to `on`; any other field takes the next argument, read as the field's type.
pub const Flag = struct {
    name: []const u8,
    field: []const u8,
    /// What a switch sets its field to.
    on: bool = true,
    /// The least and the most a number may be, an optional field's too; a value outside (or a NaN) is not one
    /// the flag takes.
    min: ?f64 = null,
    max: ?f64 = null,
};

/// How a command reads its arguments, beside its flags.
pub const Words = struct {
    /// The field an argument that is not a flag goes to, and what is said of a second one. Without it every
    /// argument is read as a flag.
    positional: ?struct { field: []const u8, again: []const u8 } = null,
    /// Whether a refused value is printed between quotes: the commands differ here, and each keeps its own.
    quoted: bool,
    /// What every refusal's line starts with (`serve: `).
    prefix: []const u8 = "",
};

/// Why `read` took no options, in the words `parse` logs.
pub const Refusal = union(enum) {
    /// A second argument that is not a flag: the command's own words for it.
    again: []const u8,
    /// A flag that takes a value, in the last place.
    valueless: []const u8,
    /// An argument that is no flag of the command.
    unknown: []const u8,
    /// A value its flag does not take.
    untaken: struct { flag: []const u8, text: []const u8, quoted: bool },

    pub fn format(self: Refusal, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .again => |said| try writer.writeAll(said),
            .valueless => |flag| try writer.print("{s} needs a value", .{flag}),
            .unknown => |flag| try writer.print("unknown option {s}", .{flag}),
            .untaken => |u| if (u.quoted)
                try writer.print("{s}: '{s}' is not a value it takes", .{ u.flag, u.text })
            else
                try writer.print("{s}: {s} is not a value it takes", .{ u.flag, u.text }),
        }
    }
};

/// `args` read into an `Options` by `flags`, or null with the reason logged.
pub fn parse(comptime Options: type, comptime flags: []const Flag, comptime words: Words, args: []const []const u8) ?Options {
    var refusal: Refusal = undefined;
    return read(Options, flags, words, args, &refusal) orelse {
        std.log.err("{s}{f}", .{ words.prefix, refusal });
        return null;
    };
}

/// `parse` without the log: null leaves the reason in `refusal`. A flag's name is looked up before its value
/// is taken, so an unknown flag is said to be unknown wherever it stands, the last place included (spec 056).
pub fn read(comptime Options: type, comptime flags: []const Flag, comptime words: Words, args: []const []const u8, refusal: *Refusal) ?Options {
    var o: Options = .{};
    var i: usize = 0;
    next: while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (words.positional) |positional| if (!std.mem.startsWith(u8, arg, "--")) {
            if (@field(o, positional.field) != null) {
                refusal.* = .{ .again = positional.again };
                return null;
            }
            @field(o, positional.field) = arg;
            continue;
        };
        inline for (flags) |flag| if (@FieldType(Options, flag.field) == bool and std.mem.eql(u8, arg, flag.name)) {
            @field(o, flag.field) = flag.on;
            continue :next;
        };
        const ok = inline for (flags) |flag| {
            if (@FieldType(Options, flag.field) != bool and std.mem.eql(u8, arg, flag.name)) {
                i += 1;
                if (i == args.len) {
                    refusal.* = .{ .valueless = arg };
                    return null;
                }
                const taken = as(@FieldType(Options, flag.field), args[i]);
                if (taken) |held| @field(o, flag.field) = held;
                break taken != null and within(flag, @field(o, flag.field));
            }
        } else {
            refusal.* = .{ .unknown = arg };
            return null;
        };
        if (!ok) {
            refusal.* = .{ .untaken = .{ .flag = arg, .text = args[i], .quoted = words.quoted } };
            return null;
        }
    }
    return o;
}

/// `text` as a `T`: a string as it is, a number parsed, an enum by its tag's name or by its own `parse`.
fn as(comptime T: type, text: []const u8) ?T {
    const Inner = switch (@typeInfo(T)) {
        .optional => |optional| optional.child,
        else => T,
    };
    if (Inner == []const u8) return text;
    // read as the field's own type first: a null here is a refusal, and must not become an optional field's
    // null value
    const held: ?Inner = switch (@typeInfo(Inner)) {
        .int => std.fmt.parseInt(Inner, text, 10) catch null,
        .float => std.fmt.parseFloat(Inner, text) catch null,
        .@"enum" => if (@hasDecl(Inner, "parse")) Inner.parse(text) else std.meta.stringToEnum(Inner, text),
        else => @compileError("a flag cannot set a " ++ @typeName(T)),
    };
    if (held) |found| return @as(T, found);
    return null;
}

fn within(comptime flag: Flag, held: anytype) bool {
    if (flag.min == null and flag.max == null) return true;
    const number: f64 = switch (@typeInfo(@TypeOf(held))) {
        .optional => return held == null or within(flag, held.?),
        .int => @floatFromInt(held),
        else => @floatCast(held),
    };
    if (flag.min) |least| if (!(number >= least)) return false;
    if (flag.max) |most| if (!(number <= most)) return false;
    return true;
}

/// A tokenizer and chat template loaded from `dir`, or null with the reason logged under `path`. `shared` is
/// `Frontend.load`'s: a tokenizer already read from `dir`, borrowed.
pub fn loadFrontend(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, shared: ?*const text_frontend.Loaded) ?text_frontend.Frontend {
    return loadFrontendWith(gpa, io, dir, path, shared, null);
}

/// `loadFrontend` with `own` the template where it is not null (`Frontend.loadWith`).
pub fn loadFrontendWith(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, shared: ?*const text_frontend.Loaded, own: ?[]const u8) ?text_frontend.Frontend {
    var diagnostic: text_frontend.Diagnostic = .{};
    return text_frontend.Frontend.loadWith(gpa, io, dir, shared, own, &diagnostic) catch |err| {
        std.log.err("{s}: {s}", .{ path, if (diagnostic.len > 0) diagnostic.message() else @errorName(err) });
        return null;
    };
}

/// A checkpoint directory opened in the order every command opens one: the directory, its config, its tensor
/// files, and where the model's weights are among them. A model gives its own `Config` (with `read`), `Layout`
/// (with `init`) and `Diagnostic`. The value is filled in place, because the layout is given pointers to the
/// config and the checkpoint beside it. A step that returns false has logged why, and the command ends with 1.
pub fn Opened(comptime Config: type, comptime Layout: type, comptime Diagnostic: type) type {
    return struct {
        path: []const u8 = "",
        dir: ?std.Io.Dir = null,
        config: Config = undefined,
        checkpoint: ?safetensors.Checkpoint = null,
        layout: Layout = undefined,
        diagnostic: Diagnostic = .{},

        const Self = @This();

        /// What a directory that does not open does: `raise` returns the error to `main`, `say` logs a line.
        /// The commands differ here, and each keeps its own.
        pub const Missing = enum { raise, say };

        /// All three steps, for a command that does nothing between them.
        pub fn open(self: *Self, init: std.process.Init, path: []const u8, missing: Missing) !bool {
            return try self.enter(init.io, path, missing) and try self.configure(init) and try self.resolve(init);
        }

        pub fn enter(self: *Self, io: std.Io, path: []const u8, missing: Missing) !bool {
            self.path = path;
            self.dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| switch (missing) {
                .raise => return err,
                .say => {
                    std.log.err("{s}: {t}", .{ path, err });
                    return false;
                },
            };
            return true;
        }

        pub fn configure(self: *Self, init: std.process.Init) !bool {
            self.config = Config.read(init.arena.allocator(), init.io, self.dir.?, &self.diagnostic) catch |err| return self.refused(err);
            return true;
        }

        pub fn resolve(self: *Self, init: std.process.Init) !bool {
            self.checkpoint = try .open(init.gpa, init.io, .cwd(), self.path);
            self.layout = Layout.init(init.arena.allocator(), &self.config, &self.checkpoint.?, &self.diagnostic) catch |err| return self.refused(err);
            return true;
        }

        /// `open`'s three steps from another machine's: its `config.json` text and the checkpoint of its files'
        /// headers (`safetensors.Checkpoint.fromHeads`), which this takes. Nothing is opened under `path`.
        pub fn given(self: *Self, init: std.process.Init, path: []const u8, config: []const u8, checkpoint: safetensors.Checkpoint) !bool {
            self.path = path;
            self.checkpoint = checkpoint;
            self.config = Config.parse(init.arena.allocator(), config, &self.diagnostic) catch |err| return self.refused(err);
            self.layout = Layout.init(init.arena.allocator(), &self.config, &self.checkpoint.?, &self.diagnostic) catch |err| return self.refused(err);
            return true;
        }

        fn refused(self: *const Self, err: anyerror) bool {
            std.log.err("{s}: {s}", .{ self.path, if (self.diagnostic.message.len > 0) self.diagnostic.message else @errorName(err) });
            return false;
        }

        pub fn close(self: *Self, io: std.Io) void {
            if (self.checkpoint) |*checkpoint| checkpoint.close();
            if (self.dir) |dir| dir.close(io);
        }
    };
}

/// What a `chat` command was asked, before it is a prompt: the words of the request and the sampling flags
/// that override the checkpoint's generation config.
pub const ChatAsk = struct {
    system: ?[]const u8 = null,
    message: ?[]const u8 = null,
    /// A conversation in a file (a JSON array), in `system` and `message`'s place, and the tools' file.
    messages_file: ?[]const u8 = null,
    tools_file: ?[]const u8 = null,
    thinking: bool = true,
    effort: ?text_messages.Effort = null,
    /// Print the prompt's text and ids and stop.
    render: bool = false,
    seed: ?i64 = null,
    temperature: ?f64 = null,
    top_k: ?u32 = null,
    top_p: ?f64 = null,
    min_p: ?f64 = null,
};

/// What `chatPrompt` made: the prompt's ids with the sampling it is decoded by, or the exit code of a command
/// that is over (`--render`, a refusal; each said).
pub const ChatPrompt = union(enum) {
    ready: struct { ids: []i32, settings: sampling.Sampling },
    exit: u8,
};

/// The messages (or the file's) through the checkpoint's chat template and the tokenizer, `--render` printed
/// to `stdout`, and the sampling: the generation config's, the seed asked or derived from the prompt, and the
/// overrides asked, validated. `label` starts the line of a sampling refusal.
pub fn chatPrompt(init: std.process.Init, stdout: *std.Io.Writer, frontend: *text_frontend.Frontend, ask: ChatAsk, label: []const u8) !ChatPrompt {
    const arena = init.arena.allocator();
    const messages: std.json.Value = if (ask.messages_file) |file| readJson(init, file) catch return .{ .exit = 1 } else messages: {
        var list: std.json.Array = .init(arena);
        for ([_]?[]const u8{ ask.system, ask.message }, [_][]const u8{ "system", "user" }) |content, role| {
            var message: std.json.ObjectMap = .empty;
            try message.put(arena, "role", .{ .string = role });
            try message.put(arena, "content", .{ .string = content orelse continue });
            try list.append(.{ .object = message });
        }
        break :messages .{ .array = list };
    };
    const request: text_frontend.Request = .{
        .messages = messages,
        .tools = if (ask.tools_file) |file| readJson(init, file) catch return .{ .exit = 1 } else null,
        .thinking = ask.thinking,
        .effort = ask.effort,
    };
    var said: text_frontend.Diagnostic = .{};
    const text = frontend.render(arena, request, &said) catch |err| {
        std.log.err("{s}", .{if (said.len > 0) said.message() else @errorName(err)});
        return .{ .exit = 1 };
    };
    var encoded: std.ArrayList(u32) = .empty;
    try frontend.tokenizer.encode(arena, text, &encoded);
    const tokens = try arena.alloc(i32, encoded.items.len);
    for (tokens, encoded.items) |*token, id| token.* = @intCast(id);
    if (ask.render) {
        try stdout.print("{s}\n\n{d} ids:", .{ text, tokens.len });
        for (tokens) |id| try stdout.print(" {d}", .{id});
        try stdout.writeAll("\n");
        return .{ .exit = 0 };
    }
    var settings = sampling.Sampling.fromGenerationConfig(frontend.generation, if (ask.seed) |seed| sampling.seedFromInt(seed) else sampling.seedFor(tokens, 0));
    if (ask.temperature) |v| settings.temperature = v;
    if (ask.top_k) |v| settings.top_k = v;
    if (ask.top_p) |v| settings.top_p = v;
    if (ask.min_p) |v| settings.min_p = v;
    settings.validate() catch |err| {
        std.log.err("{s}: {t}", .{ label, err });
        return .{ .exit = 2 };
    };
    return .{ .ready = .{ .ids = tokens, .settings = settings } };
}

fn readJson(init: std.process.Init, path: []const u8) !std.json.Value {
    const arena = init.arena.allocator();
    const text = std.Io.Dir.cwd().readFileAlloc(init.io, path, arena, .limited(1 << 28)) catch |err| {
        std.log.err("{s}: {t}", .{ path, err });
        return err;
    };
    return std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch |err| {
        std.log.err("{s}: {t}", .{ path, err });
        return err;
    };
}

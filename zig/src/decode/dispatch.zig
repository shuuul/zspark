//! The commands by model: what a model registers (`Model`, a table of its `Command`s under the `model_type`s
//! its checkpoints carry) and how `main.zig` finds the one to run for a checkpoint directory, by its
//! `config.json`'s `model_type`. Host only, and it names no model: a model's table is a file of its own
//! (`models/qwen4/commands.zig`), listed in `main.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// One command of a model: `zspark NAME DIR ..`.
pub const Command = struct {
    name: []const u8,
    /// The words after the name that it needs, the checkpoint directory first; with fewer it is not run.
    least: usize,
    /// What is printed then, where the command has a usage of its own (`chat`); null: `zspark`'s usage.
    usage: ?[]const u8 = null,
    /// `words` are those after the name: `words[0]` is the directory. Returns the process's exit code.
    run: *const fn (init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) anyerror!u8,
    /// A flag under which the command reads nothing of its directory (GLM's `serve --weights-from-peer`, whose
    /// rank 1 is given its model by rank 0): a line that holds it is this command's, its directory unread.
    dirless: ?[]const u8 = null,
};

/// `run`, and `usage` after it when it returns 2 (a word it does not take): `ranks-check`'s, whose usage is
/// its words.
pub fn usageOnTwo(comptime run: anytype, comptime usage: []const u8) @FieldType(Command, "run") {
    return struct {
        fn call(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) anyerror!u8 {
            const code = try run(init, stdout, words);
            if (code == 2) try stdout.writeAll(usage);
            return code;
        }
    }.call;
}

/// A model as the command line sees it.
pub const Model = struct {
    /// How a message names it: "Qwen3.8 Flash Next".
    name: []const u8,
    /// Every value of `config.json`'s `model_type` that is this model.
    model_types: []const []const u8,
    /// The commands it has. A model need not have them all.
    commands: []const Command,

    pub fn command(self: Model, name: []const u8) ?Command {
        for (self.commands) |each| if (std.mem.eql(u8, each.name, name)) return each;
        return null;
    }

    pub fn is(self: Model, model_type: []const u8) bool {
        for (self.model_types) |own| if (std.mem.eql(u8, own, model_type)) return true;
        return false;
    }
};

/// What `pick` found for a command's name and a directory's `model_type`.
pub const Pick = union(enum) {
    /// No model has a command of this name: it is not one of `zspark`'s.
    none,
    /// The command to run.
    command: Command,
    /// The directory is a model none of the table is; `known` says which are.
    unknown: []const u8,
    /// The directory's model has no such command.
    lacking: Model,
};

/// The command `name` of the model whose `model_type` is given. Without a `model_type` (no directory was
/// named, or its `config.json` could not be read as far as that) it is the first model's that has the
/// command, which then says what is wrong with the directory as it always did.
pub fn pick(models: []const Model, name: []const u8, model_type: ?[]const u8) Pick {
    var first: ?Command = null;
    for (models) |model| {
        if (model.command(name)) |found| {
            first = found;
            break;
        }
    }
    if (first == null) return .none;
    const found = model_type orelse return .{ .command = first.? };
    for (models) |model| {
        if (model.is(found)) return if (model.command(name)) |held| .{ .command = held } else .{ .lacking = model };
    }
    return .{ .unknown = found };
}

/// The `model_type` of the model whose command `name` has a `dirless` flag that `words` hold, or null: the
/// directory is then not read to find it.
pub fn dirless(models: []const Model, name: []const u8, words: []const []const u8) ?[]const u8 {
    for (models) |model| {
        const flag = (model.command(name) orelse continue).dirless orelse continue;
        for (words) |word| if (std.mem.eql(u8, word, flag)) return model.model_types[0];
    }
    return null;
}

/// `dir/path/config.json`'s `model_type`, or null where the directory, the file, its JSON or the string is
/// not there: the command of `pick`'s default then reports it.
pub fn modelType(arena: Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) ?[]const u8 {
    var opened = dir.openDir(io, path, .{}) catch return null;
    defer opened.close(io);
    const text = opened.readFileAlloc(io, "config.json", arena, .limited(1 << 26)) catch return null;
    const document = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch return null;
    if (document != .object) return null;
    const found = document.object.get("model_type") orelse return null;
    return if (found == .string) found.string else null;
}

/// The models of the table and their `model_type`s: "Qwen3.8 Flash Next (qwen4_exp or qwen3_8_flash_next)",
/// one after the other with ", " between.
pub fn known(arena: Allocator, models: []const Model) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (models, 0..) |model, i| {
        if (i > 0) try out.appendSlice(arena, ", ");
        try out.appendSlice(arena, model.name);
        try out.appendSlice(arena, " (");
        for (model.model_types, 0..) |model_type, j| {
            if (j > 0) try out.appendSlice(arena, " or ");
            try out.appendSlice(arena, model_type);
        }
        try out.append(arena, ')');
    }
    return out.items;
}

/// Why a directory of an unknown `model_type` is refused: what was found and what is known. With one model
/// in the table it is the line that model's own config reader gave.
pub fn unknownText(arena: Allocator, models: []const Model, model_type: []const u8) Allocator.Error![]const u8 {
    return std.fmt.allocPrint(arena, "config.json describes model_type \"{s}\", not {s}", .{ model_type, try known(arena, models) });
}

/// Why a command a model lacks is refused: the model, the command, and the commands it has.
pub fn lackingText(arena: Allocator, model: Model, model_type: []const u8, name: []const u8) Allocator.Error![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.print(arena, "{s} (model_type \"{s}\") has no command `{s}`; its commands:", .{ model.name, model_type, name });
    for (model.commands) |each| try out.print(arena, " {s}", .{each.name});
    return out.items;
}

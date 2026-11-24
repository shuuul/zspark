//! What a generic of the shared code asks of the namespace a model gives it, held when the generic is
//! instantiated: a model that lacks a declaration is told all it lacks in one error that names the contract,
//! not at the first call that happens to be compiled. `draft.Decoder` and `server/served.zig`'s `Served`
//! state theirs with it; what the declarations must do stays in their doc comments, which no compiler reads.
//! Host only.

const std = @import("std");

/// One declaration a namespace owes.
pub const Owed = struct {
    name: []const u8,
    is: union(enum) {
        /// A type (a struct, an error set), with these fields and these declarations of its own.
        type: struct { fields: []const []const u8 = &.{}, decls: []const []const u8 = &.{} },
        /// A function of this many parameters.
        function: usize,
    },
};

/// What `M` lacks of `owed`, a line each; empty when it holds.
pub fn lacking(comptime M: type, comptime owed: []const Owed) []const u8 {
    comptime {
        var out: []const u8 = "";
        for (owed) |one| {
            if (!@hasDecl(M, one.name)) {
                out = out ++ "\n  " ++ one.name ++ ": not declared";
                continue;
            }
            const found = @field(M, one.name);
            switch (one.is) {
                .type => |parts| {
                    if (@TypeOf(found) != type) {
                        out = out ++ "\n  " ++ one.name ++ ": not a type";
                        continue;
                    }
                    const container = switch (@typeInfo(found)) {
                        .@"struct", .@"union", .@"enum", .@"opaque" => true,
                        else => false,
                    };
                    for (parts.fields) |field| {
                        if (!container or !@hasField(found, field)) out = out ++ "\n  " ++ one.name ++ ": no field `" ++ field ++ "`";
                    }
                    for (parts.decls) |decl| {
                        if (!container or !@hasDecl(found, decl)) out = out ++ "\n  " ++ one.name ++ ": no declaration `" ++ decl ++ "`";
                    }
                },
                .function => |count| switch (@typeInfo(@TypeOf(found))) {
                    .@"fn" => |function| if (function.param_types.len != count) {
                        out = out ++ std.fmt.comptimePrint("\n  {s}: a function of {d} parameters, not of {d}", .{ one.name, function.param_types.len, count });
                    },
                    else => out = out ++ "\n  " ++ one.name ++ ": not a function",
                },
            }
        }
        return out;
    }
}

/// A compile error unless `M` declares `owed`; `asked_by` names the generic and where its contract is written.
pub fn hold(comptime asked_by: []const u8, comptime M: type, comptime owed: []const Owed) void {
    comptime {
        const missing = lacking(M, owed);
        if (missing.len != 0) @compileError(@typeName(M) ++ " is not what " ++ asked_by ++ " asks of a model:" ++ missing);
    }
}

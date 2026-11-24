//! What a model with bf16 weights reads of its checkpoint before it loads (spec 142 WS-04: Qwen3-ASR's and
//! Unlimited-OCR's `config.zig` and `layout.zig` held it alike, moved here): why it was refused, in the model's own
//! words (`Diagnostic`), config.json's fields (`Parser`, each refusal naming the object it read), and a tensor by
//! name, held to its bf16 shape (`Finder`). Host only.

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;
const safetensors = @import("safetensors.zig");

/// Why a config was refused, in the model's own words.
pub const Diagnostic = struct {
    message: []const u8 = "",

    pub fn set(self: *Diagnostic, arena: Allocator, comptime format: []const u8, args: anytype) void {
        self.message = std.fmt.allocPrint(arena, format, args) catch "out of memory";
    }
};

pub const Error = error{ InvalidConfig, UnsupportedModel } || Allocator.Error;

pub const Parser = struct {
    arena: Allocator,
    diagnostic: *Diagnostic,
    /// What a missing or mistyped field's message names: "config.json", or the object a model reads its fields
    /// from ("language_config").
    where: []const u8 = "config.json",

    pub fn fail(p: Parser, err: Error, comptime format: []const u8, args: anytype) Error {
        p.diagnostic.set(p.arena, format, args);
        return err;
    }

    pub fn object(p: Parser, fields: json.ObjectMap, key: []const u8) Error!json.ObjectMap {
        const value = fields.get(key) orelse return p.fail(error.InvalidConfig, "{s} has no {s}", .{ p.where, key });
        if (value != .object) return p.fail(error.InvalidConfig, "{s}: {s} is not an object", .{ p.where, key });
        return value.object;
    }

    pub fn uint(p: Parser, fields: json.ObjectMap, key: []const u8) Error!usize {
        const value = fields.get(key) orelse return p.fail(error.InvalidConfig, "{s} has no {s}", .{ p.where, key });
        if (value == .integer and value.integer >= 0) return @intCast(value.integer);
        return p.fail(error.InvalidConfig, "{s}: {s} is not a count", .{ p.where, key });
    }

    pub fn numberOr(_: Parser, fields: json.ObjectMap, key: []const u8, default: f64) f64 {
        return switch (fields.get(key) orelse return default) {
            .integer => |v| @floatFromInt(v),
            .float => |v| v,
            else => default,
        };
    }

    pub fn flagOr(_: Parser, fields: json.ObjectMap, key: []const u8, default: bool) bool {
        const value = fields.get(key) orelse return default;
        return if (value == .bool) value.bool else default;
    }

    pub fn stringOr(_: Parser, fields: json.ObjectMap, key: []const u8, default: []const u8) []const u8 {
        const value = fields.get(key) orelse return default;
        return if (value == .string) value.string else default;
    }
};

pub const FindError = error{ MissingTensor, UnexpectedShape } || Allocator.Error;

/// A checkpoint's tensors by name, each bf16 of the shape asked.
pub const Finder = struct {
    arena: Allocator,
    checkpoint: *const safetensors.Checkpoint,
    diagnostic: *Diagnostic,

    pub fn get(f: Finder, shape: []const u64, comptime format: []const u8, args: anytype) FindError!safetensors.Tensor {
        const name = try std.fmt.allocPrint(f.arena, format, args);
        const tensor = f.checkpoint.get(name) orelse {
            f.diagnostic.set(f.arena, "the checkpoint has no {s}", .{name});
            return error.MissingTensor;
        };
        if (tensor.dtype != .BF16 or !std.mem.eql(u64, tensor.shape, shape)) {
            f.diagnostic.set(f.arena, "{s} is {t} {any}, not BF16 {any}", .{ name, tensor.dtype, tensor.shape, shape });
            return error.UnexpectedShape;
        }
        return tensor;
    }
};

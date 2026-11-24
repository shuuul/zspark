//! `template.zig`'s tests: jinja2's renderings of each construct and of the checkpoints' templates, which
//! TensorFold's engine must give through `Template` as zspark's own subset gave them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tpl = @import("../../src/text/template.zig");
const Diagnostic = tpl.Diagnostic;
const Template = tpl.Template;
const among = tpl.among;

// ----------------------------------------------------------------------------------------------------------------
// Tests
//
// The expectations in `zig/tests/fixtures/text/` are jinja2's, written by `tools/reference/template_cases.py` through
// TensorFold's own `ChatTemplate`.

const testing = std.testing;

const Outcome = enum { same, different };

/// Renders one fixture case and compares it with jinja2's result: the same bytes, or a failure of the same kind
/// (`TemplateError` is `raise_exception`, whose message must match too).
fn checkCase(gpa: Allocator, name: []const u8, source: []const u8, case: std.json.ObjectMap) !Outcome {
    const expected_error: ?std.json.ObjectMap = if (case.get("error")) |e| e.object else null;
    const kind: []const u8 = if (expected_error) |e| e.get("type").?.string else "";
    const refused_by_jinja = among(&.{ "TemplateSyntaxError", "TemplateAssertionError", "SyntaxError" }, kind);
    var diag: Diagnostic = .{};
    var template = Template.parse(gpa, source, &diag) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.TemplateSyntax => {
            if (refused_by_jinja) return .same;
            std.debug.print("{s}: parse failed: {s}\n", .{ name, diag.message() });
            return .different;
        },
    };
    defer template.deinit();
    if (refused_by_jinja) {
        std.debug.print("{s}: parsed, but jinja2 refuses the template\n", .{name});
        return .different;
    }
    const rendered = template.renderAlloc(gpa, case.get("context").?, &diag) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.TemplateRaised, error.TemplateRuntime => {
            const e = expected_error orelse {
                std.debug.print("{s}: {t}: {s}\n", .{ name, err, diag.message() });
                return .different;
            };
            const raised = std.mem.eql(u8, kind, "TemplateError");
            if (raised != (err == error.TemplateRaised) or (raised and !std.mem.eql(u8, diag.message(), e.get("message").?.string))) {
                std.debug.print("{s}: {t} '{s}', expected {s} '{s}'\n", .{ name, err, diag.message(), kind, e.get("message").?.string });
                return .different;
            }
            return .same;
        },
    };
    defer gpa.free(rendered);
    if (expected_error) |e| {
        std.debug.print("{s}: rendered, expected {s} '{s}'\n", .{ name, kind, e.get("message").?.string });
        return .different;
    }
    const expected = case.get("rendered").?.string;
    if (std.mem.eql(u8, rendered, expected)) return .same;
    const at = std.mem.indexOfDiff(u8, rendered, expected).?;
    const from = at -| 40;
    std.debug.print("{s}: differs at byte {d}\n  jinja2: {f}\n  zspark: {f}\n", .{
        name,                                                           at,
        std.zig.fmtString(expected[from..@min(expected.len, at + 40)]), std.zig.fmtString(rendered[from..@min(rendered.len, at + 40)]),
    });
    return .different;
}

test "each construct renders as jinja2 renders it" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("fixtures/text/template_unit_cases.json"), .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("cases").?.array.items;
    try testing.expect(cases.len >= 160);
    var different: usize = 0;
    for (cases) |case| {
        const c = case.object;
        // The one construct TensorFold's engine renders another way than jinja2: `d|items` kept in a variable
        // is a generator there, empty the second time it is looped over, and a list here. No checkpoint's
        // template keeps one.
        if (std.mem.eql(u8, c.get("name").?.string, "filter_items_is_a_generator")) continue;
        if (try checkCase(gpa, c.get("name").?.string, c.get("source").?.string, c) == .different) different += 1;
    }
    try testing.expectEqual(0, different);
}

/// The fixture's cases for one checkpoint's template, read from the directory `env` names.
/// Where TensorFold's engine departs from jinja2 and zspark follows it. A context with no `messages`: jinja2
/// loops over an undefined name as over nothing, the engine has no undefined and refuses `None`. GLM's
/// template loops over it bare (Flash Next's raises first); no request reaches it, `messages.zig` always
/// names the list.
const departures = [_][]const u8{"glm-5.3-flash/direct/messages_missing"};

fn checkCheckpoint(comptime env: []const u8, template_name: []const u8) !void {
    const path = testing.environ.getPosix(env) orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    const source = try dir.readFileAlloc(io, "chat_template.jinja", gpa, .limited(1 << 20));
    defer gpa.free(source);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("fixtures/text/template_cases.json"), .{});
    defer parsed.deinit();

    // The expectations are for one revision of the template.
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    for (parsed.value.object.get("templates").?.array.items) |t| {
        if (!std.mem.eql(u8, t.object.get("name").?.string, template_name)) continue;
        try testing.expectEqualStrings(t.object.get("sha256").?.string, &hex);
    }

    var checked: usize = 0;
    var different: usize = 0;
    for (parsed.value.object.get("cases").?.array.items) |case| {
        const c = case.object;
        if (!std.mem.eql(u8, c.get("template").?.string, template_name)) continue;
        if (among(&departures, c.get("name").?.string)) continue;
        checked += 1;
        if (try checkCase(gpa, c.get("name").?.string, source, c) == .different) different += 1;
    }
    try testing.expect(checked >= 90);
    try testing.expectEqual(0, different);
}

test "the Flash Next checkpoint's chat template renders as jinja2 renders it" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without.
    try checkCheckpoint("ZSPARK_FLASH_NEXT_FP4_LOCAL", "qwen3.8-flash-next");
}

test "the GLM-5.3-Flash checkpoint's chat template renders as jinja2 renders it" {
    // Runs with ZSPARK_GLM_FP4_LOCAL set to the checkpoint's directory; skipped without.
    try checkCheckpoint("ZSPARK_GLM_FP4_LOCAL", "glm-5.3-flash");
}

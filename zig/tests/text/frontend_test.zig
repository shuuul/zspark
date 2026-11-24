//! `frontend.zig`'s tests.

const std = @import("std");
const frontend_file = @import("../../src/text/frontend.zig");
const template = @import("../../src/text/template.zig");
const tokenizer = @import("../../src/text/tokenizer.zig");
const Io = std.Io;
const Diagnostic = frontend_file.Diagnostic;
const messages = @import("../../src/text/messages.zig");
const Frontend = frontend_file.Frontend;

const testing = std.testing;

test "the Flash Next checkpoint turns the primes request into the reference prompt's ids" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without.
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var dir = try Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var diagnostic: Diagnostic = .{};
    var frontend: Frontend = try .load(gpa, testing.io, dir, null, &diagnostic);
    defer frontend.deinit(gpa);
    try testing.expectEqual(messages.LateSystem.user, frontend.late);
    try testing.expect(frontend.levels.eql(.initMany(&.{ .low, .medium, .xhigh })));

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // The prompt of tools/reference's `primes` case, which TensorFold's reference tensors were exported from.
    const primes = [_]u32{ 248045, 846, 198, 826, 279, 1118, 4097, 9944, 4947, 11, 18101, 539, 73982, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
    var text: std.ArrayList(u8) = .empty;
    try frontend.tokenizer.decode(arena, primes[1..14], .{}, &text);
    try testing.expect(std.mem.startsWith(u8, text.items, "user\n"));
    var message: std.json.ObjectMap = .empty;
    try message.put(arena, "role", .{ .string = "user" });
    try message.put(arena, "content", .{ .string = text.items["user\n".len..] });
    var list: std.json.Array = .init(arena);
    try list.append(.{ .object = message });

    const off = try frontend.prompt(arena, .{ .messages = .{ .array = list }, .thinking = false }, &diagnostic);
    try testing.expectEqualSlices(u32, &primes, off);
    const none = try frontend.prompt(arena, .{ .messages = .{ .array = list }, .effort = .none }, &diagnostic);
    try testing.expectEqualSlices(u32, &primes, none);
    // With thinking on, the think block is left open for the model; medium is the effort without instructions.
    const on = try frontend.prompt(arena, .{ .messages = .{ .array = list }, .effort = .medium }, &diagnostic);
    try testing.expectEqualSlices(u32, primes[0..20], on[0..20]);
    try testing.expectEqualSlices(u32, &.{198}, on[20..]);
    // The template names low, medium and xhigh: high is heard as xhigh (as no effort is), minimal as low.
    const high = try frontend.render(arena, .{ .messages = .{ .array = list }, .effort = .high }, &diagnostic);
    try testing.expect(std.mem.indexOf(u8, high, "Reasoning effort is set to xhigh.") != null);
    try testing.expectEqualStrings(high, try frontend.render(arena, .{ .messages = .{ .array = list } }, &diagnostic));
    const minimal = try frontend.render(arena, .{ .messages = .{ .array = list }, .effort = .minimal }, &diagnostic);
    try testing.expect(std.mem.indexOf(u8, minimal, "Reasoning effort is set to low.") != null);

    try testing.expectError(error.BadRequest, frontend.prompt(arena, .{ .messages = .{ .array = .init(arena) } }, &diagnostic));
    try testing.expectEqualStrings("messages must be a non-empty list", diagnostic.message());

    // Every message opens with `<|im_start|>`: the user's and the reply's header here.
    try testing.expectEqualSlices(u32, &.{frontend.tokenizer.addedId("<|im_start|>").?}, frontend.openers);
    var signed: [primes.len]i32 = undefined;
    for (&signed, primes) |*to, id| to.* = @intCast(id);
    try testing.expectEqualSlices(usize, &.{ 0, 16 }, try frontend.starts(arena, &signed));

    // Between a reply's reasoning and its call this template writes a blank line (`'\n</think>\n\n' + content`,
    // the content empty, then the call's opening): what a required call after thinking is written with.
    try testing.expectEqualStrings("\n\n", frontend.call_lead);
    // And between a reply's reasoning and its content the same blank line: what a document is written with.
    try testing.expectEqualStrings("\n\n", frontend.content_lead);
}

test "what a template writes between a reply's reasoning and its call, or nothing" {
    const gpa = testing.allocator;
    // A template in Qwen's shape with `between` after the reasoning's close and `before` before a call's opening.
    const head = "{% for m in messages %}<|end|>{{ m.role }}\n{% if m.role == 'assistant' %}";
    const call = "{% for c in m.tool_calls %}<tool_call>\n<function={{ c.function.name }}>\n</function>\n</tool_call>{% endfor %}";
    const tail = "{% else %}{{ m.content }}{% endif %}<|end|>\n{% endfor %}<|end|>assistant\n<think>\n";
    const thought = "<think>\n{{ m.reasoning_content }}\n</think>";
    const Case = struct { source: []const u8, lead: []const u8 };
    for ([_]Case{
        // a blank line, as Qwen's; one newline; nothing
        .{ .source = head ++ thought ++ "\n\n{{ m.content }}" ++ call ++ tail, .lead = "\n\n" },
        .{ .source = head ++ thought ++ "\n{{ m.content }}" ++ call ++ tail, .lead = "\n" },
        .{ .source = head ++ thought ++ "{{ m.content }}" ++ call ++ tail, .lead = "" },
        // three newlines, one of them written with the call
        .{ .source = head ++ thought ++ "\n\n{{ m.content }}{% if m.tool_calls %}{{ '\n' }}{% endif %}" ++ call ++ tail, .lead = "\n\n\n" },
        // text that is not a newline: the decoder would send it to the client as content, so it is no lead
        .{ .source = head ++ thought ++ " \n{{ m.content }}" ++ call ++ tail, .lead = "" },
        .{ .source = head ++ thought ++ "\n\ncalls:\n{{ m.content }}" ++ call ++ tail, .lead = "" },
        // the reasoning left out of the history, its tags too; and written without its closing tag
        .{ .source = head ++ "{{ m.content }}" ++ call ++ tail, .lead = "" },
        .{ .source = head ++ "<think>\n{{ m.reasoning_content }}\n\n{{ m.content }}" ++ call ++ tail, .lead = "" },
        // the reasoning written after the call
        .{ .source = head ++ call ++ "\n\n" ++ thought ++ tail, .lead = "" },
        // no call written, or written under another tag
        .{ .source = head ++ thought ++ "\n\n{{ m.content }}" ++ tail, .lead = "" },
        .{ .source = head ++ thought ++ "\n\n{{ m.content }}{% for c in m.tool_calls %}<call>{{ c.function.name }}</call>{% endfor %}" ++ tail, .lead = "" },
        // the opening after the reasoning is not the call's own: an example stands between
        .{ .source = head ++ thought ++ "\n\n<tool_call>example</tool_call>\n" ++ call ++ tail, .lead = "" },
        // a template that refuses the probe: no tool message, and nothing at all
        .{ .source = "{% for m in messages %}{% if m.role == 'tool' %}{{ raise_exception('no tools') }}{% endif %}{% endfor %}" ++ head ++ thought ++ "\n\n" ++ call ++ tail, .lead = "" },
        .{ .source = "{{ raise_exception('no') }}", .lead = "" },
    }) |case| {
        errdefer std.debug.print("template: {s}\n", .{case.source});
        var template_diagnostic: template.Diagnostic = .{};
        const parsed = try template.Template.parse(gpa, case.source, &template_diagnostic);
        var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
        const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
        var frontend: Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
        defer frontend.deinit(gpa);
        try testing.expectEqualStrings(case.lead, try frontend.probeCallLead(frontend.arena.allocator()));
    }
}

test "what a template writes between a reply's reasoning and its content, or nothing" {
    const gpa = testing.allocator;
    const head = "{% for m in messages %}<|end|>{{ m.role }}\n{% if m.role == 'assistant' %}";
    const tail = "{% else %}{{ m.content }}{% endif %}<|end|>\n{% endfor %}<|end|>assistant\n<think>\n";
    const thought = "<think>\n{{ m.reasoning_content }}\n</think>";
    const Case = struct { source: []const u8, lead: []const u8 };
    for ([_]Case{
        // a blank line, as Qwen's; one newline; nothing
        .{ .source = head ++ thought ++ "\n\n{{ m.content }}" ++ tail, .lead = "\n\n" },
        .{ .source = head ++ thought ++ "\n{{ m.content }}" ++ tail, .lead = "\n" },
        .{ .source = head ++ thought ++ "{{ m.content }}" ++ tail, .lead = "" },
        // the content trimmed by the template, and a newline of the template's own before it
        .{ .source = head ++ thought ++ "\n\n{{ '\n' + m.content|trim }}" ++ tail, .lead = "\n\n\n" },
        // text that is not a newline would be the client's content: no lead
        .{ .source = head ++ thought ++ " {{ m.content }}" ++ tail, .lead = "" },
        .{ .source = head ++ thought ++ "\n\nAnswer: {{ m.content }}" ++ tail, .lead = "" },
        // the reasoning left out of the history, or written without its closing tag, or after the content
        .{ .source = head ++ "{{ m.content }}" ++ tail, .lead = "" },
        .{ .source = head ++ "<think>\n{{ m.reasoning_content }}\n\n{{ m.content }}" ++ tail, .lead = "" },
        .{ .source = head ++ "{{ m.content }}\n\n" ++ thought ++ tail, .lead = "" },
        // the reasoning kept only for the turn not yet answered: the probe's reply is answered, so none
        .{ .source = head ++ "{% if loop.last %}" ++ thought ++ "\n\n{% endif %}{{ m.content }}" ++ tail, .lead = "" },
        // the content left out, or written otherwise than it was sent (escaped)
        .{ .source = head ++ thought ++ "\n\n" ++ tail, .lead = "" },
        .{ .source = head ++ thought ++ "\n\n{{ m.content|tojson }}" ++ tail, .lead = "" },
        // a template that refuses the probe
        .{ .source = "{{ raise_exception('no') }}", .lead = "" },
    }) |case| {
        errdefer std.debug.print("template: {s}\n", .{case.source});
        var template_diagnostic: template.Diagnostic = .{};
        const parsed = try template.Template.parse(gpa, case.source, &template_diagnostic);
        var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
        const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
        var frontend: Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
        defer frontend.deinit(gpa);
        try testing.expectEqualStrings(case.lead, try frontend.probeContentLead(frontend.arena.allocator()));
    }
}

test "a template's opener is the special token each message adds once, or none" {
    const gpa = testing.allocator;
    const Case = struct { source: []const u8, opener: ?[]const u8 };
    for ([_]Case{
        // every message and the reply's header open with `<|end|>`
        .{ .source = "{% for message in messages %}<|end|>{{ message.role }} {{ message.content }}\n{% endfor %}<|end|>assistant ", .opener = "<|end|>" },
        // a system block before them does too
        .{ .source = "<|end|>system be brief\n{% for message in messages %}<|end|>{{ message.role }} {{ message.content }}\n{% endfor %}<|end|>assistant ", .opener = "<|end|>" },
        // the token closes a message as well: two a message, so it marks no start
        .{ .source = "{% for message in messages %}<|end|>{{ message.role }} {{ message.content }}<|end|>\n{% endfor %}<|end|>assistant ", .opener = null },
        // a first token that is not special
        .{ .source = "{% for message in messages %}{{ message.role }}: {{ message.content }}<|end|>\n{% endfor %}assistant: ", .opener = null },
        // `<t>` is an added token and not a special one
        .{ .source = "{% for message in messages %}<t>{{ message.role }} {{ message.content }}\n{% endfor %}<t>assistant ", .opener = null },
        // a template that refuses the probe
        .{ .source = "{{ raise_exception('no') }}", .opener = null },
    }) |case| {
        var template_diagnostic: template.Diagnostic = .{};
        const parsed = try template.Template.parse(gpa, case.source, &template_diagnostic);
        var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
        const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
        var frontend: Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
        defer frontend.deinit(gpa);
        const opener = try frontend.probeOpener(gpa);
        try testing.expectEqual(if (case.opener) |text| frontend.tokenizer.addedId(text).? else null, opener);
        if (opener) |one| frontend.openers = &.{one};
        const ids = [_]i32{ 281, 5, 281, 281, 7, 9 };
        const found = try frontend.starts(frontend.arena.allocator(), &ids);
        try testing.expectEqualSlices(usize, if (case.opener != null) &.{ 0, 2, 3 } else &.{}, found);
    }
}

test "where the roles open with tokens of their own, each is an opener, and a message starts at either" {
    const gpa = testing.allocator;
    const Case = struct { source: []const u8, openers: []const []const u8 };
    for ([_]Case{
        // GLM's form: a header before every message, `<|end|>` the user's and `<l>` the assistant's and the reply's
        .{ .source = "<l>be brief{% for message in messages %}{% if message.role == 'user' %}<|end|>{% else %}<l>{% endif %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}<l>{% endif %}", .openers = &.{ "<|end|>", "<l>" } },
        // the assistant's opens with an added token that is not special: the user's alone
        .{ .source = "{% for message in messages %}{% if message.role == 'user' %}<|end|>{% else %}<t>{% endif %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}<t>{% endif %}", .openers = &.{"<|end|>"} },
        // the reply's header is not the assistant's message's: no opener for that role
        .{ .source = "{% for message in messages %}{% if message.role == 'user' %}<|end|>{% else %}<l>{% endif %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}<|end|>{% endif %}", .openers = &.{"<|end|>"} },
    }) |case| {
        errdefer std.debug.print("template: {s}\n", .{case.source});
        var template_diagnostic: template.Diagnostic = .{};
        const parsed = try template.Template.parse(gpa, case.source, &template_diagnostic);
        var tokenizer_diagnostic: tokenizer.Diagnostic = .{};
        const tokens = try tokenizer.Tokenizer.parse(gpa, @embedFile("fixtures/text/tiny_tokenizer.json"), &tokenizer_diagnostic);
        var frontend: Frontend = .{ .arena = .init(gpa), .tokenizer = tokens, .template = parsed, .specials = .empty, .late = .user, .levels = .initMany(&.{}), .generation = null };
        defer frontend.deinit(gpa);
        const arena = frontend.arena.allocator();
        try testing.expectEqual(@as(?u32, null), try frontend.probeOpener(gpa));
        frontend.openers = try frontend.probeOpeners(arena);
        var want: [2]u32 = undefined;
        for (want[0..case.openers.len], case.openers) |*id, text| id.* = frontend.tokenizer.addedId(text).?;
        try testing.expectEqualSlices(u32, want[0..case.openers.len], frontend.openers);
        const ids = [_]i32{ 281, 5, 282, 282, 7, 283, 9 };
        const found = try frontend.starts(arena, &ids);
        try testing.expectEqualSlices(usize, if (case.openers.len == 2) &.{ 0, 2, 3 } else &.{0}, found);
    }
}

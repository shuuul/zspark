//! Spec 017: xgrammar's masks, written down for the tests of `zig/src/text/json_schema.zig`.
//!
//!     grammar_masks TOKENIZER_JSON tiny|checkpoint XGRAMMAR_COMMIT > FIXTURE
//!
//! `grammar_masks.sh` builds this against an xgrammar checkout and runs it; xgrammar is never part of the
//! repository or of zspark. For every schema of `corpus` it compiles the schema with xgrammar (as TensorFold
//! does: strict, any whitespace up to 32 blanks between tokens) and walks token sequences through xgrammar's
//! matcher, writing the mask xgrammar fills before each token (and the tokens that mask lacks though xgrammar's
//! own `AcceptToken` takes them: see `Run.fill`). The sequences are the corpus' documents cut into
//! tokens three ways, random walks, and documents xgrammar takes that zspark must refuse.
//!
//! The Zig matcher is used here for two things: a random walk draws among the tokens both engines take, so that
//! it stays where the two can be compared; and it names the tokens put to xgrammar's acceptor when xgrammar's
//! mask lacks them. What is allowed is xgrammar's word alone; the comparison, and which differences are meant,
//! is the test's. `GRAMMAR_MASKS_SHOW=1` prints the first differing tokens of each schema.
//!
//! The file, line by line:
//!
//!     # ...                         a comment
//!     xgrammar COMMIT whitespace N  the oracle and its blank limit
//!     vocabulary NAME COUNT WIDTH STOP...   the tokenizer's ids, the mask's width, the stop ids
//!     schema NAME [narrowed]        and on the next line its text; `narrowed` when zspark reads the schema
//!                                   more narrowly than xgrammar by intent
//!     walk KIND ENDED               a token sequence; ENDED 1 when it ran to a stop
//!     t ID...                       its tokens
//!     m INDEX...                    the mask before each token (none for `special`)
//!     mask HEX                      (tiny) mask INDEX, in order of first use: the words, eight digits each
//!     mask COUNT ALL PLAIN          (checkpoint) the bits set, and two FNV-1a hashes: of the words, and of the
//!                                   words with only the tokens that hold neither byte 0xED nor a backslash
//!
//! KIND is `document` (the tokenizer's own cut), `bytes` (the shortest token each time), `cut` (a random token
//! each time), `random` (a walk), `only` (xgrammar takes all of it and zspark must refuse a token of it) or
//! `special` (a special token inside a string, which xgrammar takes when it is handed the token's text).

const std = @import("std");
const schema = @import("schema");

const Io = std.Io;
const Allocator = std.mem.Allocator;

const XgTokenizer = opaque {};
const XgCompiler = opaque {};
const XgGrammar = opaque {};
const XgMatcher = opaque {};
extern fn zxg_tokenizer_new(bytes: [*]const u8, starts: [*]const u32, count: i32, vocab_size: i32, stop_ids: [*]const i32, stops: i32, err: [*]u8, err_len: usize) ?*XgTokenizer;
extern fn zxg_compiler_new(tokenizer: *const XgTokenizer, max_threads: i32, cache_bytes: i64, err: [*]u8, err_len: usize) ?*XgCompiler;
extern fn zxg_compile_json_schema(compiler: *XgCompiler, schema: [*]const u8, len: usize, max_whitespace: i32, err: [*]u8, err_len: usize) ?*XgGrammar;
extern fn zxg_grammar_free(grammar: *XgGrammar) void;
extern fn zxg_matcher_new(grammar: *const XgGrammar, err: [*]u8, err_len: usize) ?*XgMatcher;
extern fn zxg_matcher_free(matcher: *XgMatcher) void;
extern fn zxg_fill(matcher: *XgMatcher, words: [*]u32, count: i32, err: [*]u8, err_len: usize) i32;
extern fn zxg_accept(matcher: *XgMatcher, token: i32, err: [*]u8, err_len: usize) i32;
extern fn zxg_rollback(matcher: *XgMatcher, tokens: i32, err: [*]u8, err_len: usize) i32;
extern fn zxg_terminated(matcher: *const XgMatcher) i32;

var err: [2048]u8 = @splat(0);

fn said() []const u8 {
    return std.mem.sliceTo(&err, 0);
}

/// The blanks xgrammar is told to allow between two tokens of a document, which is the Zig matcher's default.
const whitespace = 32;

const Case = struct {
    name: []const u8,
    schema: []const u8,
    /// Valid documents.
    documents: []const []const u8,
    /// Documents xgrammar takes and zspark refuses, by intent.
    only: []const []const u8 = &.{},
    /// Whether zspark reads the schema more narrowly than xgrammar, by intent: its masks are then held to be
    /// within xgrammar's, not equal to them.
    narrowed: bool = false,
};

const flat =
    \\{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"score":{"type":"number"},"active":{"type":"boolean"},"note":{"type":"null"}},"required":["name","age","score","active"]}
;

/// As many of one blank as may stand between two tokens.
fn blanks(comptime blank: u8) [whitespace]u8 {
    return @splat(blank);
}

/// Each schema says in its name what it is there for. No text holds `Q`, `z` or a tab, which the tiny tokenizer
/// has no token for.
const corpus = [_]Case{
    .{
        .name = "flat",
        .schema = flat,
        .documents = &.{
            \\{"name": "Ada Lovelace", "age": 36, "score": 9.75, "active": true}
            ,
            \\{"name":"李雷 \"the\" tester\n","age":-7,"score":-1.5e-3,"active":false,"note":null}
            ,
            "{\n  \"name\": \"caf\\u00e9 \\\\ path/to \\ud83d\\ude00 \\uD7FF\\ue000\",\n  \"age\": 0,\n  \"score\": 12,\n  \"active\": true,\n  \"note\": null\n}",
        },
        .only = &.{
            // an escaped surrogate with no partner: a high one before a quote, before a letter, before another
            // escape; a low one alone
            \\{"name":"\ud83d","age":1,"score":1,"active":true}
            ,
            \\{"name":"\uD83Dx","age":1,"score":1,"active":true}
            ,
            \\{"name":"\ud83d\u0041","age":1,"score":1,"active":true}
            ,
            \\{"name":"a\udc00","age":1,"score":1,"active":true}
            ,
            \\{"name":"\ud83d\ud83d\ude00","age":1,"score":1,"active":true}
            ,
            // a surrogate encoded as UTF-8, which is not UTF-8
            "{\"name\":\"\xed\xa0\x80\",\"age\":1,\"score\":1,\"active\":true}",
            "{\"name\":\"a\xed\xbf\xbf\",\"age\":1,\"score\":1,\"active\":true}",
        },
    },
    .{
        .name = "whitespace",
        .schema = flat,
        .documents = &.{
            "{" ++ blanks(' ') ++ "\"name\"\r:\r\n\"\" ,\n\n\"age\" : 10\r,\"score\":-0.0E+10 , \"active\"\n:\nfalse\r\n}",
            "{\"name\":\"\"," ++ blanks('\n') ++ "\"age\":1,\"score\":1,\"active\":true" ++ blanks('\r') ++ "}",
            "{\n\"name\":\" \",\"age\":1,\"score\":1,\"active\":true\n}",
        },
    },
    .{
        .name = "nested",
        .schema =
        \\{"type":"object","properties":{"id":{"type":"integer"},"customer":{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"},"tier":{"enum":["free","pro","enterprise"]}},"required":["name","tier"]},"items":{"type":"array","items":{"type":"object","properties":{"sku":{"type":"string"},"qty":{"type":"integer"},"price":{"type":"number"},"tags":{"type":"array","items":{"type":"string"}},"status":{"type":"string","enum":["new","shipped","returned"]}},"required":["sku","qty","status"]}},"gift":{"type":"boolean"},"coupon":{"type":["string","null"]}},"required":["id","customer","items"]}
        ,
        .documents = &.{
            \\{"id": 1042, "customer": {"name": "Grace Hopper", "email": "grace@example.com", "tier": "pro"}, "items": [{"sku": "A-100", "qty": 2, "price": 19.99, "tags": ["blue", "large"], "status": "new"}, {"sku": "B-7", "qty": 1, "status": "shipped"}], "gift": false, "coupon": null}
            ,
            \\{"id":0,"customer":{"name":"","tier":"enterprise"},"items":[]}
            ,
            \\{"id":-3,"customer":{"name":"n","tier":"free"},"items":[{"sku":"s","qty":0,"tags":[],"status":"returned"}],"coupon":"TEN"}
        },
    },
    .{
        // a `get_weather` tool: nothing required
        .name = "tool-weather",
        .schema =
        \\{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"},"units":{"type":"string","enum":["c","f"]}}}
        ,
        .documents = &.{
            \\{}
            ,
            \\{"city":"Paris","days":3,"units":"c"}
            ,
            \\{"days":1}
            ,
            \\{ "units" : "f" }
        },
    },
    .{
        // the search tool of the same file
        .name = "tool-search",
        .schema =
        \\{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}}}
        ,
        .documents = &.{
            \\{"query":"rust 1.80","limit":5}
            ,
            \\{"limit":0}
        },
    },
    .{
        // and its tool whose parameters are named past ASCII
        .name = "tool-names-past-ascii",
        .schema =
        \\{"type":"object","properties":{"城市":{"type":"string"},"天":{"type":"integer"}}}
        ,
        .documents = &.{
            \\{"城市":"東京","天":3}
            ,
            \\{"天":0}
        },
    },
    .{
        // a tool as a client writes one: descriptions, a default, one required, closed
        .name = "tool-file",
        .schema =
        \\{"type":"object","properties":{"path":{"type":"string","description":"the file"},"lines":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2},"mode":{"type":"string","enum":["read","write","append"]},"force":{"type":"boolean","default":false}},"required":["path"],"additionalProperties":false}
        ,
        .documents = &.{
            \\{"path":"/tmp/a.txt","lines":[1,20],"mode":"append","force":true}
            ,
            \\{"path":""}
            ,
            \\{"path":"C:\\dir\\f","mode":"read"}
        },
    },
    .{
        // every property required and nothing else taken, as OpenAI's strict mode asks
        .name = "strict",
        .schema =
        \\{"type":"object","properties":{"id":{"type":"string"},"count":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}},"parent":{"anyOf":[{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false},{"type":"null"}]}},"required":["id","count","tags","parent"],"additionalProperties":false}
        ,
        .documents = &.{
            \\{"id":"a","count":1,"tags":[],"parent":null}
            ,
            \\{"id":"a","count":1,"tags":["x","y"],"parent":{"id":"b"}}
        },
    },
    .{
        .name = "open-object",
        .schema =
        \\{"type": "object"}
        ,
        .documents = &.{
            \\{}
            ,
            \\{"a":1,"b":[true,null,{"c":"d"}],"e":{"f":-1.5E+3}}
        },
    },
    .{
        .name = "anything",
        .schema =
        \\{}
        ,
        .documents = &.{
            \\null
            ,
            \\[1,"a",{"b":[]}]
            ,
            \\"x"
            ,
            \\-12.5e3
        },
    },
    .{
        // as Pydantic 2 writes a model
        .name = "pydantic",
        .schema =
        \\{"$defs":{"Address":{"properties":{"street":{"title":"Street","type":"string"},"post":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"title":"Post"}},"required":["street"],"title":"Address","type":"object"},"Kind":{"enum":["home","work"],"title":"Kind","type":"string"}},"properties":{"name":{"title":"Name","type":"string"},"age":{"anyOf":[{"type":"integer"},{"type":"null"}],"default":null,"title":"Age"},"address":{"$ref":"#/$defs/Address"},"kind":{"$ref":"#/$defs/Kind","default":"home"},"previous":{"anyOf":[{"$ref":"#/$defs/Address"},{"type":"null"}],"default":null},"scores":{"items":{"anyOf":[{"type":"number"},{"type":"integer"},{"type":"boolean"}]},"title":"Scores","type":"array"}},"required":["name","address"],"title":"Person","type":"object"}
        ,
        .documents = &.{
            \\{"name":"Ann","address":{"street":"1 Main St"}}
            ,
            \\{"name":"Bo","age":null,"address":{"street":"x","post":"12345"},"kind":"work","previous":{"street":"y","post":null},"scores":[1,2.5,true]}
            ,
            \\{"name":"Cy","age":41,"address":{"street":""},"kind":"home","previous":null,"scores":[]}
        },
    },
    .{
        // a schema that names itself through `definitions` (xgrammar does not read `~1` in a pointer)
        .name = "ref-tree",
        .schema =
        \\{"definitions":{"node":{"type":"object","properties":{"value":{"type":"integer"},"children":{"type":"array","items":{"$ref":"#/definitions/node"}},"next":{"oneOf":[{"$ref":"#/definitions/node"},{"type":"null"}]}},"required":["value"]}},"$ref":"#/definitions/node"}
        ,
        .documents = &.{
            \\{"value":1}
            ,
            \\{"value":1,"children":[{"value":2,"children":[]},{"value":3,"next":null}],"next":{"value":4}}
        },
    },
    .{
        // and through `#`
        .name = "ref-root",
        .schema =
        \\{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#"}}},"required":["v"]}
        ,
        .documents = &.{
            \\{"v":1,"kids":[{"v":2,"kids":[{"v":3}]},{"v":4}]}
            ,
            \\{"v":0}
        },
    },
    .{
        // two definitions whose names begin alike, each used
        .name = "ref-two",
        .schema =
        \\{"$defs":{"A":{"type":"integer"},"AB":{"type":"boolean"},"B":{"type":"array","items":{"$ref":"#/$defs/AB"}}},"type":"object","properties":{"a":{"$ref":"#/$defs/A"},"ab":{"$ref":"#/$defs/AB"},"b":{"$ref":"#/$defs/B"}},"required":["a","ab","b"]}
        ,
        .documents = &.{
            \\{"a":1,"ab":true,"b":[false,true]}
        },
    },
    .{
        .name = "any-of",
        .schema =
        \\{"anyOf":[{"type":"string"},{"type":"integer"},{"type":"null"},{"type":"array","items":{"type":"boolean"}},{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}]}
        ,
        .documents = &.{
            \\"s"
            ,
            \\-5
            ,
            \\null
            ,
            \\[true,false]
            ,
            \\{"a":1}
        },
    },
    .{
        .name = "one-of",
        .schema =
        \\{"type":"object","properties":{"shape":{"oneOf":[{"type":"object","properties":{"r":{"type":"number"}},"required":["r"]},{"type":"array","items":{"type":"number"},"minItems":2,"maxItems":2},{"type":"string","enum":["none"]}]}},"required":["shape"]}
        ,
        .documents = &.{
            \\{"shape":{"r":1.5}}
            ,
            \\{"shape":[1,2]}
            ,
            \\{"shape":"none"}
        },
    },
    .{
        .name = "deep",
        .schema =
        \\{"type":"object","properties":{"a":{"type":"array","items":{"type":"array","items":{"type":"object","properties":{"b":{"type":"array","items":{"type":"object","properties":{"c":{"type":"array","items":{"type":"boolean"},"maxItems":2}},"required":["c"]},"minItems":1,"maxItems":2}},"required":["b"]},"maxItems":2},"minItems":1,"maxItems":2}},"required":["a"]}
        ,
        .documents = &.{
            \\{"a":[[{"b":[{"c":[true,false]}]}]]}
            ,
            \\{"a":[[],[{"b":[{"c":[]},{"c":[true]}]},{"b":[{"c":[false]}]}]]}
        },
    },
    .{
        // empty objects and arrays, and the object that takes no key
        .name = "empty",
        .schema =
        \\{"type":"object","properties":{"none":{"type":"object","additionalProperties":false},"nothing":{"type":"array","maxItems":0},"open":{"type":"object"},"list":{"type":"array"},"props":{"type":"object","properties":{}}},"required":["none","nothing"]}
        ,
        .documents = &.{
            \\{"none":{},"nothing":[]}
            ,
            \\{"none":{ },"nothing":[ ],"open":{"k":[]},"list":[{}],"props":{}}
        },
    },
    .{
        // values that go on from one another, as strings, as numbers and across kinds
        .name = "enum-prefixes",
        .schema =
        \\{"type":"array","items":{"enum":["a","ab","abc","b","",1,12,123,1.5,-1,true,null,"true","nul"]},"minItems":1,"maxItems":6}
        ,
        .documents = &.{
            \\["a"]
            ,
            \\["ab","abc","b","",1,12]
            ,
            \\[123,1.5,-1,true,null,"true"]
            ,
            \\["nul", 12 ,"a"]
        },
    },
    .{
        // escapes and characters past ASCII in a value and in a property's name
        .name = "literals",
        .schema =
        \\{"type":"object","properties":{"Zo\u00eb":{"enum":["\u6771\u4eac","a\"b","a\\b","\n","/"]},"a\"b":{"type":"string"},"\u6771":{"const":"\u00eb"}},"required":["Zo\u00eb"]}
        ,
        .documents = &.{
            \\{"Zoë":"東京"}
            ,
            \\{"Zoë":"a\"b","a\"b":"x","東":"ë"}
            ,
            \\{"Zoë":"\n"}
            ,
            \\{"Zoë":"a\\b"}
            ,
            \\{"Zoë":"/"}
        },
    },
    .{
        .name = "const",
        .schema =
        \\{"type":"object","properties":{"kind":{"const":"point"},"n":{"const":3},"ok":{"const":true},"none":{"const":null}},"required":["kind"]}
        ,
        .documents = &.{
            \\{"kind":"point"}
            ,
            \\{"kind":"point","n":3,"ok":true,"none":null}
        },
    },
    .{
        // each kind of number alone and together
        .name = "numbers",
        .schema =
        \\{"type":"object","properties":{"i":{"type":"integer"},"n":{"type":"number"},"both":{"type":["integer","number"]},"list":{"type":"array","items":{"type":"number"},"minItems":1},"or":{"anyOf":[{"type":"integer"},{"type":"string"},{"type":"null"}]}},"required":["i","n"]}
        ,
        .documents = &.{
            \\{"i":0,"n":0}
            ,
            \\{"i":-12,"n":-0.5e+10,"both":3,"list":[1,2.5,-3E2],"or":"x"}
            ,
            \\{"i":7,"n":1E5,"both":2.5,"list":[0],"or":null}
            ,
            \\{"i":1,"n":2,"or":-9}
        },
    },
    .{
        .name = "number",
        .schema =
        \\{"type":"number"}
        ,
        .documents = &.{ "0", "-0", "10.01", "1e5", "-1.5E-7", "0.0", "120" },
    },
    .{
        .name = "integer",
        .schema =
        \\{"type":"integer"}
        ,
        .documents = &.{ "0", "-17", "1234567890" },
    },
    .{
        .name = "string",
        .schema =
        \\{"type":"string"}
        ,
        .documents = &.{
            \\""
            ,
            \\"a\tb\u00e9\/\b\f\r\\\""
            ,
            \\"東京 ë"
        },
    },
    .{
        .name = "type-list",
        .schema =
        \\{"type":["boolean","null","integer"]}
        ,
        .documents = &.{ "true", "false", "null", "42" },
    },
    .{
        .name = "items-2-to-3",
        .schema =
        \\{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":3}
        ,
        .documents = &.{
            \\["a","b"]
            ,
            \\["a","b","c"]
            ,
            \\[ "a" , "b" ]
        },
    },
    .{
        .name = "items-1-to-1",
        .schema =
        \\{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":1}
        ,
        .documents = &.{"[5]"},
    },
    .{
        .name = "items-from-2",
        .schema =
        \\{"type":"array","items":{"type":"integer"},"minItems":2}
        ,
        .documents = &.{ "[1,2]", "[1,2,3,4,5]" },
    },
    .{
        .name = "items-none",
        .schema =
        \\{"type":"array","maxItems":0}
        ,
        .documents = &.{ "[]", "[ ]" },
    },
    .{
        .name = "map",
        .schema =
        \\{"type":"object","additionalProperties":{"type":"integer"}}
        ,
        .documents = &.{
            \\{}
            ,
            \\{"a":1,"bb":-2}
        },
    },
    .{
        // keywords that say what a value is without a `type`
        .name = "typeless-map",
        .schema =
        \\{"additionalProperties":{"additionalProperties":{"type":["integer","null"]}}}
        ,
        .documents = &.{
            \\{}
            ,
            \\{"a":{"b":1,"c":null},"d":{}}
        },
    },
    .{
        .name = "typeless-items",
        .schema =
        \\{"items":{"properties":{"a":{"items":{"type":"integer"},"maxItems":1}},"required":["a"]},"minItems":1}
        ,
        .documents = &.{
            \\[{"a":[]}]
            ,
            \\[{"a":[1]},{"a":[]}]
        },
    },
    .{
        // `maxItems` and nothing else: xgrammar takes any value, of any length; zspark an array within the bound
        .name = "typeless-bound",
        .schema =
        \\{"type":"object","properties":{"a":{"maxItems":1},"b":{"minItems":2}},"required":["a","b"]}
        ,
        .documents = &.{
            \\{"a":[],"b":[1,2]}
            ,
            \\{"a":[{"k":null}],"b":["x","y","w"]}
        },
        .narrowed = true,
    },
};

/// FNV-1a over the words' bytes, least significant first: what a checkpoint mask is written as.
fn hash(words: []const u32, keep: ?[]const u32) u64 {
    var value: u64 = 0xcbf29ce484222325;
    for (words, 0..) |word, at| {
        const kept = if (keep) |k| word & k[at] else word;
        for (0..4) |byte| {
            value ^= (kept >> @intCast(8 * byte)) & 0xff;
            value *%= 0x100000001b3;
        }
    }
    return value;
}

const Kind = enum { document, bytes, cut, random, only, special };

const Run = struct {
    gpa: Allocator,
    out: *Io.Writer,
    tokenizer: *const schema.Tokenizer,
    stops: []const u32,
    width: usize,
    random: std.Random,
    /// xgrammar's mask and the Zig matcher's.
    theirs: []u32,
    ours: []u32,
    /// The tokens that hold neither 0xED nor a backslash; null when masks are written whole.
    plain: ?[]const u32,
    /// The ids whose bytes begin with each byte, special tokens left out.
    by_first: [256]std.ArrayList(u32) = @splat(.empty),
    /// Each distinct mask as its line's text; a mask's index is its place.
    table: std.StringArrayHashMapUnmanaged(void) = .empty,
    arena: std.heap.ArenaAllocator,
    tokens: std.ArrayList(u32) = .empty,
    marks: std.ArrayList(u32) = .empty,
    candidates: std.ArrayList(u32) = .empty,
    steps: usize = 0,
    differing: usize = 0,
    walks: usize = 0,
    /// Tokens xgrammar's acceptor takes that its mask lacked.
    completed: usize = 0,
    /// How many more differing tokens are printed.
    show: usize = 0,

    /// The index of xgrammar's mask as it stands.
    fn mark(self: *Run) !u32 {
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(self.gpa);
        if (self.plain) |plain| {
            var count: usize = 0;
            for (self.theirs) |word| count += @popCount(word);
            try text.print(self.gpa, "{d} {x:0>16} {x:0>16}", .{ count, hash(self.theirs, null), hash(self.theirs, plain) });
        } else {
            for (self.theirs) |word| try text.print(self.gpa, "{x:0>8}", .{word});
        }
        if (self.table.getIndex(text.items)) |index| return @intCast(index);
        try self.table.put(self.gpa, try self.arena.allocator().dupe(u8, text.items), {});
        return @intCast(self.table.count() - 1);
    }

    /// The next token of `rest` for a cut of `kind`.
    fn piece(self: *Run, kind: Kind, rest: []const u8) !u32 {
        var best: ?u32 = null;
        self.candidates.clearRetainingCapacity();
        for (self.by_first[rest[0]].items) |id| {
            const bytes = self.tokenizer.tokenBytes(id);
            if (!std.mem.startsWith(u8, rest, bytes)) continue;
            if (best == null or bytes.len < self.tokenizer.tokenBytes(best.?).len) best = id;
            try self.candidates.append(self.gpa, id);
        }
        if (best == null) return error.NoTokenForByte;
        return if (kind == .bytes) best.? else self.candidates.items[self.random.uintLessThan(usize, self.candidates.items.len)];
    }

    fn fill(self: *Run, xg: *XgMatcher, matcher: ?*schema.Matcher) !void {
        if (zxg_fill(xg, self.theirs.ptr, @intCast(self.theirs.len), &err, err.len) < 0) return error.Xgrammar;
        // the bits past the width are no token's
        if (self.width % 32 != 0) self.theirs[self.theirs.len - 1] &= (@as(u32, 1) << @intCast(self.width % 32)) - 1;
        self.steps += 1;
        if (matcher) |m| {
            try m.fill(self.ours);
            // xgrammar's mask can lack a token its own AcceptToken takes (at 8262b5c: `},` and the like where an
            // object reached through a recursive `$ref` ends). The mask written is xgrammar's with those: each
            // token the Zig matcher takes and the mask lacks is put to xgrammar's acceptor, whose word it is.
            for (self.ours, self.theirs, 0..) |ours, *theirs, word| {
                var more = ours & ~theirs.*;
                while (more != 0) : (more &= more - 1) {
                    const id: u32 = @intCast(word * 32 + @ctz(more));
                    if (zxg_accept(xg, @intCast(id), &err, err.len) != 1) continue;
                    if (zxg_rollback(xg, 1, &err, err.len) < 0) return error.Xgrammar;
                    theirs.* |= @as(u32, 1) << @intCast(id % 32);
                    self.completed += 1;
                }
            }
        }
        try self.marks.append(self.gpa, try self.mark());
        if (matcher != null) {
            if (std.mem.eql(u32, self.ours, self.theirs)) return;
            self.differing += 1;
            // with GRAMMAR_MASKS_SHOW set, the first tokens only one of the two takes, for whoever reads a
            // difference the test reports
            for (self.ours, self.theirs, 0..) |ours, theirs, word| {
                var either = ours ^ theirs;
                while (either != 0 and self.show > 0) : (either &= either - 1) {
                    const id: u32 = @intCast(word * 32 + @ctz(either));
                    self.show -= 1;
                    std.debug.print("  after {d} tokens ending in {any}: {d} `{f}` is taken by {s} alone\n", .{
                        self.marks.items.len - 1,                                                  self.tokens.items[self.tokens.items.len -| 4..][0..@min(4, self.marks.items.len - 1)],
                        id,                                                                        std.zig.fmtString(if (id < self.tokenizer.count()) self.tokenizer.tokenBytes(id) else ""),
                        if (ours >> @intCast(id % 32) & 1 != 0) "the Zig matcher" else "xgrammar",
                    });
                }
            }
        }
    }

    fn write(self: *Run, kind: Kind, ended: bool, masks: bool) !void {
        try self.out.print("walk {s} {d}\nt", .{ @tagName(kind), @intFromBool(ended) });
        for (self.tokens.items) |id| try self.out.print(" {d}", .{id});
        if (masks) {
            try self.out.writeAll("\nm");
            for (self.marks.items) |index| try self.out.print(" {d}", .{index});
        }
        try self.out.writeAll("\n");
        self.walks += 1;
    }

    /// A document both take, cut by `kind`, and the stop after it.
    fn document(self: *Run, compiled: *XgGrammar, grammar: *schema.Grammar, kind: Kind, text: []const u8) !void {
        const xg = zxg_matcher_new(compiled, &err, err.len) orelse return error.Xgrammar;
        defer zxg_matcher_free(xg);
        var matcher: schema.Matcher = try .init(grammar);
        defer matcher.deinit();
        self.tokens.clearRetainingCapacity();
        self.marks.clearRetainingCapacity();
        if (kind == .document) {
            try self.tokenizer.encode(self.gpa, text, &self.tokens);
        } else {
            var at: usize = 0;
            while (at < text.len) {
                const id = try self.piece(kind, text[at..]);
                try self.tokens.append(self.gpa, id);
                at += self.tokenizer.tokenBytes(id).len;
            }
        }
        try self.tokens.append(self.gpa, self.stops[0]);
        for (self.tokens.items, 0..) |id, at| {
            try self.fill(xg, &matcher);
            const theirs = zxg_accept(xg, @intCast(id), &err, err.len);
            const ours = try matcher.accept(id);
            if (theirs != 1 or !ours) {
                std.debug.print("token {d} of {d} ({d}) refused: xgrammar {d}, Zig {}\ndocument: {s}\n", .{ at, self.tokens.items.len, id, theirs, ours, text });
                return error.DocumentRefused;
            }
        }
        if (zxg_terminated(xg) != 1 or !matcher.finished()) return error.DocumentDidNotEnd;
        try self.write(kind, true, true);
    }

    /// A document xgrammar takes whole, cut into the shortest tokens; the Zig matcher must refuse one of them.
    fn onlyTheirs(self: *Run, compiled: *XgGrammar, grammar: *schema.Grammar, text: []const u8) !void {
        const xg = zxg_matcher_new(compiled, &err, err.len) orelse return error.Xgrammar;
        defer zxg_matcher_free(xg);
        var matcher: schema.Matcher = try .init(grammar);
        defer matcher.deinit();
        self.tokens.clearRetainingCapacity();
        self.marks.clearRetainingCapacity();
        var at: usize = 0;
        while (at < text.len) {
            const id = try self.piece(.bytes, text[at..]);
            try self.tokens.append(self.gpa, id);
            at += self.tokenizer.tokenBytes(id).len;
        }
        try self.tokens.append(self.gpa, self.stops[0]);
        var refused = false;
        for (self.tokens.items) |id| {
            try self.fill(xg, null);
            if (zxg_accept(xg, @intCast(id), &err, err.len) != 1) {
                std.debug.print("xgrammar refuses a document it was said to take: {s}\n", .{text});
                return error.DocumentRefused;
            }
            if (!refused and !try matcher.accept(id)) refused = true;
        }
        if (!refused) {
            std.debug.print("the Zig matcher takes a document it was said to refuse: {s}\n", .{text});
            return error.DocumentTaken;
        }
        try self.write(.only, true, true);
    }

    /// A walk over tokens both masks hold, short and structural ones drawn more often so that it ends.
    fn walk(self: *Run, compiled: *XgGrammar, grammar: *schema.Grammar, limit: usize) !void {
        const xg = zxg_matcher_new(compiled, &err, err.len) orelse return error.Xgrammar;
        defer zxg_matcher_free(xg);
        var matcher: schema.Matcher = try .init(grammar);
        defer matcher.deinit();
        self.tokens.clearRetainingCapacity();
        self.marks.clearRetainingCapacity();
        var ended = false;
        for (0..limit) |_| {
            try self.fill(xg, &matcher);
            const draw = self.random.float(f32);
            self.candidates.clearRetainingCapacity();
            var all: usize = 0;
            for (self.theirs, self.ours, 0..) |theirs, ours, word| {
                var both = theirs & ours;
                while (both != 0) : (both &= both - 1) {
                    const id: u32 = @intCast(word * 32 + @ctz(both));
                    all += 1;
                    const bytes = if (id < self.tokenizer.count()) self.tokenizer.tokenBytes(id) else "";
                    const stop = std.mem.findScalar(u32, self.stops, id) != null;
                    const keep = if (draw < 0.3)
                        stop or (bytes.len == 1 and std.mem.findScalar(u8, "\"}],:", bytes[0]) != null)
                    else if (draw < 0.6)
                        stop or bytes.len == 1
                    else if (draw < 0.75)
                        stop or std.mem.findAny(u8, bytes, "\"\\,:}]") != null
                    else
                        true;
                    if (keep) try self.candidates.append(self.gpa, id);
                }
            }
            if (all == 0) break;
            if (self.candidates.items.len == 0) {
                for (self.theirs, self.ours, 0..) |theirs, ours, word| {
                    var both = theirs & ours;
                    while (both != 0) : (both &= both - 1) try self.candidates.append(self.gpa, @intCast(word * 32 + @ctz(both)));
                }
            }
            const id = self.candidates.items[self.random.uintLessThan(usize, self.candidates.items.len)];
            try self.tokens.append(self.gpa, id);
            const theirs = zxg_accept(xg, @intCast(id), &err, err.len);
            const ours = try matcher.accept(id);
            if (theirs != 1 or !ours) {
                std.debug.print("a token both masks hold was refused: {d}, xgrammar {d}, Zig {}\n", .{ id, theirs, ours });
                return error.MaskedTokenRefused;
            }
            if (matcher.finished() != (zxg_terminated(xg) == 1)) return error.EndsDiffer;
            if (matcher.finished()) {
                ended = true;
                break;
            }
        }
        if (!ended) _ = self.marks.pop();
        if (!ended) _ = self.tokens.pop();
        if (self.tokens.items.len == 0) return;
        try self.write(.random, ended, true);
    }
};

pub fn main(init: std.process.Init) !u8 {
    // one run and out: everything is the process arena's
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    if (args.len != 4) {
        std.debug.print("usage: grammar_masks TOKENIZER_JSON tiny|checkpoint XGRAMMAR_COMMIT\n", .{});
        return 2;
    }
    const tiny = std.mem.eql(u8, args[2], "tiny");
    var stdout_buffer: [1 << 16]u8 = undefined;
    var stdout_file: Io.File.Writer = .initStreaming(.stdout(), io, &stdout_buffer);
    const out = &stdout_file.interface;

    var tokenizer = try schema.Tokenizer.load(gpa, io, Io.Dir.cwd(), args[1], undefined);
    defer tokenizer.deinit(gpa);
    const count = tokenizer.count();
    // ids past the tokenizer's, as a model's logits have them
    const width: usize = if (tiny) count + 40 else 248320;
    var stop_ids: [2]u32 = undefined;
    const stops: []const u32 = if (tiny) blk: {
        stop_ids[0] = tokenizer.addedId("<|end|>").?;
        break :blk stop_ids[0..1];
    } else blk: {
        stop_ids = .{ tokenizer.addedId("<|im_end|>").?, tokenizer.addedId("<|endoftext|>").? };
        break :blk stop_ids[0..2];
    };
    var xg_stops: [2]i32 = undefined;
    for (stops, 0..) |stop, at| xg_stops[at] = @intCast(stop);

    // xgrammar's view of the vocabulary: each token's bytes, a special token's none (never allowed); and a
    // second view where a special token has its text, as TensorFold's `from_huggingface` gives it
    const starts = try gpa.alloc(u32, count + 1);
    defer gpa.free(starts);
    var views: [2]*XgCompiler = undefined;
    var special: ?u32 = null;
    for (0..2) |view| {
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        for (0..count) |index| {
            const id: u32 = @intCast(index);
            starts[index] = @intCast(bytes.items.len);
            if (tokenizer.isSpecial(id)) {
                if (special == null and std.mem.findScalar(u32, stops, id) == null) special = id;
                if (view == 0) continue;
            }
            try bytes.appendSlice(gpa, tokenizer.tokenBytes(id));
        }
        starts[count] = @intCast(bytes.items.len);
        const info = zxg_tokenizer_new(bytes.items.ptr, starts.ptr, @intCast(count), @intCast(width), &xg_stops, @intCast(stops.len), &err, err.len) orelse return fail();
        views[view] = zxg_compiler_new(info, 8, 256 << 20, &err, err.len) orelse return fail();
    }

    var vocabulary: schema.Vocabulary = try .init(gpa, &tokenizer, width, stops);
    defer vocabulary.deinit(gpa);
    const words = vocabulary.words();
    var prng: std.Random.DefaultPrng = .init(0x017_c0de);
    var run: Run = .{
        .gpa = gpa,
        .out = out,
        .tokenizer = &tokenizer,
        .stops = stops,
        .width = width,
        .random = prng.random(),
        .theirs = try gpa.alloc(u32, words),
        .ours = try gpa.alloc(u32, words),
        .plain = null,
        .arena = .init(gpa),
    };
    const plain = try gpa.alloc(u32, words);
    @memset(plain, 0);
    for (0..count) |index| {
        const id: u32 = @intCast(index);
        const bytes = tokenizer.tokenBytes(id);
        if (bytes.len == 0 or tokenizer.isSpecial(id)) continue;
        try run.by_first[bytes[0]].append(gpa, id);
        if (std.mem.findAny(u8, bytes, "\xed\\") == null) plain[index / 32] |= @as(u32, 1) << @intCast(index % 32);
    }
    for (stops) |stop| plain[stop / 32] |= @as(u32, 1) << @intCast(stop % 32);
    if (!tiny) run.plain = plain;

    try out.print("# xgrammar's masks over the schemas and token sequences of tools/reference/grammar_masks.zig, which\n", .{});
    try out.print("# says what each line is. Written by tools/reference/grammar_masks.sh; not edited by hand.\n", .{});
    try out.print("xgrammar {s} whitespace {d}\n", .{ args[3], whitespace });
    try out.print("vocabulary {s} {d} {d}", .{ args[2], count, width });
    for (stops) |stop| try out.print(" {d}", .{stop});
    try out.writeAll("\n");

    const walks: usize = if (tiny) 10 else 3;
    const limit: usize = if (tiny) 90 else 48;
    for (corpus) |case| {
        const compiled = zxg_compile_json_schema(views[0], case.schema.ptr, case.schema.len, whitespace, &err, err.len) orelse {
            std.debug.print("{s}: xgrammar refuses it: {s}\n", .{ case.name, said() });
            return 1;
        };
        defer zxg_grammar_free(compiled);
        var diagnostic: schema.Diagnostic = .{};
        var grammar = schema.Grammar.init(gpa, &vocabulary, case.schema, .{}, &diagnostic) catch {
            std.debug.print("{s}: the Zig matcher refuses it: {s}\n", .{ case.name, diagnostic.message() });
            return 1;
        };
        defer grammar.deinit();
        try out.print("schema {s}{s}\n{s}\n", .{ case.name, if (case.narrowed) " narrowed" else "", case.schema });
        const before = .{ run.steps, run.differing, run.walks, run.completed };
        run.show = if (init.environ_map.get("GRAMMAR_MASKS_SHOW") != null) 6 else 0;
        for (case.documents) |text| {
            errdefer std.debug.print("{s}: {s}\n", .{ case.name, said() });
            try run.document(compiled, &grammar, .document, text);
            if (tiny) try run.document(compiled, &grammar, .bytes, text);
            try run.document(compiled, &grammar, .cut, text);
        }
        for (case.only) |text| try run.onlyTheirs(compiled, &grammar, text);
        for (0..walks) |_| try run.walk(compiled, &grammar, limit);
        if (std.mem.eql(u8, case.name, "flat")) {
            // a special token in a string: taken by the view that has its text
            const told = zxg_compile_json_schema(views[1], case.schema.ptr, case.schema.len, whitespace, &err, err.len) orelse return fail();
            defer zxg_grammar_free(told);
            const xg = zxg_matcher_new(told, &err, err.len) orelse return fail();
            defer zxg_matcher_free(xg);
            run.tokens.clearRetainingCapacity();
            try tokenizer.encode(gpa, "{\"name\":\"a", &run.tokens);
            try run.tokens.append(gpa, special.?);
            for (run.tokens.items) |id| {
                if (zxg_accept(xg, @intCast(id), &err, err.len) != 1) {
                    std.debug.print("xgrammar refuses the special token {d} though it has its text\n", .{id});
                    return 1;
                }
            }
            try run.write(.special, false, false);
        }
        std.debug.print("{s:<24} {d:>3} walks, {d:>5} steps, {d:>4} where the two masks differ, {d:>4} tokens xgrammar's mask lacked\n", .{ case.name, run.walks - before[2], run.steps - before[0], run.differing - before[1], run.completed - before[3] });
    }
    for (run.table.keys()) |line| try out.print("mask {s}\n", .{line});
    try out.print("# {d} tokens were added to masks that lacked them though xgrammar's AcceptToken takes them\n", .{run.completed});
    try out.flush();
    std.debug.print("{d} walks, {d} steps, {d} distinct masks, {d} steps where the two masks differ\n", .{ run.walks, run.steps, run.table.count(), run.differing });
    return 0;
}

fn fail() u8 {
    std.debug.print("{s}\n", .{said()});
    return 1;
}

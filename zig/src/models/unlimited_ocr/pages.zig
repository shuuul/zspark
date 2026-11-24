//! Unlimited-OCR-2's page coordinates (spec 147 WS-05), host only: what a row's page2d RoPE, its router's query
//! page and a block's image pages are, from the prompt's ids and the reply's tokens. HF builds them from the whole
//! prompt (`_sparse_base_impl.py` `_build_page_rope_pos`, `_page_rope_offsets`, `_build_page_aware_ids` with
//! `_PAGE_SCOPE_PREFILL` on as `configure_dash_inference` sets it; `modeling_unlimitedocr.py`
//! `_uocr_dash_decode_allow`) and then a reply's tokens one at a time (`_main_rope_coords`' decode state machine):
//! the prompt's last image token, its `<PAGE>` count and a page bucket's prompt rows are known only once the
//! whole prompt is, so `plan` takes it whole and `Track` carries the reply on (Apache-2.0, read 2026-10-27).
//!
//! The prompt (`cum` the 1-based `<PAGE>` count at a row, `last` the last image token's row, `n = cum[last]`):
//! - the RoPE page is `cum` up to `last` and `max(cum - n, 0)` after (0, the instruction's bucket, is reserved);
//!   the RoPE token is the row's running count within its page's bucket, from 0;
//! - the query page is `cum - 1` up to `last` and `cum - 1 - n` after, -1 below 0;
//! - an image token's page is `cum - 1` (`per_page_image_tokens` buckets before the first marker), a block's
//!   pages the least and largest of its image tokens', -1 both without one.
//! A reply's token t after the row before (its page, token, marker count): a `<PAGE>` moves the page on by one,
//! at most to the prompt's last bucket, its token to that bucket's count of prompt rows up to `last`, and counts a
//! marker; any other token moves the token on by one. Its query page is `markers - 1 - n`, -1 below 0.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// A row's coordinates, int32 [rows, 3] on the device: its page2d RoPE's token and page, and its query page: the
/// image page whose blocks its router picks among, -1 for none (every remote block).
pub const Coords = extern struct { tok: i32, page: i32, qp: i32 };

/// A block's least and largest image page, int32 [blocks, 2]; -1 both for a block without an image token.
pub const Pages = extern struct { least: i32, most: i32 };

pub const none: Pages = .{ .least = -1, .most = -1 };

pub const Tokens = struct {
    image: i32,
    page: i32,
    /// `per_page_image_tokens`: an image token's page before the prompt's first marker
    per_page: usize,
};

pub const Error = error{
    /// A prompt with no image token, or no `<PAGE>` up to its last one: HF falls back to 1d RoPE on
    /// `rope_theta`, which zspark does not read on DASH.
    Unpaged,
};

/// The reply's state machine after a row: `advance` gives the next row's coordinates.
pub const Track = struct {
    page: i32,
    tok: i32,
    markers: i32,
    /// `<PAGE>` markers up to the prompt's last image token
    prompt_pages: i32,
    page_token: i32,
    /// a page bucket's prompt rows up to the last image token (HF's `pf_cnt`), one a bucket
    starts: []const i32,

    pub fn advance(self: *Track, token: i32) Coords {
        if (token == self.page_token) {
            self.page = @min(self.page + 1, @as(i32, @intCast(self.starts.len)) - 1);
            self.tok = self.starts[@intCast(self.page)];
            self.markers += 1;
        } else self.tok += 1;
        const qp = self.markers - 1 - self.prompt_pages;
        return .{ .tok = self.tok, .page = self.page, .qp = if (qp >= 0) qp else -1 };
    }
};

/// The prompt's rows' coordinates into `coords` (one a row), its blocks' image pages into `blocks` (one a
/// 32-key block it touches, `ceil(rows / 32)`), and the track after its last row, whose `starts` lie in `starts`
/// (at least the prompt's rows plus one).
pub fn plan(ids: []const i32, t: Tokens, block: usize, coords: []Coords, blocks: []Pages, starts: []i32) Error!Track {
    std.debug.assert(coords.len == ids.len and blocks.len == (ids.len + block - 1) / block and starts.len > ids.len);
    const last = std.mem.lastIndexOfScalar(i32, ids, t.image) orelse return error.Unpaged;
    var n: i32 = 0;
    for (ids[0 .. last + 1]) |id| n += @intFromBool(id == t.page);
    if (n == 0) return error.Unpaged;
    // the RoPE page a row, then the buckets' running counts
    var cum: i32 = 0;
    var buckets: usize = 0;
    for (ids, coords, 0..) |id, *c, i| {
        cum += @intFromBool(id == t.page);
        const prefill = i <= last;
        c.page = if (prefill) cum else @max(cum - n, 0);
        const qp = if (prefill) cum - 1 else cum - 1 - n;
        c.qp = if (qp >= 0) qp else -1;
        buckets = @max(buckets, @as(usize, @intCast(c.page)) + 1);
    }
    const counts = starts[0..buckets];
    @memset(counts, 0);
    for (coords) |*c| {
        const k: usize = @intCast(c.page);
        c.tok = counts[k];
        counts[k] += 1;
    }
    // HF's pf_cnt: the bucket's rows up to the last image token
    @memset(counts, 0);
    for (coords[0 .. last + 1]) |c| counts[@intCast(c.page)] += 1;
    // a block's image pages
    @memset(blocks, none);
    cum = 0;
    var run: usize = 0;
    for (ids, 0..) |id, i| {
        cum += @intFromBool(id == t.page);
        if (id != t.image) continue;
        run += 1;
        const page: i32 = if (cum > 0) cum - 1 else @intCast((run - 1) / t.per_page);
        const b = &blocks[i / block];
        b.least = if (b.least < 0) page else @min(b.least, page);
        b.most = @max(b.most, page);
    }
    const end = coords[coords.len - 1];
    return .{ .page = end.page, .tok = end.tok, .markers = cum, .prompt_pages = n, .page_token = t.page, .starts = counts };
}

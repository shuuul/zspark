//! A scored row's start, found on the device (`zig/kernels/common/pick.cu`, spec 028): what a choice reads of a row
//! of logits without reading the row back. Flash Next's MTP head asks it of the row it scored; GLM's ranks ask
//! it each of its half of the vocabulary, and the halves' are joined (`joined`, spec 045).

const std = @import("std");
const cuda = @import("../cuda/cuda.zig");
const sampling = @import("sampling.zig");

/// What a draft's choice reads of a scored row, found on the device (spec 028): `out` u32
/// [`Picked.wordsFor` its length], as `Picked` reads it and asked to a rank by `Picked.ask`, from `logits`
/// bf16, one row.
pub fn pick(logits: cuda.View, out: cuda.View) cuda.Error!void {
    return cuda.launched(zs_pick(&logits, &out));
}

/// `pick` of each row of `logits` (bf16 [rows, n]) into the same row of `out` (u32 [rows, `Picked.wordsFor(n)`
/// or more]), in one set of launches for all of them (spec 144 WS-03): each row's words are what `pick` gives it
/// alone. A row is cut at its byte of `asks` (u8 [rows]) where given, else at the rank its own words ask.
pub fn pickRows(logits: cuda.View, out: cuda.View, asks: ?cuda.View) cuda.Error!void {
    return cuda.launched(zs_pick_rows(&logits, &out, if (asks) |*given| given else null));
}

/// `pick`'s words on the host (zig/kernels/common/pick.cu).
pub const Picked = struct {
    /// The row's largest and smallest key (`sampling.keyOf`).
    largest: u16,
    smallest: u16,
    /// The sum of exp(v - largest) over the row's logits v > largest - 24, as `normalizer` takes it but of fp32
    /// terms added in fp32: within 1e-5 of the host's, relatively.
    sum: f64,
    /// The row's logits at or above the cut, the key of the rank asked to: how many, and where they fit
    /// (`count` is then that many) their keys and columns by (key down, column up). More than `slots` of them
    /// (a cut among many equal logits) leaves `count` 0: the row has to be read.
    total: usize,
    keys: [slots]u16,
    columns: [slots]u32,
    count: usize,

    /// The highest rank a caller may ask to.
    pub const most = 64;
    pub const slots = 128;
    /// The words the host reads; the kernels keep more after them.
    pub const words = 8 + 2 * slots;
    /// The word the caller writes, the rank it asks to, which the kernel reads and never writes; its byte offset.
    pub const rank_word = 3;
    const rank_at = rank_word * 4;

    /// The buffer's words for rows of `n` logits.
    pub fn wordsFor(n: usize) usize {
        return words + 2 * 256 + 8 + 3 * ((n + 1023) / 1024);
    }

    /// Asks `out` to the rank `rank`, 1 to `most`: the next `pick` into it cuts at the row's `rank`-th logit.
    pub fn ask(out: cuda.Buffer, rank: u32) cuda.Error!void {
        std.debug.assert(rank >= 1 and rank <= most);
        return out.uploadLaunched(rank_at, std.mem.asBytes(&rank));
    }

    /// The rank a row of `columns` logits of a vocabulary of `vocab` is asked to for `settings`; 0 where its
    /// start cannot settle them (`sampling.leadingNeeded`: no cut at all, a `top_k` past `most` or the
    /// columns), and the row is read whole.
    pub fn rankFor(settings: sampling.Sampling, columns: usize, vocab: usize) u32 {
        const needed = sampling.leadingNeeded(settings, vocab) orelse 0;
        return if (needed <= @min(most, columns)) @intCast(needed) else 0;
    }

    pub fn read(raw: *const [words]u32) Picked {
        var self: Picked = .{ .largest = @intCast(raw[0]), .smallest = @intCast(raw[1]), .sum = @as(f32, @bitCast(raw[4])), .total = raw[2], .keys = undefined, .columns = undefined, .count = 0 };
        if (self.total > slots) return self;
        // The kernel leaves them in the order its threads came: an insertion by key down, then column up.
        for (0..self.total) |i| {
            const key: u16 = @intCast(raw[8 + 2 * i]);
            const column = raw[9 + 2 * i];
            var at = i;
            while (at > 0 and (self.keys[at - 1] < key or (self.keys[at - 1] == key and self.columns[at - 1] > column))) : (at -= 1) {
                self.keys[at] = self.keys[at - 1];
                self.columns[at] = self.columns[at - 1];
            }
            self.keys[at] = key;
            self.columns[at] = column;
        }
        self.count = self.total;
        return self;
    }
};

/// Scratch of `joined`: a part's tokens as the row's ids, and the joined start.
pub const Joined = struct {
    part_ids: [most_parts][Picked.slots]u32 = undefined,
    keys: [most_parts * Picked.slots]u16 = undefined,
    ids: [most_parts * Picked.slots]u32 = undefined,

    const most_parts = 2;
};

/// The start of a row whose parts were picked apart, each of `part` columns and in the row's order
/// (`sampling.joinLeading`); null where a part's pick gave none (more than `Picked.slots` at its cut).
pub fn joined(picked: []const Picked, part: usize, scratch: *Joined) ?sampling.Leading {
    std.debug.assert(picked.len <= Joined.most_parts);
    var parts: [Joined.most_parts]sampling.Leading = undefined;
    for (picked, parts[0..picked.len], scratch.part_ids[0..picked.len], 0..) |*one, *out, *ids, index| {
        for (ids[0..one.count], one.columns[0..one.count]) |*id, column| id.* = @intCast(index * part + column);
        out.* = .{ .keys = one.keys[0..one.count], .ids = ids[0..one.count], .smallest = one.smallest, .len = part, .sum = one.sum };
    }
    return sampling.joinLeading(parts[0..picked.len], &scratch.keys, &scratch.ids);
}

extern fn zs_pick(logits: *const cuda.View, out: *const cuda.View) c_int;
extern fn zs_pick_rows(logits: *const cuda.View, out: *const cuda.View, asks: ?*const cuda.View) c_int;

//! Recall (spec 106 WS-08, `serve --recall`, off by default): drafts copied from a record of the ids earlier
//! requests read and wrote, where a reply repeats what another request's prompt or reply held (a request sent
//! again, an agent's call re-run, a file written again). TensorFold 1.0.12's (`b2f5b0c8`, `74f542b6`), whose
//! record, store and pricing are the vendored host's `lanes` and are used as they are: ids in chunk files under
//! one directory a tokenizer, found by 8-token keys in RAM (`lanes/recall_record.zig`), the oldest chunk dropped
//! past the cap (`lanes/recall_store.zig`), and a copy proposed only where its expected tokens, at the landing
//! rate its match length's band has shown, beat the stream's own rounds, a band that loses probed one ask in 32
//! backing off to 1,024 (`lanes/recall.zig`). `copied.zig` asks the record where the context's own lookup
//! finds nothing, as TensorFold's CUDA lone driver does (`families/nemotron/cuda_drafts.zig`, `copies`), and a
//! copy is then offered as a context's is: where its first token is the head's first draft. A draft chooses
//! nothing (`draft.zig`), so a reply's bits are those without Recall.
//!
//! Departures from TensorFold: the record lies under `~/.cache/zspark/recall/` and not `~/.cache/tensorfold/`,
//! so a TensorFold server on the same machine (the LAN's) neither holds its lock nor reads zspark's; a prompt
//! goes in when its reply begins, whole, and the store's own check skips what the record holds, where
//! TensorFold adds it after the prompt pass from its resume point; no fill grafts (zspark's windows are one
//! chain). `GET` and `DELETE /v1/recall` are the vendored host's routes over the same store.

const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");

pub const Store = lanes.recall_store.Store;
pub const Identity = lanes.recall_store.Identity;
pub const Reader = lanes.recall.Reader;

/// What `serve`'s flags ask of the record.
pub const Asked = struct {
    /// `--recall-dir`; null: `defaultDir`.
    dir: ?[]const u8 = null,
    /// `--recall-gib`: the record's disk, TensorFold's default; its RAM is about twice as much once full.
    gib: f64 = 1.0 / 16.0,
    /// `--recall-read-only`: the record is read and nothing is added.
    read_only: bool = false,
};

/// The least `--recall-gib`: two chunk files of 4 MiB, as TensorFold's `cli.zig` refuses below.
pub const least_gib: f64 = 1.0 / 128.0;

/// `$HOME/.cache/zspark/recall/<the tokenizer's identity, 16 hex digits>`.
pub fn defaultDir(a: Allocator, home: []const u8, identity: Identity) ![]const u8 {
    const short = identity.short();
    return std.fmt.allocPrint(a, "{s}/.cache/zspark/recall/{s}", .{ home, &short });
}

/// The record in `asked.dir` or the default one, or null with `problem` set (another tokenizer wrote it, the
/// directory cannot be made, no HOME and no directory). Closed by `Store.close`.
pub fn open(gpa: Allocator, io: std.Io, a: Allocator, identity: Identity, asked: Asked, home: ?[]const u8, problem: *[]const u8) !?*Store {
    const dir = asked.dir orelse try defaultDir(a, home orelse {
        problem.* = "--recall: no HOME and no --recall-dir";
        return null;
    }, identity);
    const cap: u64 = @intFromFloat(asked.gib * (1 << 30));
    return Store.open(gpa, io, identity, .{ .dir = dir, .cap_bytes = cap, .read_only = asked.read_only }, problem, a);
}

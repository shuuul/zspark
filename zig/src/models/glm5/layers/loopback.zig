//! Both of GLM's ranks in one process, their partials gathered in place: a rank's share of a branch is written
//! where the other reads it, in the gather's order, so one Spark gives the bits two give (spec 044). What the
//! tests run, and the reference a rank a process is held to (`ranks_forward.zig`, whose `Held(world)` this is; spec
//! 045's `zspark ranks-check`).
//!
//! `Model`, `State`, `Buffers`, `Graphs` and the functions below are what `draft.Decoder` asks of a model
//! (`zig/src/decode/draft.zig`), so the shared decoder runs both shares as one.

const held = @import("ranks_forward.zig");
const layout = @import("layout.zig");

const both = held.Held(layout.world);

pub const Error = held.Error;
pub const Score = held.Score;
pub const Point = held.Point;
pub const Model = both.Model;
pub const State = both.State;
pub const Buffers = both.Buffers;
pub const Graphs = both.Graphs;

pub const partial = held.partial;
pub const gathered = held.gathered;
pub const stage = both.stage;
pub const compute = both.compute;
pub const computeWatched = both.computeWatched;
pub const splits = both.splits;
pub const forward = both.forward;
pub const forwardScoring = both.forwardScoring;
pub const commit = both.commit;
pub const logitsRow = both.logitsRow;

//! How deep a seat of the pool may draft by the round's width (spec 144 WS-04). Host only.
//!
//! On two ranks the MTP head's levels are launched kernel by kernel and the whole round waits for the deepest
//! seat's chain, a level at a time; a draft past the fourth is kept often enough to earn its row, but not the
//! level it makes three other seats wait for. WS-04's model over WS-01's recorded rounds (two ranks, bf16
//! caches) put a cap of 4 drafts at +4.8% summed tok/s at 3 seats and +7.0% at 4 (+5 to 7%), +1.6% at 2 and
//! -3.0% at 1; on one Spark, whose head passes are graphs, +3.8% at 4 and less below: not taken there.
//!
//! A departure from TensorFold (6 drafts at any width) and from vLLM and SGLang, neither of which cuts the
//! depth at 1 to 4 streams: vLLM's `num_speculative_tokens_per_batch_size` is opt-in with no default schedule,
//! SGLang's adaptive tiers cut by width from 8 streams. Drafts choose no token: the cap changes the rows a
//! round verifies, never a reply's bits. DFlash2's drafts (`flashJob`) are one pass, no levels, and take none.

/// Where the cap holds: on two ranks, everywhere (a test on one rank) or nowhere (a test, or a bench's base).
pub const Cap = enum { two_ranks, always, never };

/// The most drafts a seat chains while at least `from_seats` seats decode in the round.
pub const most_drafts: usize = 4;
pub const from_seats: usize = 3;

/// Whether a round of `decoding` seats is capped, on two ranks or one.
pub fn engages(cap: Cap, two_ranks: bool, decoding: usize) bool {
    const on = switch (cap) {
        .two_ranks => two_ranks,
        .always => true,
        .never => false,
    };
    return on and decoding >= from_seats;
}

/// The drafts a seat may chain where it would chain `most` alone.
pub fn room(cap: Cap, two_ranks: bool, decoding: usize, most: usize) usize {
    return if (engages(cap, two_ranks, decoding)) @min(most, most_drafts) else most;
}

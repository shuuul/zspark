//! Clock-free tokens per modelled cost, used by Qwen's MTP budgets. Costs,
//! confidence, trust and the decision to draft remain the family's; only the EWMA arithmetic is shared.

pub const Rate = struct {
    tokens: f64 = 2,
    cost: f64,

    pub const weight = 0.125;

    pub fn begin(cost: f64) Rate {
        return .{ .cost = cost };
    }

    pub fn of(self: Rate) f64 {
        return self.tokens / self.cost;
    }

    /// Preserve this order: reassociating to `(1-weight)*previous + weight*sample` changes bits.
    pub fn average(previous: f64, sample: f64) f64 {
        return previous + weight * (sample - previous);
    }

    /// A round produced `tokens` at modelled `cost`, not elapsed time.
    pub fn done(self: *Rate, tokens: usize, cost: f64) void {
        self.tokens = average(self.tokens, @floatFromInt(tokens));
        self.cost = average(self.cost, cost);
    }
};

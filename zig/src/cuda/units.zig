//! Bytes and nanoseconds as the logs and reports print them. Host only: nothing here reaches the device.

const std = @import("std");

pub fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

pub fn seconds(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

/// The seconds since `since` on the awake clock.
pub fn secondsSince(io: std.Io, since: std.Io.Timestamp) f64 {
    return seconds(@intCast(since.untilNow(io, .awake).toNanoseconds()));
}

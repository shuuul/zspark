//! Capture a Responses ID while streaming the backend's raw HTTP reply.

const std = @import("std");

/// A reply's response id taken on the way past: the first `"id"` field of the body's first 64 KiB, its
/// bytes to the closing quote. The scan is a state machine carried across chunks and reads, because a
/// read's or a chunk's edge falls where it falls: the host's own JSON writes `"id": "` with a space
/// after the colon, and an id can be cut in two by the socket. A `"id"` whose value is not a string
/// (or a name that only begins `"id"`, like `"idx"`) puts the scan back to seeking.
pub const Capture = struct {
    state: State = .seek,
    id: [64]u8 = undefined,
    id_len: usize = 0,
    carry: [seek_needle.len]u8 = undefined,
    carried: usize = 0,
    scanned: usize = 0,

    const State = enum { seek, colon, open, in_id, done };
    const seek_needle = "\"id\"";
    const scan_limit = 1 << 16;

    /// Nothing more is scanned: an id is held whole, or this body has none the scan can take.
    fn done(cap: *const Capture) bool {
        return cap.state == .done;
    }

    pub fn scan(cap: *Capture, chunk: []const u8) void {
        if (cap.done()) return;
        cap.scanned += chunk.len;
        if (cap.scanned > scan_limit) {
            cap.state = .done;
            return;
        }
        var rest = chunk;
        while (rest.len > 0 and !cap.done()) {
            const byte = rest[0];
            rest = rest[1..];
            switch (cap.state) {
                .seek => {
                    cap.carry[cap.carried] = byte;
                    cap.carried += 1;
                    if (cap.carried == seek_needle.len) {
                        if (std.mem.eql(u8, &cap.carry, seek_needle)) {
                            cap.state = .colon;
                            cap.carried = 0;
                        } else {
                            std.mem.copyForwards(u8, cap.carry[0 .. seek_needle.len - 1], cap.carry[1..seek_needle.len]);
                            cap.carried = seek_needle.len - 1;
                        }
                    }
                },
                .colon => if (byte == ' ' or byte == '\t') {
                    // the host's own reply has the space here: `"id": "`
                } else if (byte == ':') {
                    cap.state = .open;
                } else {
                    cap.restart(byte);
                },
                .open => if (byte == ' ' or byte == '\t') {
                    // JSON lets whitespace stand between the colon and the value
                } else if (byte == '"') {
                    cap.state = .in_id;
                } else {
                    cap.restart(byte);
                },
                .in_id => {
                    if (byte == '"') {
                        cap.state = .done;
                    } else if (cap.id_len < cap.id.len) {
                        cap.id[cap.id_len] = byte;
                        cap.id_len += 1;
                    } else {
                        cap.state = .done; // nothing of this length is a response id
                    }
                },
                .done => unreachable,
            }
        }
    }

    /// What was scanned was not the id field after all: back to seeking, and the byte that said so may
    /// begin the next `"id"` itself.
    fn restart(cap: *Capture, byte: u8) void {
        cap.state = .seek;
        cap.carried = 0;
        cap.id_len = 0;
        cap.scan(&[1]u8{byte});
    }
};

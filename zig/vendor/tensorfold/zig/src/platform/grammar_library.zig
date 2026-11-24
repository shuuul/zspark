//! The optional grammar runtime's source pin and source-build instruction.
// zspark: v0.2.9, the checkout build.zig's `zig build grammar` gates on (spec 174); upstream 1.2.8 pins v0.2.8
pub const pinned = "v0.2.9";
pub const clone = "git clone --branch " ++ pinned ++ " --recurse-submodules https://github.com/mlc-ai/xgrammar";

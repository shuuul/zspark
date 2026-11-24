# Patches

What differs from upstream (`UPSTREAM`) in this directory: one row a change, in the commit that makes it.
`zig build check-specs` refuses a file whose bytes are not `MANIFEST.sha256`'s without a row here, and a row
whose file is upstream's as it was. The path is from this directory.

| File | What | Why | Proposed upstream |
| --- | --- | --- | --- |

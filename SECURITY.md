# Security policy

## Reporting a vulnerability

**Please do not file a public GitHub issue for a suspected vulnerability.** Report it
privately to the maintainer:

Use the contact information on the [maintainer’s GitHub profile](https://github.com/shuuul).

Include enough to reproduce: the affected version or commit (`zspark version` prints the
commit), the build configuration, a description of the impact, and reproduction steps or a
proof of concept. Do not include model weights, checkpoints, or third-party credentials in
a report.

Acknowledgement should be expected within a few days; reports are handled confidentially,
and a fix and disclosure timing are coordinated with the reporter. There is currently no
bug bounty.

## Scope

zspark is an inference engine and HTTP server (`zspark serve`, `zspark stack`, the
`zspark-router`). In scope: request-parsing bugs, cross-request state leaks, unsafe binding
or token-handling behavior, denial of service a remote client can trigger, and anything that
lets a client reach files, processes, or devices it should not. Operator-side concerns —
serving on an untrusted network without `--api-key`, running untrusted checkpoints — are
documented behavior, but reports that would harden the defaults are welcome.

The vendored TensorFold HTTP host (`zig/vendor/tensorfold/`) is in scope: report here and
it will be forwarded upstream as needed.

## Supported versions

Only the latest `main` (and the newest `vYYYY.M.D` tag) is supported. Earlier tagged
versions receive no security backports; upgrade first.

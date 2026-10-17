# Contributing

This is an early extraction of the developer-facing solver. Discuss substantial API or algorithm changes before implementing them. Small reproducible bug reports, tests and integration examples are welcome.

Build with `cargo build --locked`; check formatting with `cargo fmt --all -- --check`; run `cargo test --locked --all-targets` and `cargo test --locked --doc`. Mandatory synthetic tests work without a local catalogue. Optional tests using external indexes are not evidence of real-sky coverage when those data are absent.

Keep numerical changes separate from packaging and interface changes. Report false solutions, unsolved fields and latency independently. Include the exact revision, index/model identity, hardware and options in benchmark reports. Never publish another person's images without permission.

Code contributions are made under Apache License 2.0. Add a `Signed-off-by` line to confirm the [Developer Certificate of Origin](https://developercertificate.org/). No copyright assignment is requested. Previously published research manuscripts retain their MIT license.

The source tree must remain independent of the author's private checkout. No web UI, server accounts, product activation, production credentials or full-sky index binaries belong here.

Do not synchronize agent instruction files (`AGENTS.md`, `agents.md`,
`CLAUDE.md`, `claud.md` or case variants), private tool configuration or source
Git history. Every selective import needs source revision and original hashes;
do not recursively copy historical directories. Release tooling excludes agent
instruction filenames from source snapshots.

Release/data preparation uses Python 3.11+ with no third-party packages. Run
`python3 -m unittest discover -s tools -p 'test_*.py'` after tooling changes. See
[RELEASE.md](docs/RELEASE.md) for candidate packaging and verification. Integrators
can use this repository's issue tracker and the reproducible bug-report template.

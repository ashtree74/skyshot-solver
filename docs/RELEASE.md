# Preparing a CLI release

Publication requires a separate instruction. The commands below create local
candidates; the workflow stores CI artifacts only and has read-only repository
permissions. It does not create tags, GitHub releases or registry packages.

## Candidate build

Use Python 3.11+ (`python3 --version`; on some Macs use `python3.12` explicitly),
the tested Rust 1.95.0 toolchain and native build tools. CI pins that version;
newer toolchains need their own acceptance and notice review. Install the Rust docs
component so that standard-library notices are available. For rustup:

```sh
rustup component add rust-docs
cargo test --locked --all-targets
cargo test --locked --doc
cargo fmt --all -- --check
cargo check --locked --all-targets --features solver-debug
python3 -m unittest discover -s tools -p 'test_*.py'
```

Choose the actual host target (`rustc -Vv`), for example:

```sh
cargo build --release --locked --target aarch64-apple-darwin
python3 tools/package_cli.py --target aarch64-apple-darwin \
  --binary target/aarch64-apple-darwin/release/skyshot --out outputs/candidate
```

The tool uses offline Cargo metadata, so fetch/build dependencies for the target
first. It packages normal/build dependency license texts (including host build
tools conservatively), a version-specific supplement if the crate omitted texts,
Rust standard-library notices, `Cargo.lock`, source hashes and binary hash.
Check the generated inventory and native runtime dependencies for each target
before publication. A license expression alone is not a full notice bundle.

The archive contains only the executable, selected CLI/install/data docs and
notices. It contains no index, coder, source project, Git history or agent
instructions. Unix candidates use `.tar.gz`; Windows uses `.zip`; every archive
has a `.sha256` file. The binaries are unsigned and not notarized. Inspect
`build.json` and `source-manifest.json` to associate test evidence with actual
files even while the original extraction is uncommitted.

Unpack in a new directory, verify hashes and run the **unpacked executable**:
`--version`, `inspect`, synthetic `solve --wcs`, mixed-success `batch` and the
Python subprocess example. Do not qualify a binary only by running `cargo test`.
The CI matrix performs a packaged synthetic smoke test in addition to Rust tests.

## Acceptance record

The current evidence is recorded in
[verification-2026-10-17.md](verification-2026-10-17.md). Update it after each
relevant run; a workflow configuration or successful cross-compilation does not
establish runtime support.

| Candidate target | Required runtime acceptance |
| --- | --- |
| `aarch64-apple-darwin` | macOS Apple Silicon native tests, release build, packaged solve |
| `x86_64-unknown-linux-gnu` | Linux x86-64 native CI tests, release build, packaged solve; record glibc baseline |
| `x86_64-pc-windows-msvc` | Windows x64 native CI tests, release build, ZIP extraction and packaged solve |

Other targets can be built explicitly by the local tool but are not implicitly
supported. Linux ARM64 container results are evidence for that environment only.
The GitHub matrix must still pass after the source is committed and pushed under
a separate publication instruction.

## Publication gates

- Review the extraction diff and its provenance, excluded files and independent
  build. Preserve Apache 2.0 implementation / MIT manuscripts / upstream terms.
- Complete per-platform checks above, dependency notice review and native runtime
  baseline/signing decisions; retain run links and artifact SHA256.
- Provide a qualified legal real-sky data path with immutable bundles, working
  downloads, model card where needed, and a redistributable qualification image.
- Benchmark this exact source/model/data/hardware: cold startup and warm engine,
  p50/p95, solved/unsolved/ambiguous and false acceptances. Historical manuscript
  numbers remain historical. A one-image smoke test is not a benchmark.
- Have two or three outside developers follow the install path without the
  author's source checkout; review package name availability before any registry
  release (`publish = false` remains in Cargo.toml).
- After separate authorization: freeze source, tag, build final artifacts, review
  their notices/checksums, publish code/CLI/data as their respective rights allow.
  Announcements including Cloudy Nights are a separate publication action.

## Integration order after the foundation

1. Persistent JSONL stdin/stdout requests: request IDs, bounded message sizes,
   one engine per process, per-request errors, documented EOF and cancellation.
   Extend the existing Python subprocess example first.
2. Small C ABI: opaque engine handle, explicit byte lengths and ownership, ABI
   version, errors and panic boundary; solve pixels and star rows, return WCS.
3. C++ RAII and .NET P/Invoke examples against that ABI; test lifetime and encoding
   on native platforms. Choose a native Python binding only after the session
   contract and demand are established.

These are planned integrations, not current commands or bindings.

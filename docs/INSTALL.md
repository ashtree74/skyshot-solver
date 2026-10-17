# Build and install the CLI

Ready-to-use indexes are available at [Skyshot Downloads](https://skyshot.app/downloads/):
Full learned is recommended, with DEEP optional. Public prebuilt CLI binaries are
not released yet. Build from source using a current stable Rust toolchain and the native
linker/build tools (Xcode Command Line Tools on macOS, a C toolchain on Linux,
Visual Studio C++ Build Tools for the Windows MSVC target).

```sh
cargo build --release --locked
cargo test --locked --all-targets
cargo run --release --locked --example synthetic_demo -- demo
cargo run --release --locked -- solve demo/stars.json --index demo/index --wcs demo/solution.wcs
```

Use a new `demo` directory. On Windows, use `target\release\skyshot.exe` in
PowerShell; on macOS/Linux use `./target/release/skyshot`. Add that directory to
PATH, or install from the checkout with `cargo install --path . --locked`.
Rust is needed to build; it is not needed to run a prebuilt candidate.

For a future binary download, choose the exact OS and architecture named in the
archive. Obtain its SHA256 file from the trusted release channel, verify the
archive before extraction, and unpack into a new directory. A matching hash
detects corruption; authenticity depends on where the checksum came from.

```sh
# macOS, in the download directory; replace ARCHIVE with the actual filename.
shasum -a 256 -c ARCHIVE.sha256
# Linux
sha256sum -c ARCHIVE.sha256
```

```powershell
# Windows: compare this digest with the trusted .sha256 file.
Get-FileHash .\ARCHIVE.zip -Algorithm SHA256
Expand-Archive .\ARCHIVE.zip -DestinationPath .\skyshot-new
```

Keep `LICENSE`, `NOTICE`, `third_party/` and the other package files with the
executable. `build.json` records the binary hash, toolchain and target;
`source-manifest.json` identifies the actual source files, including local edits
that a Git commit ID alone would miss. Archives are not byte-for-byte reproducible
builds: toolchains, native linkers and archive timestamps can affect their bytes.

Candidates are currently unsigned and unnotarized. Inspect and build the source
if your OS or organization will not permit an unsigned downloaded executable.
No signing or notarization claim is made for macOS or Windows.

Try `skyshot --version`, then the synthetic example above. For a real image,
follow [DATA.md](DATA.md) to build or verify a compatible external index:

```sh
skyshot inspect /path/to/index
skyshot solve your-frame.fits --index /path/to/index --wcs your-frame.wcs
```

For ready-to-use data, use the checksummed downloader and preset instructions
in [DATA-INSTALL.md](DATA-INSTALL.md). It needs Python
3.11+ only for installation and refuses existing destination directories.

Prebuilt full-sky data is a separate, versioned public preview. `inspect` checks index
structure and coder identity; it does not hash every catalogue file. A successful
synthetic solve does not qualify real-sky accuracy. See [RELEASE.md](RELEASE.md)
for platform evidence and remaining release gates.

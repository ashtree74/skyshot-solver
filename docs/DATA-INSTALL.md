# Install the local Skyshot data candidates

Author, LGQ-GD model author and bundle maintainer: **Adam Jesionkiewicz**.
Catalogue attribution remains with ASTAP / Han Kleijn and ESA Gaia / DPAC.
These are locally prepared candidates, not published downloads. No data URL or
distribution permission is implied. Read each bundle's `LICENSE-DATA.txt` and
`PROVENANCE.json`; non-commercial intended use and incomplete source/terms
records remain explicit. Apache 2.0 on the solver is separate.

## Requirements and package choice

Build Skyshot from its source checkout, or use a matching local CLI candidate:

```sh
cargo build --release --locked
cargo install --path . --locked
skyshot --version
```

The qualified engine version is **0.1.0**. Data files are platform independent;
qualification of these exact candidate bytes is on macOS ARM64. This does not
claim Windows/Linux runtime qualification for this data combination. Installation
uses Python **3.11 or newer**, standard library only. Runtime solving uses neither
Python nor the original development repository.

| Preset | Contents | Use |
| --- | --- | --- |
| `full` (default) | Learned full as `index/`, including its coder | Start here for ordinary astronomical images |
| `complete` | Learned full + DEEP + historical wide companion | Adds narrow-field and wide-field searches; requires substantially more disk space |
| `handcrafted` | Full handcrafted as `index-v1/`, without companions | Reproducible baseline with no learned model |
| `model` | Standalone LGQ-GD as `model/coder.f32` + `coder.json` | Build a new learned index from separately obtained D80 |

Exact archive sizes, SHA256, bundle IDs and coder identities are in `release.json`.
Allow space for both archives and unpacked data. Full learned runtime files are
about 0.88 GB; DEEP about 4.33 GB; wide about 0.022 GB; handcrafted full about
0.88 GB. These are decimal, uncompressed sizes, not peak build-memory estimates.
The wide bundle is the existing R7=7.76 arcmin companion with historical `lite`
profile metadata; it is not the default lite builder preset. The model is about
0.263 MB. Each learned index is self-contained with the same qualified coder.

## Check and install

Obtain the selected local archives, `release.json`, `SHA256SUMS`, `INSTALL.md`
and the accompanying `tools/` directory from the prepared release folder.
Keep their relative layout. Use a trusted copy of the checksums/release manifest;
hashes detect corruption but do not establish origin on their own.

From that folder, verify the complete supplied set on macOS or Linux:

```sh
# macOS
shasum -a 256 -c SHA256SUMS
# Linux
sha256sum -c SHA256SUMS
```

Then install the selected preset into a **new** directory. Replace the destination
with your own absolute path; it must not already exist:

```sh
python3 tools/install_data.py --release release.json --preset full \
  --engine-version 0.1.0 --dest /path/to/skyshot-data-new
```

For full + DEEP + wide together, use `--preset complete` instead. The installer
checks the selected archives against `release.json`, rejects unexpected archive
members, links and path traversal, verifies unpacked runtime hashes and exact
engine compatibility, and records `INSTALLATION.json`. An interrupted or failed
installation has `.incomplete`; do not use it. Retry with a new destination.
No existing installation is overwritten or merged.

On Windows with Python 3.11+ installed, use PowerShell:

```powershell
py -3 tools/install_data.py --release release.json --preset full --engine-version 0.1.0 --dest C:\Skyshot\data-new
```

The same installer verifies the archive hashes on Windows. For manual checksum
comparison, use `Get-FileHash ARCHIVE.tar.gz -Algorithm SHA256` and compare with
the trusted release record. Native Windows solve verification is still pending.

## Solve your own image

```sh
skyshot inspect /path/to/skyshot-data-new/index
skyshot solve your-frame.fits --index /path/to/skyshot-data-new/index --wcs solution.wcs
```

PNG, JPEG and TIFF are also supported. `solve` writes JSON to stdout and
diagnostics to stderr. Check `solution.status`: exit 0 means success, 1 means
unsolved/ambiguous, and 2 means an input/configuration/I/O error. The output WCS
path must be new. An index does not guarantee a solution for every image.

The `complete` preset installs the exact sibling names used by `Engine`:

```text
skyshot-data-new/
  index/         learned full, primary --index path
  index-deep/    learned DEEP companion
  index-phone/   learned wide companion
  INSTALLATION.json
```

With that layout, ordinary solving discovers companions automatically. For a
narrow field you can explicitly request DEEP:

```sh
skyshot solve narrow-frame.fits --index /path/to/skyshot-data-new/index --mode deep --wcs narrow.wcs
```

The handcrafted preset uses `--index /path/to/BASELINE_NEW/index-v1`. Install it
in a separate parent from learned companions to keep comparisons meaningful.
For multiple images, reuse the mapped index with `skyshot batch` or Rust `Engine`.

## Build with the standalone learned model

Install `--preset model` to another new directory, obtain the compatible D80
1476-5 / Gaia DR3 / epoch 2025 catalogue separately, then:

```sh
skyshot build --d80 /path/to/d80 --out /path/to/new-learned-index \
  --profile full --coder /path/to/MODEL_NEW/model/coder
skyshot inspect /path/to/new-learned-index
skyshot solve your-frame.fits --index /path/to/new-learned-index --wcs new-result.wcs
```

Pass the coder filename prefix, without an extension. Never replace a coder in
an existing index: its pair digest includes both exact files. This export adds
author metadata and omits private training paths; numeric weights and quantile
tables are unchanged. New-copy qualification is in `QUALIFICATION.json`.
The full catalogue builder needs substantial RAM even for smaller profiles.

Keep an old installation intact for rollback. Close all engines mapping it
before changing the configured path; never replace mapped files in place.
All archives and weights stay outside Git. Publication requires a separate
instruction from the owner.

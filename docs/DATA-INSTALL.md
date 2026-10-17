# Download and install Skyshot data

Author, LGQ-GD model author and bundle maintainer: **Adam Jesionkiewicz**.
Catalogue attribution remains with ASTAP / Han Kleijn and ESA Gaia / DPAC.
Public preview downloads: **[skyshot.app/downloads](https://skyshot.app/downloads/)**.
Version **2026.10.17.2** is compatible with Skyshot Solver **0.1.0**.
Read each bundle's `LICENSE-DATA.txt`, `PROVENANCE.json` and model card;
non-commercial intended data use and incomplete source/terms records remain
explicit. Apache 2.0 on the implementation is separate.

## Quick start from a fresh checkout

Install Rust and Python 3.11+, then:

```sh
git clone https://github.com/ashtree74/skyshot-solver.git
cd skyshot-solver
cargo build --release --locked
python3 tools/fetch_data.py --preset full --dest data/skyshot-full-2026.10.17.2
./target/release/skyshot solve your-image.fits --index data/skyshot-full-2026.10.17.2/index --wcs solution.wcs
```

**Full learned is recommended.** It includes its exact coder; no separate model
download or D80 build is needed. The downloader checks the public release record
against the catalogue committed in this repository, verifies every archive and
unpacked runtime file, and installs only into a **new** directory. Python is
needed for installation, not for runtime solving. On Windows use `py -3` and
`target\release\skyshot.exe`. Check that your Python is 3.11 or newer.

For the optional DEEP companion, select `full-deep` and another new destination:

```sh
python3 tools/fetch_data.py --preset full-deep --dest data/skyshot-full-deep-2026.10.17.2
./target/release/skyshot solve narrow-image.fits --index data/skyshot-full-deep-2026.10.17.2/index --mode deep --wcs narrow.wcs
```

The default cache is `data/download-cache/skyshot-data-2026.10.17.2`. Repeated
commands reuse verified archives, so adding DEEP does not redownload Full.
Interrupted `.part` transfers resume when the command is rerun. Use `--cache`
to choose another cache location. One process may use a cache at a time; after a
process crash remove only its stale `.download.lock` before retrying. Corrupt
cache files are reported rather than silently replaced. Installed indexes are
never merged or overwritten.

Allow about **2 GB** for Full's archive and installed files together, or **11 GB**
for Full + DEEP. Data files are platform independent. Actual real-image data
qualification is on macOS ARM64; native Linux/Windows CI verifies synthetic
solving and CLI packaging, not this full-sky data combination.

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
| `full-deep` | Learned full + `index-deep/` companion | Optional support for narrow fields, without the wide download |
| `complete` | Learned full + DEEP + historical wide companion | Adds narrow-field and wide-field searches; requires substantially more disk space |
| `handcrafted` | Full handcrafted as `index-v1/`, without companions | Reproducible baseline with no learned model |
| `model` | Standalone LGQ-GD as `model/coder.f32` + `coder.json` | Build a new learned index from separately obtained D80 |

### Field of view (FOV)

Use the **longer side of the image**, not its diagonal. As a first choice, use
Full learned from about **0.3° (18 arcmin)** upwards; add DEEP especially below
that scale. These overlap: Full can solve some smaller fields, and DEEP can
help with a larger field when its stars are too faint or dense for Full.

| Index / model | Real-image checks on 17 October 2026 | Guidance |
| --- | --- | --- |
| Full learned | **0.24°–12.5°** | Recommended starting point |
| Full handcrafted | **0.24°–12.5°** | Same index scales as Full learned; comparison baseline |
| DEEP companion | **0.17°–0.50°** (about **10–30 arcmin**) | Especially useful below about **0.3°** |
| Historical wide companion | **5.3°–12.5°**, three images | Wider fields remain experimental |
| Standalone LGQ-GD coder | No independent FOV range | Coverage depends on the index built with it; it cannot solve an image by itself |

These are the extrema of correctly solved, reference-checked images in the local
macOS ARM64 qualification of the downloaded `2026.10.17.2` data. The Full ranges
come from the 90-image full-only blind comparison (86 solved per arm); DEEP's
range comes from the five DEEP solutions in the complete CLI run, which allowed
FITS hints. The separate explicit-DEEP check covered three fields from 0.17°
to 0.24°. This is the existing positive-image corpus, not an independent holdout.
Raw images and per-image references remain private.

The ranges describe tested images, **not hard coverage limits or a guarantee**
for every field between them. Star density, detectable star count, image quality
and distortion also matter. The default main search permits 0.15°–75° and the
DEEP search 0.03°–0.5°; these are search bounds in the engine, not measured
capabilities of every distributed index.

Exact archive sizes, SHA256, bundle IDs and coder identities are in `release.json`.
Allow space for both archives and unpacked data. Full learned runtime files are
about 0.88 GB; DEEP about 4.33 GB; wide about 0.022 GB; handcrafted full about
0.88 GB. These are decimal, uncompressed sizes, not peak build-memory estimates.
The wide bundle is the existing R7=7.76 arcmin companion with historical `lite`
profile metadata; it is not the default lite builder preset. The model is about
0.263 MB. Each learned index is self-contained with the same qualified coder.

## Check and install

For manual or offline installation, obtain the selected archives, `release.json`, `SHA256SUMS`, `INSTALL.md`
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
py -3 tools/install_data.py --release release.json --preset full --engine-version 0.1.0 --dest data\skyshot-new
```

The same installer verifies the archive hashes on Windows. For manual checksum
comparison, use `Get-FileHash ARCHIVE.tar.gz -Algorithm SHA256` and compare with
the trusted release record. Native Windows CI verifies synthetic solving; real-image
qualification of this full-sky data combination remains macOS ARM64 only.

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
All archives and weights stay outside Git; downloads are versioned HTTPS files
linked from the repository. Never change files under an existing published ID.

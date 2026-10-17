# Verification record — 2026-10-17

This record starts with local extraction and release preparation; the final
sections record the first source push and native CI findings. The solver
mathematics have not been rewritten. Candidate archives contain a
source file inventory/hash; the original Git HEAD alone does not identify them.

## Initial local acceptance, before the source push

| Check | Observed result |
| --- | --- |
| macOS ARM64, Rust 1.95.0, native | 147 Rust tests passed (`--all-targets`); release build passed |
| Rust docs / formatting / solver-debug | Documentation with warnings denied, formatting and all-target feature check passed; zero doctests defined |
| Data bundle tooling, Python 3.12 | 11 tests passed, including same-size corruption, compatibility, path traversal, symlinks, pending-terms packaging with notices, missing-notice rejection, archive round trip and unsafe TAR rejection |
| Import provenance | All 39 recorded original source hashes match the fixed source commit; extraction edits remain separate |
| macOS packaged CLI | SHA256 verified, archive unpacked into a new directory; expected synthetic coordinates, WCS/no overwrite, inspect and mixed-success batch passed; Python subprocess example passed |
| Linux ARM64 container | 147 tests, release build, synthetic CLI solve/WCS and unpacked candidate smoke passed, with only selected sources and no catalogue or source project mounted; 10 tooling tests passed on Python 3.11.2 |
| Linux x86-64 / Windows x64 native CI | Not run; workflow exists, no runtime support claim |
| Real-sky solve / rebuilt handcrafted index | Local M66 FITS solved with existing learned full index and a new handcrafted lite index; 417 matches, RMS 0.15836 px, same WCS |

Host: macOS 27.0, Apple Silicon (`aarch64-apple-darwin`), Rust 1.95.0
(`59807616e`), Cargo 1.95.0. No external index environment variable was supplied
to the mandatory test suite. Some unit tests can return early when local indexes
are absent; the mandatory synthetic CLI and FITS tests do execute solves.

Linux environment: official `rust:1.95.0-bookworm` ARM64 image, pulled manifest
digest `sha256:6258907abe69656e41cd992e0b705cdcfabcbbe3db374f92ed2d47121282d4a1`,
Rust 1.95.0, `aarch64-unknown-linux-gnu`, Debian bookworm/glibc 2.36, OrbStack
kernel 7.2.9. The selected
snapshot contains neither `.git` nor `target`, agent instructions or external
data. Cargo fetched dependencies into the fresh container; the build and tests
were run there. This is a Linux ARM64 container acceptance, not a Linux x86-64
runner or a Windows result. The synthetic solution centre was RA
83.7999998834°, Dec −5.4000000647° (expected 83.8°, −5.4°).

The new handcrafted build decoded 264,830,158 stars from 1,476 D80 tiles and
produced 14,481,534 records in the `lite` profile, with no coder. It wrote only a
new temporary directory. The builder reported about 37 seconds in this one run;
this is not a build-time benchmark or a low-memory claim. The internal catalogue
fingerprint was `9f74976668a9832e90ee057b635e58743db8891c53248e9e0268439484efa334`
(names/sizes, not content hashes).

The actual 1,476 input tile content hashes are retained in
`provenance/d80-local-inputs.json`: 1,332,217,055 bytes, aggregate inventory digest
`8336f5ee28371c16e723d7232543ca54b3104d8e73e3d5895fb1e72a7c9d035e`.
This identifies the local build inputs; the original download archive/terms are
still missing. The M66 input SHA256 was
`1667c97f3da7b342932cf10e3ce160b27573eccc52080d85a105cf5eab8cf673`.
The exact builder source/Cargo hashes, output index checksums and local solve
receipt are recorded in `provenance/local-lite-build.json`.

The local M66 engineering check returned RA 169.9740003657°, Dec 13.2689664177°
with each index. Matching two configurations is a consistency check, not an
independent accuracy reference. No input photo or catalogue was copied into the
repository and their redistribution rights are not asserted. The existing full
index's 9 runtime files (882,960,559 bytes) were hashed read-only and verified
against `provenance/existing-full.bundle.json`, explicitly marked `pending`.
The existing full index also solved the same M66 input when addressed through a
temporary isolated directory with no discoverable wide/DEEP siblings: 417 matches.

Tooling note: the system `python3` on this Mac is 3.9.6; the release/data tools
require 3.11+. Their tests and packaging were run with Python 3.12, and the tools
now emit an explicit version requirement on older interpreters.
The Linux run exposed the missing safe-TAR filter in early Python 3.11; extraction
now copies only validated regular files/directories and rejects traversal and
links, without relying on that later stdlib API. Both positive and rejection
tests passed. Candidate packaging also passed with rustup's documentation layout
using the reviewed Rust 1.95.0 root-license supplement.

The macOS Cargo notice inventory contains 89 normal/build dependencies, plus the
Rust library notices. `otool -L` showed only system `libiconv` and `libSystem`.
Linux `ldd` showed system `libgcc_s`, `libm`, `libc` and the AArch64 loader; those
system libraries are not bundled. Per-platform notice/runtime review remains a
publication gate. Workflow syntax passed local `actionlint`; the workflow itself
has not been run on GitHub.
The planned Windows target's Cargo license inventory was also generated from
downloaded locked dependencies. This checks notice collection only; no Windows
binary was built or executed.

No Windows runtime or GitHub runner was used in this local verification. No
package, release, tag, code or announcement has been published. The broader
real-image comparison and subsequent data-candidate qualification are described
below; these do not establish a global false-acceptance rate. See
[RELEASE.md](RELEASE.md) for the remaining release gates.

## Follow-up: actual solving and local packaging policy

The unpacked macOS candidate was run again on the real M66 FITS with the newly
built `data/local-lite-20261017` index: exit 0, `solved`, 417 matches,
RMS 0.1583624422 px, RA 169.9740003657°, Dec 13.2689664177°, WCS written.
This confirms real-image solving through the distributed executable path.

The owner confirmed non-commercial data use with openly listed sources,
dependencies, attribution and applicable terms. Removed the extra requirement
for a `cleared` status before **local** packaging. Pending documentation remains
recorded verbatim and notices/provenance files are still required. Compatibility,
checksum, path, immutability and no-overwrite checks remain in force. Eleven
tooling tests passed after this change. The historical Linux tooling run above
had ten tests; this follow-up was verified locally on macOS, without rerunning
the unchanged Rust solver or claiming a new Linux/Windows runtime result.

## Follow-up: real-image comparison and local data candidates

The subsequent paired comparison used 90 real images from the existing local
research corpus, excluding seven synthetic entries, with 89 reference WCS.
Both full indexes used identical catalogue geometry and explicit main-index
search without learned companions or FITS hints. Both solved 86 images
correctly and left four unsolved, including the one without a reference. There
were zero wrong-field or inaccurate accepted solutions in this positive-image
corpus. Repeated fields, historical selection, shared catalogue ancestry and
one reference without independent confirmation limit interpretation.

Timing aggregates used each image's median of five measured repeats after
warmup, then p50/p95 across images. On the shared Apple M5 Max with four Rayon
workers, solver-only times on the 86 common correct images were 3.39/9.32 ms
learned and 4.50/34.73 ms handcrafted. Pipeline times across all 90 were
24.62/87.52 ms and 25.97/116.59 ms respectively. The machine carried substantial
background CPU load; these are local experimental results, not clean-room
performance promises. Private corpus details, raw results and exact run sources
remain in ignored local outputs; `bench/real_images_ab.py` documents the method.

Local data version `2026.10.17.1` prepares four index bundles (learned full,
DEEP, historical wide, full handcrafted) and a separate LGQ-GD coder. The new
metadata records author **Adam Jesionkiewicz** and keeps catalogue attribution
separate. It omits private paths; archive headers omit account names and local
ownership/timestamps. Numeric weights, quantiles and index record files were
not changed. The coder's new exact pair SHA256 is
`b2a991cfbf658e406f2d7a8f060fb31a39a7371885064d6306ffbbdd9983569a`.

`check_coder_export` verified bit-identical catalogue/image keys, embeddings,
noise/CDF/logodds and probes on 512 deterministic configurations with both image
parity branches. New full copies then produced identical statuses and WCS on
all 90 images compared with the earlier benchmark. The complete learned
companion layout through ordinary CLI auto mode solved all 89 referenced images
correctly; one image without reference remained unsolved. Of these 90 attempts,
36 used FITS hints and five successful solutions used DEEP. This CLI check is
kept separate from the blind comparison above. Three explicit narrow-field DEEP
checks also passed. The wide copy passed three standalone real-image checks at
5.28–12.52 degrees; wider coverage is not qualified. Maximum accepted reference
error was 1.002 px on the inner grid and 2.542 px on the outer grid.

A fresh learned lite build with the standalone export decoded 264,830,158 stars
from D80 and wrote 14,481,534 records in 53.88 seconds (one observation, not a
benchmark). It solved M66 correctly with 0.048 px maximum inner reference error
and 0.078 px maximum edge error. All 1,476 D80 tile hashes still matched the
retained inventory afterward.

The installer and packaging tools passed 17 local Python tests, covering
checksums, metadata identity, compatibility, archive allowlists, author
preservation, omission of local TAR ownership, no overwrites and failure
markers. All 147 Rust tests and formatting also passed after adding the coder
qualification example. These new data checks are macOS ARM64 observations;
the earlier Linux test does not qualify the new bundle combination on Linux or
Windows. Installation instructions are in [DATA-INSTALL.md](DATA-INSTALL.md).

All five archive SHA256 checks passed. The supplied installer sources were run
from the release folder for all four presets (`full`, `complete`, `handcrafted`,
`model`), into fresh directories. Repeated qualification on unpacked runtime
files reproduced the results above, and all installed runtime hashes matched
the staged copies. A separate default-full CLI solve of M66 returned `solved`,
417 matches and RMS 0.1583624422 px. Source runtime files were content-hashed
again and remained unchanged; canonical DEEP retained its `uchg` protection.

The ignored local `outputs/data-release-2026.10.17.1/` folder holds the five
archives, detached manifests, `SHA256SUMS`, installation tools, original exact
preparation/qualification sources and receipts. `INSTALLATION-VERIFICATION.json`
and its own checksum record the post-packaging test without changing the
immutable archive bytes. The source-review catalogue is
[provenance/data-candidate-2026.10.17.1.json](../provenance/data-candidate-2026.10.17.1.json).
Nothing was published or added to Git staging.

## Final source review before the first implementation push

Prepared the selected source tree for `main`, excluding all data archives,
weights, images, agent instruction files and private configuration. The two
previously published PDFs remain unchanged and their original MIT license is
preserved in `research/LICENSE`. Removed a historical absolute generator path
from the synthetic panorama fixture and made its generator record a relative
path; regeneration preserved all numeric values. The CLI candidate now also
includes `DATA-INSTALL.md`.

A clean snapshot exported from the Git index, without `.git`, external indexes
or the source project, passed all 147 Rust tests, the all-target `solver-debug`
check, Rust documentation with warnings denied, release build, synthetic example
and solve, CLI packaging and unpacked candidate smoke checks. This reused the
local Cargo dependency cache, not private catalogue data. All 17 Python tooling
tests, Rust formatting and staged whitespace checks passed. Local Markdown
links resolved. The 103-file source tree was about 1.76 MB including the two
existing PDFs; no indexed file exceeded 1 MB. The GitHub Actions commit pins
were verified against their upstream repositories. Native GitHub CI results
must be attributed to the actual pushed commit when they are available.

## First source push and native CI

Implementation source was pushed to `main` as `d755ca98b1803cb4758036c673db83e4d6bde9d2`.
Data archives and weights remain local and ignored. The
[first native CI run](https://github.com/ashtree74/skyshot-solver/actions/runs/39056005119)
passed the complete macOS and Linux jobs, including tests, release build,
synthetic solving, packaging and unpacked candidate checks. Windows passed the
Python tools and formatting but exposed a compile error: the optional
`Index::lock_in_memory` called a `memmap2::Mmap::lock` method available only on
Unix. No Windows solving result is inferred from that failed job.

The correction preserves Unix behavior and returns `ErrorKind::Unsupported`
explicitly on other platforms. Normal index mapping and solving do not call
this optional method. A Windows regression test requires that error and then
solves the synthetic field successfully. Local targeted CLI tests, all-target
feature check, Rust documentation and formatting passed; actual Windows
execution is delegated to the successor native CI run for the correction.
The post-fix diff review found no additional issue in this boundary. See the
[workflow history](https://github.com/ashtree74/skyshot-solver/actions/workflows/ci.yml)
for the result associated with each exact commit.

## Public Downloads acceptance

Data preview `2026.10.17.2` is available at [Skyshot Downloads](https://skyshot.app/downloads/).
A fresh public GitHub checkout built the CLI and synthetic example; all five
public archives were downloaded, SHA256 checked and installed into new directories.
Full is recommended and `full-deep` is optional. The small standalone installer
ZIP also passed. Full's interrupted download resumed from a public 1 MiB partial.

The fresh CLI solved M66 and three narrow DEEP fields. Requalification of the
installed downloads on 90 real images preserved prior status/WCS results:
full-only blind learned and handcrafted each 86 correct / 4 unsolved, complete
CLI 89 correct / 1 unsolved with 36 FITS-hinted attempts. No incorrect accepted
field occurred in this set; no general false-positive rate is inferred.

[CI run 39060058544](https://github.com/ashtree74/skyshot-solver/actions/runs/39060058544)
passed on native Linux x86-64, macOS ARM64 and Windows x86-64, including 147 Rust
and 25 Python tests, release build, docs and packaged synthetic smoke tests.
Full-sky real-image data qualification is macOS ARM64 only. Archive and installation evidence is in the
[public verification receipt](../provenance/data-releases/2026.10.17.2/public-verification.json)
and [installed-data qualification](../provenance/data-releases/2026.10.17.2/installed-qualification.json). Source photographs and references
were not published.

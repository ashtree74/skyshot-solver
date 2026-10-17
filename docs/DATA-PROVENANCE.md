# Data and model provenance review — 2026-10-17

Project direction confirmed by the owner on 2026-10-17: **non-commercial data
use with openly listed sources, dependencies, attribution and applicable terms**.
Continue preparation on that basis; this inventory is not a blanket project
blocker or an extra approval process. It records inspected facts and missing
documentation. Apache 2.0 covers the implementation; external data and weights
retain their own terms. The non-commercial data-use scope does not restrict the
Apache-licensed code.

## Catalogue

The Rust D80 reader requires the ASTAP 1476-5 format, 1,476 files named
`d80_NNNN.1476`, a 110-byte header with a five-byte record marker, `GAIA DR3` and
`Epoch=2025` in the description. V50, older epochs and arbitrary Gaia tables are
not interchangeable inputs. The CLI builder uses this reader directly.

The [ASTAP author page](https://www.hnsky.org/astap.htm), inspected on 2026-10-17,
provides D80 downloads and documents the 1476-5 layout in Appendix 3. It describes
the density-limited catalogue construction. These facts establish an acquisition
and format reference, not permission to mirror the archive or redistribute
derived Skyshot indexes. The license of the ASTAP program is not evidence of the
catalogue's terms.

The [ESA Gaia data license page](https://www.cosmos.esa.int/web/gaia-users/license)
could not be retrieved during this review (HTTP 451). The
[Gaia DR3 release page](https://www.cosmos.esa.int/web/gaia/dr3) links to the
release-specific credit requirements. No license is inferred from secondary
search snippets. Even a confirmed Gaia license would need to be considered
together with the terms of the particular D80 compilation used here.

Before a data release, retain the exact D80 archive URL, version/date, SHA256,
per-tile content hashes, included terms/readme and the applicable upstream
permission or other documented basis for redistributing the compilation and its
derived indexes. Record Gaia release/epoch, transformations and attribution.
Record unresolved details explicitly rather than substituting a guessed license.
Local builds, tests and packaging continue while that inventory is completed.
No message to the producer has been sent.

## Existing LGQ-GD coder and the separate local export

The source project's `models/lgq_gd_u30.json` declares:

| Item | Recorded value |
| --- | --- |
| Export | `skyanchor2-learned-coder-lgq`, version 1, float32 little endian |
| Architecture | GD feature set, element width 16, global width 2, hidden widths 64/128, embedding width 8 |
| Quantization | `[4, 3, 3, 3, 4, 5, 5, 5]` bits, 32 total |
| Parameters | 32,424, plus quantile tables |
| Training pairs | 3,210,419 (metadata declaration, not independently reproduced) |
| Training | 10,000 steps, batch 8,192, seed 0, learning rate 0.002, weight decay 0.0001 |
| Recorded device | NVIDIA GeForce RTX 4090 |

The existing full learned index declares coder pair digest
`f8fbce6c7c5736f36c60f92431d0e3918c5d31361a14f4e31536fbe688db88b7`.
That digest covers weights **and the original JSON bytes**. The original model
metadata contains absolute training-machine paths. It is not copied into release
metadata or Git. The local `2026.10.17.1` candidate uses a new
operational-field-only JSON export, adding author **Adam Jesionkiewicz**, model
name and export ID. Weights and quantile tables remain byte-identical.
The new pair digest is
`b2a991cfbf658e406f2d7a8f060fb31a39a7371885064d6306ffbbdd9983569a`.
Copied index manifests reference this digest; canonical originals remain intact.
`check_coder_export` and real-image qualification test the new copies, with
receipts included in the separate local data candidates. This is metadata
sanitization and requalification, not a reproduced training run. See
[DATA-RELEASE.md](DATA-RELEASE.md) and [DATA-INSTALL.md](DATA-INSTALL.md).

The training entry points were located and read selectively:
`lab/train/gen_fields.py`, `build_trainset.py`, `train_lgq.py`, `export_final.py`,
and their `skyanchor_lab.learned` modules. The inspected path generates simulated
fields tied to catalogue records, builds pairs/negatives, trains, folds whitening
into the exported embedding and builds quantile tables. This does not prove
the complete lineage of the existing shards/checkpoint or rule out additional
inputs. Training has not been reproduced in this repository.

The local weight export and qualification do not establish reproducible training.
Before publication or claiming training reproducibility, complete:

- A model card naming the owner, weight terms, exact checkpoint/export hashes,
  limitations, intended use and matching index bundles.
- An inventory of catalogue and all other training/calibration/evaluation inputs,
  their rights, hashes and split membership; evidence for the declared pair count.
- A selective, reviewed import of the minimal generator/features/trainer/exporter
  with dependency versions, source hashes, seeds and executable relative-path
  commands. Do not copy historic run directories or agent instruction files.
- Preserve the new sanitized export's fixed key/probe vectors, new-copy index
  qualification and real-image acceptance results. Current export qualification
  compares the two coders in Rust; cross-language training/export parity remains
  to be added with the minimal Python reproduction path.

Handcrafted v1 is the current independent build route. It requires no learned
weights and must not inherit the historical LGQ-GD timing claims.

## Real images and current decision

Local read-only solves may use existing private fixtures for engineering checks.
Their presence on disk is not consent to redistribute them. No such image has
been imported. A public qualification set still needs permission, input SHA256,
reference WCS, tolerances and both success and rejection cases.

Code/CLI and non-commercial data preparation continue, including local packaging.
`pending` identifies incomplete terms/provenance documentation; it does not stop
local packaging and is not automatically promoted to a legal approval. Packages
include source/terms/attribution documents. Publication still requires the owner's
separate instruction; no archive URL or external license grant is invented.

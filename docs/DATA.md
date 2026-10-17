# Catalogues, indexes and models

Code and data are separate distributions. This source snapshot does not contain full-sky catalogues, generated production indexes, atlas packs, model weights or users' photographs. The synthetic demo and tests create artificial data locally and need no catalogue download.

Local candidate archives now exist separately from Git: learned full, DEEP and
historical wide, full handcrafted, and a sanitized standalone LGQ-GD coder.
Follow [DATA-INSTALL.md](DATA-INSTALL.md) to select a preset, check hashes, install
into a new directory and solve your own image. Public hosting is not configured
and nothing has been published. Each learned index contains its exact coder;
installing the standalone model is unnecessary for ordinary solving.

For real images, supply an existing compatible index directory. The current readers support the ordinary and DEEP formats used by the extracted solver; the manifest records the version, profile, declared file sizes and key type. A learned-key index also contains its matching coder pair and digest. Do not mix a coder from another build with an existing index.

```text
data/
  index/         main index
  index-deep/    optional DEEP companion
  index-phone/   optional wide-field companion
```

No canonical data is moved by this extraction. The old project's primary `work/` remains the owner of its published indexes. Use read-only paths for local comparison; never rebuild into a directory used by a running process.

Current scope: **non-commercial data use with openly recorded sources,
dependencies, attribution and applicable terms**. The [provenance inventory](DATA-PROVENANCE.md)
separates inspected facts from missing documentation; it does not block local
builds or packaging. The [bundle protocol and tooling](DATA-RELEASE.md) define immutable
versions, exact engine compatibility, SHA256 verification and local packaging.

## Building your own index

The builder accepts a separately obtained ASTAP D80 catalogue. Start from the
[catalogue producer's download page](https://www.hnsky.org/astap.htm) and retain the
exact archive and its terms. This reader specifically requires all 1,476
`d80_NNNN.1476` tiles in the 1476-5 format, Gaia DR3, epoch 2025. Extract the
upstream archive into a new directory and pass the directory containing those
tiles directly (not an enclosing download folder). Other ASTAP catalogues and
older D80 epochs are not compatible substitutes. Obtain it under its applicable
terms. Rights to the catalogue and derived data must be resolved before
redistributing an index; Apache on this implementation does not grant those rights.

```sh
mkdir -p data
skyshot build --d80 /path/to/d80 --out data/index-new --profile full
# Optional, with a separately supplied matching model:
skyshot build --d80 /path/to/d80 --out data/index-learned --profile full --coder /path/to/lgq_gd_u30
```

The output directory must not exist. The builder reserves it before writing and leaves a failed partial build for inspection. Validate every manifest entry and the coder, then perform a known solve before using a new index. Never replace mapped files in place.

Build profiles exposed by this CLI are `lite`, `standard`, `full` and `deep`.
Every profile first decodes the entire D80 input in memory; `lite` reduces index
records, not the catalogue-loading memory requirement. Plan for substantial RAM
and temporary storage rather than assuming a small-output profile is a low-RAM
builder. The CLI does not currently expose the historical wide/phone preset.
The internal catalogue fingerprint is based on tile names and sizes; retain
independent input content hashes for reproducibility.

After building:

```sh
skyshot inspect data/index-new
skyshot solve your-frame.fits --index data/index-new --wcs your-frame.wcs
```

Check `solution.status` and exit status. Unsolved input is a valid outcome; a
single catalogue/profile does not guarantee a solution for every field. A known
local M66 image has been checked against a newly built handcrafted index in the
[verification record](verification-2026-10-17.md); the image is not distributed.

Without `--coder`, the CLI builds the handcrafted v1 variant. This is useful for development and reproducibility, but must not be presented as the measured LGQ-GD configuration in the manuscripts.

## Before publishing a data release

- Record catalogue and training input provenance, permissions and attribution.
- Prepare versioned full/wide/deep packages outside Git, with byte sizes, SHA256, compatible engine versions and model identities.
- Publish a model card and training/export recipe with the exact weights when their provenance is ready.
- Provide a verified download/install path, including interrupted download handling and a known real-image test.
- Keep the tiny synthetic example independent of all external downloads.

Local archives, checksummed installation and new-copy real-image qualification
are prepared. Public delivery, complete terms/input lineage, an openly
redistributable real-image qualification fixture and remaining native platform
checks are still release work. A source build and a synthetic solve alone are
not a ready public distribution of real-sky data.

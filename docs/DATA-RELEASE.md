# Data bundle protocol (version 1)

Public preview **2026.10.17.2** is served from [Skyshot Downloads](https://skyshot.app/downloads/).
The [pinned release catalogue](../provenance/data-releases/2026.10.17.2/release.json)
contains exact archive sizes, SHA256, coder identity, engine compatibility and presets.
Full learned is recommended; `full-deep` adds the optional DEEP companion.
The following protocol also documents the earlier local candidates. The owner has confirmed non-commercial data use with openly recorded
sources, dependencies and terms. See [DATA-PROVENANCE.md](DATA-PROVENANCE.md)
for the inventory; it is not an additional project approval gate.

Code version, index format version, learned coder version and bundle ID are
independent. Use immutable bundle IDs such as `skyshot-full-v1-2026.10.17.1`.
A change to any bytes requires a new bundle and new checksums. Never replace
files in a mapped index, and never alter the canonical source indexes.

## Earlier local candidates

Version `2026.10.17.1` contains four separate index archives (`full-lgq`,
`deep-lgq`, historical `wide-lgq`, `full-v1`) and one standalone `lgq-gd` model.
All remain outside Git. `release.json` pins their byte sizes, archive SHA256,
detached manifest hashes, exact engine version and installation presets.
`SHA256SUMS` additionally covers the release record, qualification receipt,
instructions and supplied installer sources. The small reviewable catalogue is
retained in [provenance/data-candidate-2026.10.17.1.json](../provenance/data-candidate-2026.10.17.1.json);
archive bytes remain in ignored `outputs/data-release-2026.10.17.1/`.
See [DATA-INSTALL.md](DATA-INSTALL.md).

`tools/prepare_data_release.py stage` reads selected runtime files from a
caller-supplied source checkout, hashes them before/after, creates new copies,
omits private catalogue/training paths and adds author **Adam Jesionkiewicz**.
The sanitized coder preserves every operational field and every weight byte;
its new pair SHA256 is
`b2a991cfbf658e406f2d7a8f060fb31a39a7371885064d6306ffbbdd9983569a`.
Only copied index manifests name that new digest. Catalogue attribution remains
separate. Archive headers omit local account names, ownership and timestamps.

Before `pack`, qualify the exact staged bytes. `check_coder_export` compares
catalogue/image keys, embedding/noise/CDF/logodds and probe costs with the old
coder on 512 fixed configurations, including both image parity branches.
`bench/qualify_data_release.py` checks the two full copies against the prior
90-image results and reference WCS, then exercises ordinary CLI companion
discovery, explicit DEEP and the wide copy. Private images and raw results are
excluded from the sanitized receipt. These checks do not reproduce training or
establish a general false-positive rate. Wide coverage beyond the three tested
images (5.28–12.52 degrees) remains unqualified.

Preparation uses Python 3.11+ stdlib. Qualification also needs the benchmark's
NumPy/Astropy/psutil environment. Example commands, with new local output directories:

```sh
python3 tools/prepare_data_release.py stage --source READ_ONLY_SOURCE \
  --out outputs/data-release-new --version YYYY.MM.DD.N
cargo build --release --locked --example check_coder_export --example benchmark_images
target/release/examples/check_coder_export --original READ_ONLY_SOURCE/work/index/coder \
  --exported outputs/data-release-new/staging/MODEL_BUNDLE_ID/coder \
  --out outputs/data-release-new/coder-parity.json
python3 bench/qualify_data_release.py --release-dir outputs/data-release-new \
  --plan PRIVATE_REFERENCE_PLAN --baseline PRIVATE_PRIOR_PAIRED_RESULTS \
  --out outputs/qualification-new
python3 tools/prepare_data_release.py pack --source READ_ONLY_SOURCE \
  --out outputs/data-release-new --qualification outputs/qualification-new/qualification.json
```

The stage operation imports only a fixed runtime allowlist; it never copies
source directories recursively, images, logs, Git history or agent instructions.
The pack step requires a passing receipt covering every staged runtime hash.
After packaging, install archives into fresh directories and repeat qualification
with `--installed INSTALL_PARENT` containing `complete/`, `handcrafted/` and
`model/` presets. Keep the separate installation receipt with the candidates.
No upload or remote publication is implemented by these preparation commands.
For an explicitly authorized public preview, stage with `--public-preview`,
qualify and pack as above. Keep released archives immutable and distribute them
separately from Git, together with manifests, notices and installer helpers.
Public availability is separate from the recorded upstream terms status; the
flag never changes `pending` to `cleared`. See [DATA-INSTALL.md](DATA-INSTALL.md)
for the published downloads and installation instructions.

`tools/data_bundle.py` uses Python 3.11+ and the standard library. It reads selected
runtime files, never recursively imports a directory and never writes into the
index. `create`, `verify` and local `pack` work while terms documentation is
`pending`. Packaging requires source/attribution documents and preserves the
recorded status verbatim. It neither publishes files nor changes that status to
`cleared`. A checksum cannot establish rights or a model's training history.

## Prepare and qualify a bundle

1. Copy `provenance/data-release.template.json` to a new local metadata file.
   Fill in the input archive SHA256, input file checksum inventory, applicable
   terms and evidence, exact build command, source revision/tree identity and
   toolchain. A learned build also needs model provenance and its exact coder
   digest. Use public logical paths in new metadata.
2. Build into a **new** directory, using the handcrafted default unless a
   separately qualified coder is available. Hash the D80 tile contents as well as
   the downloaded archive. The internal `catalog_fingerprint` hashes tile names
   and sizes, **not contents**; it is not an integrity check.
3. Create the detached manifest outside the index:

   ```sh
   python3 tools/data_bundle.py create --index data/index-new \
     --metadata /path/to/metadata.json --out /path/to/bundle.json
   python3 tools/data_bundle.py verify --index data/index-new \
     --manifest /path/to/bundle.json --engine-version 0.1.0
   skyshot inspect data/index-new
   skyshot solve /path/to/known-image.fits --index data/index-new
   ```

4. Record expected coordinates, tolerance, measured status and options for each
   known real-image solve. Verify failure cases too. Add only engine versions
   actually qualified against this bundle to `engine_versions`; it is an exact
   allowlist, not a guessed future semver range. `verify` checks the supplied
   version string; compare it with `skyshot --version` yourself.
5. Record source/derived-data terms, model provenance and the non-commercial
   intended use. Review every released JSON for local paths or sensitive metadata.
   Put attribution, applicable terms and any explicitly unresolved items in
   `LICENSE-DATA.txt`, source/build/qualification evidence in
   `PROVENANCE.json`, and, for learned indexes, `MODEL-CARD.md` in a separate
   evidence directory. Keep `redistribution.status` as an honest inventory record;
   `pending` does not prevent local packaging. Do not invent a license or mark
   undocumented terms as confirmed. Regenerate the detached manifest after edits.
6. Package locally:

   ```sh
   python3 tools/data_bundle.py pack --index data/index-new \
     --manifest /path/to/bundle.json --engine-version 0.1.0 \
     --evidence /path/to/evidence --out /path/to/skyshot-data.tar.gz
   ```

The archive contains one root named by `bundle_id`, the runtime files,
`bundle.json` and the evidence files. A sibling `.sha256` covers the archive.
The detached manifest covers every runtime file, including the internal
`manifest.json`, optional deep-star tables and both coder files. Source images,
training shards, logs, agent instructions and arbitrary directory contents are
excluded. Human review of the three evidence files is still required.

The index reader's historical `skyanchor2-index` name is deliberately retained
for compatibility. Supported reader layouts are v1 (handcrafted, and legacy
learned files with an explicit key type), v2 learned, and v3 DEEP. A learned
coder's identity is SHA256 of `coder.f32` followed by the exact `coder.json`
bytes. Editing metadata changes that identity even if the numeric weights stay
the same. New sanitized exports require new validation and bundle identity;
never patch a protected production index in place.

## Download and installation contract

The proposed delivery is **public static HTTPS object storage**, outside Git,
with one immutable prefix per bundle. No account or download API key is needed
by the solver user. This accommodates multi-gigabyte profiles and resumable
downloads without adding a service to this repository. Keep release manifests
and their archive checksums under source review; store archive bytes in the
separate data host. No bucket, account, domain or public download has been created.

```text
indexes/<bundle_id>/bundle.json
indexes/<bundle_id>/<bundle_id>.tar.gz
indexes/<bundle_id>/<bundle_id>.tar.gz.sha256
indexes/<bundle_id>/LICENSE-DATA.txt
indexes/<bundle_id>/PROVENANCE.json
indexes/<bundle_id>/MODEL-CARD.md       # learned bundles only
```

Require HTTP Range support, fixed content length, stable public URLs and no
in-place overwrites from the chosen host. A future friendly "recommended" pointer
can identify a bundle, but installation must pin its immutable ID and trusted
checksums. Mirror only reviewed releases with the same bytes and terms. Provider,
retention/cost policy and the actual host remain a publication decision; there
are no invented release URLs in this repository.
A published release must offer a stable archive URL, byte size, archive SHA256,
detached bundle manifest, data terms and engine compatibility table. Prefer
resumable HTTPS downloads. With the actual URL supplied by that release:

```sh
curl --fail --location --continue-at - --output skyshot-data.tar.gz RELEASE_URL
```

Verify the archive checksum against the trusted release record before extracting
into a new staging directory (macOS `shasum -a 256 -c`, Linux `sha256sum -c`,
Windows `Get-FileHash`). List the archive first; accept only the documented
single root and regular runtime/evidence files. Use a current archive tool that
rejects absolute paths, traversal, links and special devices. Verify the unpacked
files with `data_bundle.py verify`, then run `skyshot inspect` and the published
qualification example. Promote the **whole new directory** by changing the
configured path after closing old `Engine` instances. Do not merge or overwrite
an installed index. Keep the old directory until rollback is no longer needed.

Full, DEEP and wide data should be separate versioned downloads so that a basic
solve does not require every profile. Each bundle must specify its install name
and qualified companions. The current engine discovers sibling `index-deep` and
`index-phone` (`index-phone-v1` for a main name ending in `-v1`). A bundle
directory named by its version is safe for standalone use; enable companions
only in a separate installation directory with those explicit names and a
qualified combination. Similar filenames or matching coder hashes alone do not
qualify a combination. The current CLI builds lite/standard/full/deep; it does
not expose a wide/phone build preset.

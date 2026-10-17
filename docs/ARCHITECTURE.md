# Developer architecture

The initial package keeps one Cargo library and one thin CLI binary. This avoids splitting tightly coupled numerical code into multiple crates before the integration boundaries have settled. It has no HTTP transport, web assets, UI framework or daemon.

```text
FITS / PNG / JPEG / TIFF              StarField from your detector
          |                                       |
      image_io                                    |
          |                                       |
 extract / highpass / skymask                      |
          +-------------------+-------------------+
                              |
                    Engine / pipeline
                              |
             select / fingerprint / learned
                              |
                index -> voting -> verification
                              |
                 wcsfit / geometry / types
                              |
             JSON / WCS / coordinate transforms
```

`Engine` owns the main memory-mapped index and compatible `index-deep` and `index-phone` companions when present. The caller owns its lifetime and request options. Start with one engine reused sequentially; concurrency and cancellation contracts need explicit validation before promising them as a stable integration API.

`pipeline` handles decoded images, multiple extraction attempts and the available solve budget. `solve` operates on star lists and retains the existing verification and refinement algorithms. A field can be unsolved or ambiguous; callers must check the status before using its WCS. The JSON envelope is shared by image and source-list input.

`builder` and `d80` form an offline data preparation path. Real catalogues are external inputs. `learned` implements LGQ-GD inference and probing; it does not download a model. A learned index contains its coder and manifest identity. The default CLI builder uses the handcrafted v1 key unless `--coder` is provided.

`Index::lock_in_memory` is an optional Unix `mlock` operation and needs sufficient
OS memlock allowance. On other platforms it returns `ErrorKind::Unsupported`;
ordinary memory mapping and solving remain available without page locking.

`measure` contains image measurements shared with the extractor. It is included because detector paths use it. `coordinates` contains only the deterministic WCS mathematics formerly shared with the annotation layer; catalogue annotations, atlas packs and web presentation are excluded.

## Changes made during extraction

- Selected the solver modules from a fixed source revision; no source history or private working data was copied.
- Replaced the original multi-product Cargo manifest and CLI with this standalone package.
- Added `Engine` and a uniform `SolveOutput` envelope, keeping the search algorithm in the existing modules.
- Removed HTTP client modules from `api`; its remaining content is serializable types and mathematical output contracts.
- Extracted deterministic coordinate functions into `coordinates` and retained their numerical parity tests. No web app or atlas reader was imported.
- Restricted optional local-index test discovery to this checkout or an explicit environment variable. Mandatory synthetic integration tests do not require external indexes.
- Kept the independently generated panorama WCS fixture and its regeneration script.

## Compatibility boundary

Version 0.1 is an integration preview, not a stable Rust ABI. Applications can depend on a pinned source revision or version. The package is currently `publish = false`; a registry release and package-name availability need their own release step.

The first stable boundary should be a small session API: open index, solve pixels, solve star rows, return a typed result, convert coordinates, close session. A C ABI can then expose opaque handles and explicit ownership. It should not expose internal Rust structures, allocator-dependent buffers or the phone navigation engine as its contract.

FITS output may represent the base projection/SIP only when an optional local residual grid is enabled by a low-level caller. The public CLI does not enable that grid. Projection support must be negotiated by integrations; ARC and panorama models are explicit CLI options.

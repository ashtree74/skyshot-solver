# CLI contract

Run `skyshot --help` and `skyshot <command> --help` for available options. There are five commands in this snapshot:

| Command | Input | Output |
| --- | --- | --- |
| `solve` | Image or `StarField` JSON, required `--index` | One JSON `SolveOutput`; optional new FITS WCS header |
| `batch` | Multiple image/JSON paths, required `--index` | JSONL with `input` and either `result` or `error`; index opened once |
| `extract` | Image | `StarField` JSON, no index needed |
| `inspect` | Index directory | Index manifest JSON after the reader's structural checks; not a complete checksum audit |
| `build` | External D80 directory and new `--out` | Index files plus manifest JSON |

`solve` and `batch` accept `--mode auto|standard|deep`, `--deadline-ms`, `--min-fov`, `--max-fov`, `--matches`, `--arc` and `--panorama`. The deadline is a solve budget, not a bound on process startup, decoding or index mapping. There is no remote fallback.

`auto` here selects between ordinary and DEEP catalogue search; it never means a network backend. DEEP requires a compatible companion or a DEEP main index. Companions are mapped once by `Engine`; corrupt companions are errors, not silently ignored data.

Source lists use the `StarField` schema: `width`, `height`, `stars`, optional `epoch`. Positions are zero-based native pixels, sky angles are degrees and scale is arcseconds/pixel. Each star includes `x`, `y`, `flux` and positive `sigma`; additional detector fields have their Rust serde defaults. `extract` and the synthetic demo produce valid examples. Image files can supply FITS position/scale hints; a failed constrained search still requires normal verification.

The `solve` envelope contains `schema_version: 1`, `solution`, `timings` and `n_stars`. Always inspect `solution.status`; absence of a process error does not itself establish a valid WCS. Detailed diagnostic fields are experimental. Non-finite floating-point diagnostics serialize as JSON null.

Exit codes are 0 for success, 1 for unsolved/ambiguous and 2 for invalid arguments, configuration or I/O failure. A batch returns 2 if any input errors, otherwise 1 if any input is unsolved, otherwise 0. Failure to open the shared index aborts the batch before results. JSON goes to stdout, diagnostics to stderr. A failed solve never creates a WCS file and an existing WCS path is refused.

There is no full ASTAP command-line emulation or `solve-field` executable adapter in this snapshot. The low-level `compat` writers are present; adapters must have dedicated conformance tests before advertising compatibility with capture applications such as N.I.N.A.

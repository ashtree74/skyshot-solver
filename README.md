# Skyshot Solver

**Turn a view of the stars into a live map of the sky.**

## [Try Skyshot live → skyshot.app](https://skyshot.app)

**Experience Skyshot in your browser with the live web demo.**

[How it works](#how-skyshot-solves-in-milliseconds) · [What you can build](#what-you-can-build) · [Build the solver](#build-and-try-without-downloading-a-catalogue) · [Watch the video](https://youtu.be/lBq3xXwquJM) · [Research](#research)

Skyshot is an open-source **blind plate-solving engine built for millisecond-scale sky recognition**. Give it an image or a list of detected stars, and it works out where the camera is pointing, its image scale and orientation, and a World Coordinate System (WCS) that maps pixels to the sky. It can solve without an initial position, scale, rotation or parity hint, using a local catalogue index and a CPU.

Created by **Adam Jesionkiewicz**. Available here as a **Rust library and CLI**, with source code, research papers and examples you can build on.

The live web demo and video demonstrate the broader Skyshot project. This repository contains the engine and developer tools; the web application and visual overlays are separate.

[![Watch the Skyshot demo on YouTube](https://img.youtube.com/vi/lBq3xXwquJM/hqdefault.jpg)](https://youtu.be/lBq3xXwquJM)

## What changes when the sky can be solved in milliseconds

A camera that knows which stars it is seeing becomes a new kind of instrument. Pan across the Milky Way and imagine a coordinate grid staying attached to the sky. Point a telescope and recover its orientation from the next exposure. Feed a sequence of frames into an application and give each one its own celestial coordinates.

**Millisecond solving makes sky recognition a building block for live interaction.** It opens a path to video annotation, camera-based AR, smart optics and autonomous instruments that continuously recover where they are looking. The same foundation can calibrate an astrophotography session or help a researcher prototype a star tracker.

Skyshot brings that foundation into a local, embeddable engine. You can inspect the method, run it on your own images, connect your own detector and build the application around it.

## How Skyshot solves in milliseconds

The speed comes from making the search selective at every stage: reproducible star selections, compact lookup keys, agreement between independent patterns and geometric verification.

1. **Find the same stellar neighbourhoods at different scales.** Each star gets a *dominance distance*: the distance to its nearest brighter neighbour. Nested thresholds on that distance select locally prominent stars at several densities. Removing fainter stars leaves a retained star's dominance distance unchanged, helping a shallow exposure and a deep catalogue reproduce the same neighbourhoods. The image searches these levels in pixels, without needing to know its angular scale in advance.

2. **Compress a neighbourhood into a 32-bit retrieval key.** Local geometry is normalized for position, scale and rotation, with explicit handling of mirrored images. Skyshot supports a handcrafted key and **LGQ-GD**, a small learned encoder that makes the keys more selective. Its job is to retrieve promising catalogue matches efficiently; the final answer is established by geometric verification.

3. **Make each lookup carry a pointing hypothesis.** Index records retain geometry that lets a match propose sky position, scale, rotation and parity. Patterns from distinct anchor stars vote for a common solution, concentrating verification on candidates with independent support.

4. **Verify the stars, then fit the camera.** Candidate solutions are checked against catalogue stars with statistical acceptance tests. WCS refinement models the projection and lens distortion, producing coordinates an application can use across the image.

5. **Keep the working path small and local.** Compact, memory-mapped indexes support selective reads; learned-key inference runs in Rust on the CPU. An application can keep one `Engine` open and reuse its index mappings across frames. Runtime solving needs no GPU, network service or account.

In the author-run experiments reported in the [LGQ-GD paper, Section 7.4 and Table 5](research/skyshot-learned-key.pdf), the learned-key configuration achieved a **4.2 ms median solver time** and **14.4 ms median decode–detect–solve time** on the 93 correctly solved images from a 96-image corpus (89 real, seven synthetic), on an Apple M5 Max. Each image's timing was the minimum over repeated runs on a shared machine; these are historical research measurements, not cold-start timings or a benchmark of this extracted package. Hardware, image content, index choice and initialization affect latency; difficult fields can take substantially longer.

Read the [method paper](research/skyshot-method.pdf) for the full algorithm and evaluation, and the [developer architecture](docs/ARCHITECTURE.md) for the code layout. The initial CLI builder uses the handcrafted key by default; learned-key operation requires a compatible index containing its coder. See [index setup](docs/DATA.md).

## What you can build

These application directions come from the [Skyshot project](https://skyshot.app). The library and CLI provide image/star-list solving and coordinate transforms today; the examples below describe systems developers can build around those primitives.

| Application | What Skyshot can contribute |
| --- | --- |
| **Astrophotography and astronomy** | Solve frames across a session and export WCS for stacking, mosaics, photometry and astrometry. Use the coordinates to add catalogue annotations or align survey imagery such as DSS2, Pan-STARRS and 2MASS in your application. |
| **Image verification and OSINT** | Recover the sky coordinates and orientation of a photograph or video frame. Combine them with horizon geometry and an independently known or claimed time/location to check consistency; a plate solution alone does not establish where on Earth an image was taken. |
| **Video, timelapse and live streams** | Feed decoded frames into a persistent engine, then draw coordinate grids, constellation figures and object labels in your renderer. Possible integrations include meteor cameras, all-sky cameras and telescope streams. |
| **Outreach, planetaria and sky shows** | Identify and explain the actual sky seen by a camera, with labels and educational overlays aligned to the captured image. |
| **Mobile AR** | Anchor an overlay to observed stars and use it to correct an application's inertial orientation estimate. Camera capture, rendering and sensor fusion belong to the integration. |
| **Celestial navigation without GNSS** | Explore star-derived attitude as an input to navigation, combined with accurate time, a local vertical or horizon reference, and sensor calibration. Geographic positioning is a research direction requiring additional sensors, algorithms and field validation. |
| **Defence and security** | Explore passive optical orientation references, sensor pointing verification and imagery consistency checks where radio navigation is unavailable. Operational validation belongs to the resulting system. |
| **Star trackers: satellites, balloons and UAVs** | Prototype recovery of celestial attitude from a single frame without an initial pointing estimate, then add tracking and sensor fusion. This is research and prototyping software, not flight-qualified hardware or a complete navigation system. |
| **Telescopes, binoculars and AR optics** | Add stellar pointing determination to finders, smart telescopes and camera-equipped optics, with the solver embedded in the device or a companion computer. |

Video decoding, tracking, overlays, mobile sensor fusion and navigation are not included in this developer snapshot. Their latency and accuracy must be evaluated as complete systems.

## What is in this repository

The initial source package includes star extraction, dominance fingerprints, handcrafted and learned-key inference, index building and reading, blind and hinted solving, verification, WCS fitting and JSON/FITS WCS output. It contains no web application, desktop UI, hosted service, agent or activation system.

**Development snapshot:** the Rust API and CLI are being prepared for a first public implementation release. This checkout is not a published binary release. Platform results and their limits are recorded in the [current verification report](docs/verification-2026-10-17.md); the CI matrix alone does not establish tested support. See [installation](docs/INSTALL.md) and the [release checklist](docs/RELEASE.md).

## Build and try without downloading a catalogue

Install a current stable Rust toolchain and your platform's native build tools. Then:

```sh
cargo build --release --locked
cargo test --locked

# Creates a small artificial catalogue and a matching star list in a NEW directory.
cargo run --release --locked --example synthetic_demo -- demo

# Runs the actual index search and verification; the expected centre is RA 83.8°, Dec −5.4°.
cargo run --release --locked -- solve demo/stars.json --index demo/index --wcs demo/solution.wcs
```

The synthetic example checks the full development path. Its small catalogue cannot solve real sky images and does not measure real-world accuracy or speed.

## Solve your images

Use a compatible, separately obtained index:

```sh
skyshot solve frame.fits --index /path/to/index --wcs solution.wcs
skyshot extract frame.fits > stars.json
skyshot solve stars.json --index /path/to/index
skyshot batch frame1.fits frame2.fits --index /path/to/index > results.jsonl
skyshot inspect /path/to/index
```

The executable is built at `target/release/skyshot` (`skyshot.exe` on Windows). Use that path directly, or install it locally with `cargo install --path . --locked`. The examples using `skyshot` assume it is on `PATH`.

Image inputs: supported FITS images, PNG, JPEG and TIFF. `solve` prints JSON; diagnostics go to stderr. Exit codes: **0** solved/success, **1** unsolved or ambiguous, **2** input/configuration/I/O failure. `batch` emits one JSON line per input, continues after individual failures and returns the most severe exit status. WCS files are written only on success and existing files are never replaced.

See [the CLI contract](docs/CLI.md) and [index setup](docs/DATA.md). Prebuilt catalogue indexes and model weights are not bundled in this initial extraction. The engine can read compatible learned-key indexes containing their coder; building a new learned-key index requires a separately supplied coder pair. Dataset provenance and distribution are part of the release plan.

The independent build route uses an externally obtained **ASTAP D80, Gaia DR3,
epoch 2025** catalogue and the handcrafted key. [Data setup](docs/DATA.md) gives
the required layout and commands. [Versioned bundle tooling](docs/DATA-RELEASE.md)
is available for preparing and checking separate downloads. Data preparation
targets non-commercial use with openly recorded sources, attribution and terms;
external data conditions remain separate from Apache 2.0 on the code. Download
hosting and publication are still pending. Local versioned candidates for full
learned, DEEP, wide, full handcrafted and the standalone LGQ-GD coder can now be
prepared and installed with the [data installation guide](docs/DATA-INSTALL.md).
Each learned index already includes its matching coder; the standalone model is
for building new indexes. Metadata identifies Adam Jesionkiewicz as the
solver/model author and bundle maintainer, with catalogue attribution separate.

## Embed the engine

The Cargo package is `skyshot-solver`; its Rust library is named `skyshot`:

```rust,no_run
use skyshot::{Engine, solve::SolveConfig};

fn main() -> anyhow::Result<()> {
    let engine = Engine::open("/path/to/index")?;
    let options = SolveConfig::default();
    for image in ["frame1.fits", "frame2.fits"] {
        let result = engine.solve_path(image, &options)?;
        println!("{}", result.solution.status);
    }
    Ok(())
}
```

Reuse the same engine to keep indexes mapped across frames. `Engine::solve_sources` accepts a `StarField` from your own detector. `Wcs` exposes pixel-to-sky and sky-to-pixel transforms. Algorithms remain accessible in lower-level modules for research and custom integrations.

Runnable examples: [repeated image solving](examples/repeated_solve.rs), [synthetic catalogue](examples/synthetic_demo.rs), [Python subprocess integration](examples/python_cli.py). A stable C ABI, .NET/C++ examples and persistent JSONL request mode are planned; they are not part of this snapshot. The developer scope and sequence are in [PUBLIC-REPO-PLAN.md](docs/PUBLIC-REPO-PLAN.md).

## Architecture and development

Read [ARCHITECTURE.md](docs/ARCHITECTURE.md), [CONTRIBUTING.md](CONTRIBUTING.md) and the [data policy](docs/DATA.md). Generate Rust documentation with `cargo doc --no-deps`. The initial extraction records its source hashes in [provenance/import.json](provenance/import.json).

No account, API key, telemetry or network request is required at runtime. Dependencies are fetched by Cargo during the first build; real sky solving requires compatible catalogue data obtained separately.

## Research

The two English manuscripts were first published on 11 October 2026:

| Manuscript | PDF |
| --- | --- |
| Skyshot Solver: Blind astrometry with nested stellar dominance levels and compact local descriptors | [Read the paper](research/skyshot-method.pdf) |
| LGQ-GD: A Learned 32-bit Retrieval Key for Stellar Neighbourhoods | [Read the paper](research/skyshot-learned-key.pdf) |

They describe the revisions and experiments evaluated at the time. They are not a benchmark report for this extracted package. Independent evaluation is welcome.

For software attribution, use [CITATION.cff](CITATION.cff) and record the source
revision plus the data/model bundle used in your work.

## License

The new implementation and development documentation are under [Apache License 2.0](LICENSE). The previously published manuscripts retain their [MIT license](research/LICENSE). Copyright and attribution are in [NOTICE](NOTICE). External catalogues, model training data and dependencies retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

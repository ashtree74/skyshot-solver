# Local real-image comparison

`real_images_ab.py` and `examples/benchmark_images.rs` compare the existing full
LGQ-GD and handcrafted indexes against a caller-supplied local reference corpus.
They do not download, modify or publish images, indexes or reference files.
Results must go into a new directory under this checkout's ignored `outputs/`.

Build and run using Python 3.11+ with NumPy, Astropy and psutil installed:

```sh
cargo build --release --locked --example benchmark_images
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s bench -p 'test_real_images_ab.py'
PYTHONDONTWRITEBYTECODE=1 python3 bench/real_images_ab.py \
  --source /path/to/read-only-source-checkout \
  --out outputs/real-sky-ab-new --threads 4 --repeats 5
```

The source layout is `work/truth/manifest.json`, existing WCS truth records and
referenced images, plus `work/index` and `work/index-v1`. Synthetic entries and
exact duplicate image hashes are excluded. Referenced images are hashed against
their recorded identities before the run. Catalogue geometry and levels must
match between the indexes. The runner uses those main indexes explicitly,
without auto-discovering a shared learned DEEP or wide companion.

All solves run serially, with position/scale hints removed in memory. A full
untimed warmup precedes measured paired repeats. Input order is shuffled with a
recorded seed and arm order alternates. Timing aggregates use each image's median
across repeats, then p50/p95 across images. Both all-input and common-correct
subsets are reported, with pipeline and solver-only times kept separate.
Fresh-process measurements include startup/mapping and have a warm OS cache;
they are not a disk-cold experiment. Background-load samples are approximate and
must accompany interpretation on a shared machine.

Astropy/WCSLIB scores the emitted WCS against stored references. The report
separates wrong fields, right fields with inaccurate WCS, unverified results and
unsolved inputs. The inner 10/50/90% grid has a 3 px accuracy gate; a separate
0/50/100% grid checks the edges. Input/reference/index hashes are checked again
after measurement. Keep group composition, reference ancestry and any weak or
missing references visible; do not infer a global false-positive rate from a
positive-image corpus or count repeated exposures as independent sky fields.

`qualify_data_release.py` rechecks a new sanitized data copy or fresh installed
archives against that existing private plan and paired baseline. It also runs
ordinary CLI companion discovery, explicit DEEP and three standalone wide
checks. Its sanitized aggregate receipt covers exact runtime hashes; raw private
paths and per-image solutions remain in the ignored output directory. See
[the bundle workflow](../docs/DATA-RELEASE.md) for preparation and installation.
This is data qualification, not a replacement for the timing experiment above.

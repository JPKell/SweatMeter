# Phase 4 Performance Validation

Measured 2026-08-22 with Python 3.13.15 on the reference machine described in
[performance targets §1](architecture/performance-targets.md) (NVIDIA GeForce RTX 5060 Ti, driver
580.173.02). Each snapshot value is the median of warmed iterations. Sampler CPU is process CPU
divided by 2.2 seconds of elapsed time at a 1-second interval. Workload distortion is the median of
five sampled runs, each bracketed by two baseline runs of the same fixed 20-million-iteration
integer workload. The test pins the process to one available CPU, warms it three times, and
compares the sampled throughput with the geometric mean of its flanking baselines to control
gradual frequency drift. Each sampled run lasts long enough to include the sampler's immediate
reading and at least one 1-second periodic reading.

| Measure | Result | Target | Ceiling | Outcome |
|---|---:|---:|---:|---|
| Snapshot, Linux host + null GPU | 0.241 ms median | 3 ms | 10 ms | Pass |
| Snapshot, real single GPU via **NVML** (`pynvml`) | **0.749 ms median** | 3 ms | 10 ms | Pass |
| Snapshot, real single GPU via `nvidia-smi` | 57.6 ms median | 40 ms | 120 ms | Ceiling met, target missed |
| Snapshot, deterministic single-GPU CSV parsing | 0.277 ms median | — | — | Package/parser overhead only |
| Sampler CPU, 1-second interval | 0.242% of one core | 0.5% | 1.5% | Pass |
| Sampler peak traced allocation | 22,551 bytes | 5 MiB steady memory budget | — | Pass |
| Synthetic workload throughput distortion | 0.000% median | 1% | 2% | Pass |

The degradation samples were 0%, 0%, 0%, 0%, and 0.098%; sampled runs faster than their flanking
baselines count as zero degradation. The median is used as required by the suite performance
standard rather than selecting one noisy run. That measurement is sensitive to other load on the
machine: on a busy host it drifts to 2–3% and the assertion fails, so run the performance job on an
otherwise idle runner.

## Why there are two GPU numbers

The `nvidia-smi` backend spends two processes per snapshot — one for core metrics, one for throttle
reasons — and on this machine the processes, not the queries, are the cost:

| Invocation | Median |
|---|---:|
| Core metric query (13 fields) | 29.8 ms |
| Throttle-reason query (9 fields) | 23.0 ms |
| Both sets of fields in **one** query (21 fields) | 28.6 ms |

The extra columns are free; the second process is not. The two queries are nonetheless deliberate,
and the reason is recorded in [ADR-0021](adr/0021-telemetry-collection-strategy.md): one unknown
field fails the whole command, which was confirmed against this driver —

```console
$ nvidia-smi --query-gpu=index,memory.used,clocks_throttle_reasons.bogus --format=csv,noheader
Field "clocks_throttle_reasons.bogus" is not a valid field to query.    (exit 2)
```

`memory.used` returns nothing either. Folding the throttle columns into the core query would let one
unsupported sensor erase every valid metric beside it.

The NVML backend removes the question rather than trading against it. Each metric is an independent
in-process call, so an unsupported sensor degrades only itself, and a snapshot costs **0.75 ms
instead of 57.6 ms** — a 77× reduction that brings the single-GPU path inside the *no-GPU* budget.
That is what makes sub-second sampling viable: at a 1-second interval the command backend spends
roughly 5% of a core on process startup alone.

`nvidia-smi` remains the default wherever the optional extra is absent, and its 120 ms ceiling is
asserted by a performance test that skips when no GPU is present. Its 40 ms target stays recorded as
missed rather than quietly rewritten: installing `sweatmeter[pynvml]` is the supported way to meet
it, and a combined-query-with-fallback would still be available if the command backend ever has to
carry that budget on its own.

Run the checks with:

```bash
pytest -m performance
pytest -m live
```

The default unit suite excludes both markers and requires no GPU, provider, or network. The GPU rows
above skip automatically when no device is present.

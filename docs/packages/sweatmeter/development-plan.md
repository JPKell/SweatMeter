# SweatMeter — Development Plan

**Sequence position:** fourth component. Depends on BaseAiCore Phase 4 (machine types) — can be built
in parallel with ModelRack.
**Target:** `sweatmeter 0.3.0` by the end of Phase 4. Met as code; `0.3.0` was never published,
because the NVML backend ([ADR-0021](../../adr/0021-telemetry-collection-strategy.md) §7) landed
before it reached the index, so the first published release is **`0.4.0`**.

---

## Phase 1 — Linux host readers

**Goal:** CPU, RAM, load, disk throughput and CPU temperature are read from `/proc` and `/sys`, with
every parser tested from fixture text.

**Prerequisites:** `baseaicore>=0.4,<0.5`.

**Work**
* Repository skeleton.
* `readers/linux.py`: `LinuxHostReader` composed of small pure parsers —
  `parse_proc_stat` (with delta state), `parse_meminfo`, `parse_loadavg`, `parse_diskstats`,
  `parse_cpuinfo`, `read_thermal_zones`, `read_hwmon`, `read_block_devices`.
* Every reader takes an injected text source or root path plus a clock.
* `_safe(fn, default=UNSUPPORTED)`: the single sanctioned narrow-catch helper, documented and tested.
* `types.py`: `MemoryReading`, `DiskThroughput`, `HostFacts`.

**Files/subsystems**
```text
src/sweatmeter/{__init__,__about__,types,safe}.py
src/sweatmeter/readers/{__init__,protocols,linux}.py
tests/unit/test_linux_readers.py
tests/fixtures/telemetry/proc/*  tests/fixtures/telemetry/sys/**
```

**Tests**
* `proc/stat`: first call has no delta (returns `UNSUPPORTED`), second computes a percentage;
  counter wrap handled; multi-core aggregation.
* `meminfo`: normal, minimal, missing `MemAvailable` (fallback documented), malformed lines.
* `diskstats`: throughput between two readings, device filtering, first-call behaviour.
* `cpuinfo`: x86 multi-socket, ARM (no `model name`), physical vs logical core derivation.
* Thermal: `thermal_zone*` present, hwmon fallback, neither present, permission error.
* `_safe`: converts a raising callable to the default and logs at DEBUG; does not catch
  `KeyboardInterrupt`/`SystemExit`.

**Acceptance criteria**
1. Every parser is covered by fixture-based tests that run on any machine.
2. No parser raises on malformed input; each returns `UNSUPPORTED` for the affected field.
3. Coverage ≥ 95 %; strict typing clean.

**Known risks:** distribution differences in `/proc` and `/sys` layouts. Mitigated by fixtures from
more than one layout and by field-level degradation.
**Likely failure modes:** first-call CPU percentage reported as 0 instead of `UNSUPPORTED`; counter
wrap producing negative throughput.
**Gold standards:** unsupported-safe; fixture-tested; zero dependencies.
**Deferred:** GPU, sampler, machine profile.

---

## Phase 2 — NVIDIA GPU reader

**Goal:** GPU utilization, VRAM, temperature, power, fan, clocks and throttle reasons are available
for one or more NVIDIA GPUs, and their absence is handled honestly.

**Prerequisites:** Phase 1.

**Work**
* `readers/nvidia.py`: `NvidiaSmiReader` with an injectable subprocess runner; CSV query by column
  name; `available()`; `static_info()` (name, UUID, VRAM total, driver, CUDA, compute capability);
  `sample()`.
* Defensive parsing: `[N/A]`, `[Not Supported]`, blank fields, unexpected column order, extra columns.
* Throttle reasons via `clocks_throttle_reasons.*` where available, with
  `throttle_reasons_available` set truthfully.
* Fixtures captured from driver 580.173.02, annotated with the driver version.

**Files/subsystems**
```text
src/sweatmeter/readers/nvidia.py
tests/unit/test_nvidia_reader.py
tests/fixtures/telemetry/nvidia/*.csv
```

**Tests**
* Single GPU, two GPUs, zero GPUs; `nvidia-smi` absent, non-zero exit, timeout, empty output,
  malformed CSV, missing columns, `[N/A]` values.
* Unit conversion: MiB → bytes; percentages bounded 0–100; watts as float.
* Reordered/extra columns parsed correctly (name-based, not positional).
* No shell invocation (asserted on the runner's call arguments); timeout always supplied.

**Acceptance criteria**
1. Tests pass on a machine with no GPU.
2. A missing power sensor yields `UNSUPPORTED` with reason, never `0`.
3. `throttle_reasons_available` distinguishes "none" from "cannot tell".

**Known risks:** driver-version output changes. Mitigated by name-based parsing and versioned
fixtures.
**Likely failure modes:** positional parsing; treating `[N/A]` as 0; a hung subprocess without a
timeout.
**Gold standards:** honest availability reporting; no shell; bounded runtime.
**Deferred:** AMD, Intel, Apple, `pynvml`.

---

## Phase 3 — Collector, machine profile and sampler

**Goal:** a consumer gets a complete snapshot and a stable machine profile, and can sample
continuously in the background without affecting measurements.

**Prerequisites:** Phases 1–2.

**Work**
* `collector.py`: `TelemetryCollector.snapshot()` and `.machine_profile()`, both non-raising, both
  composing readers through `_safe`, plus `unavailable_reasons()`.
* Factories: `create_host_reader`, `create_gpu_reader`, plus **public** `NullHostReader` and
  `NullGpuReader` (§13's degradation path tells consumers to construct one, so it cannot be private),
  and the documented tier-3 stubs raising `UnsupportedPlatformError`.
* `sweatmeter.testing`: `ScriptedHostReader`, `ScriptedGpuReader`, `FaultInjectingReader` — shipped as
  supported API, because LoadCoach's admission control and FreeWeight's energy integration both need a
  deterministic telemetry series and would otherwise monkeypatch the readers.
* `sampler.py`: `TelemetrySampler` — thread, configurable interval, callback and iterator interfaces,
  `latest()`, optional bounded ring buffer, context manager, clean shutdown, callback exception
  isolation.

**Files/subsystems**
```text
src/sweatmeter/{collector,sampler,platform}.py
src/sweatmeter/readers/{windows,darwin}.py     # documented stubs
tests/unit/{test_collector,test_sampler,test_platform_factory}.py
```

**Tests**
* Fault injection: each reader method raises in turn; `snapshot()` still returns, only that field
  degrades, and the reason is recorded.
* Machine profile assembles correctly and produces a stable fingerprint; GPU ordering normalized.
* Sampler: interval accuracy within tolerance using a fake clock; `stop()` joins within the timeout;
  a raising callback does not kill the thread; no thread leak across 100 start/stop cycles;
  `latest()` returns the newest snapshot and exposes its age.
* Platform factory: each branch, including `UnsupportedPlatformError` for an unknown platform; stubs
  raise rather than returning zeros; `NullHostReader` degrades every field to `UNSUPPORTED` **with a
  reason** and never to `0`.
* Test doubles replay a fixed series deterministically, and `FaultInjectingReader` degrades exactly
  the named field.

**Acceptance criteria**
1. `snapshot()` cannot raise — proven by exhaustive fault injection.
2. Machine profile matches the exclusion policy (no driver version in the fingerprint).
3. Sampler starts and stops cleanly under a stress test.
4. Coverage ≥ 95 %.

**Known risks:** thread lifecycle bugs surfacing only under load. Mitigated by the stress test and by
a context manager that always stops.
**Likely failure modes:** a sampler thread outliving the process; a stale `latest()` presented as
current (mitigated by exposing sample age).
**Gold standards:** never raises; never fabricates; clean shutdown.
**Deferred:** derived metrics; performance validation.

---

## Phase 4 — Derived metrics, performance validation, publication

**Goal:** consumers can compute peak VRAM, mean power, energy and throttling from a sample window,
with measured overhead, and the package ships.

**Prerequisites:** Phase 3.

**Work**
* `window.py`: `TelemetryWindow` — `peak_vram_bytes`, `mean_power_watts`, `energy_joules`
  (Σ power × dt, with dt from actual sample timestamps, not the nominal interval),
  `max_temperature_c`, `suspected_throttling`, `sample_count`. Every method returns `UNSUPPORTED`
  when its inputs are unavailable, and reports the number of supported samples it used.
* Performance tests for every budget, including the throughput-distortion measurement.
* README, quickstart, `docs/platform-support.md`; publish the package (shipped as `0.4.0`).

**Files/subsystems**
```text
src/sweatmeter/window.py
tests/unit/test_window.py
tests/performance/test_overhead.py
tests/live/test_real_machine.py         # marked
docs/{quickstart.md,platform-support.md}
```

**Tests**
* Energy integration against a known power/time series with a hand-computed expected value, **per
  device**; a two-GPU window yields two figures and no total, because a summed watt-hour describes no
  hardware that exists ([ADR-0027](../../adr/0027-multi-gpu-semantics.md)).
* Irregular sample intervals integrated correctly (dt from timestamps).
* Windows with some `UNSUPPORTED` samples: statistics computed from supported samples only, with the
  count reported; all-unsupported ⇒ `UNSUPPORTED`, never 0.
* Throttle heuristic: clock drop with high temperature ⇒ suspected; clock drop with low utilization
  ⇒ not suspected; no clock data ⇒ verdict `unknown`.
* Performance: snapshot latency with and without GPU; sampler CPU at 1 s; a synthetic workload run
  with sampling on and off, asserting ≤ 1 % throughput difference.
* Live (marked): plausible values on the reference machine.

**Acceptance criteria**
1. Every §20 criterion in the [spec](spec.md) is met.
2. Energy is labelled an estimate everywhere it appears, including in docstrings.
3. Performance budgets met and recorded.
4. The package is published and usable standalone. Shipped as `0.4.0` rather than `0.3.0`: the
   criterion is that a published version installs and works on its own, and publishing a
   superseded `0.3.0` first would have put a version with a known budget miss and an
   unreachable diagnostic into the compatibility matrix permanently.

**Known risks:** the throttle heuristic producing false positives. Mitigated by returning a verdict
with a reason and a confidence, never a bare boolean, and by documenting it as a heuristic.
**Likely failure modes:** averaging over unsupported samples as if they were zero; using the nominal
interval instead of real timestamps for energy.
**Gold standards:** graceful degradation proven by fault injection; measured, bounded overhead;
honest estimates; zero dependencies; ≥ 95 % coverage.
**Deferred:** other GPU vendors, Windows/macOS host readers, per-process GPU attribution, RAPL.
The `pynvml` backend was *not* deferred in the end: it shipped in `0.4.0` as the automatically
selected reader wherever the optional extra is installed.

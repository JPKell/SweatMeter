# Changelog

All notable changes to `sweatmeter` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/), pre-1.0 per
`docs/standards/packaging-and-release-standards.md` §3.

## [Unreleased]

## [0.4.0] — 2026-08-22

The in-process NVML GPU backend, plus a documentation-conformance review of the completed package
against the suite's master documents.

**First published release.** `0.3.0` completed the development plan but was never published; the
NVML backend landed before it reached the index, so `0.4.0` is the first version on PyPI. Consumers
pin `sweatmeter>=0.4,<0.5`.

### Added
- **NVML GPU backend** (`NvmlGpuReader`), delivering ADR-0021 §7 and the spec's remaining §21
  extension. It reads NVIDIA devices in-process through the optional `pynvml` extra instead of
  running `nvidia-smi` twice per snapshot: a real single-GPU snapshot drops from 57.6 ms to
  0.75 ms, meeting the 40 ms target with room to spare and making sub-second sampling viable.
  Every metric is an independent NVML call, so an unsupported sensor degrades only itself, and
  `throttle_reasons_available` is answered from the driver's supported-reason mask rather than
  inferred from parseable text.
- `GpuBackend.PYNVML`, and `create_gpu_reader()` now selects NVML when the extra is importable and
  `nvidia-smi` otherwise. The probe inspects the import system only — it neither loads NVML nor
  touches a device. An explicit `PYNVML` request without the extra raises
  `DependencyUnavailableError` naming it, rather than silently downgrading.
- A `GpuReader` **conformance suite** (`tests/contract/`) running one set of protocol invariants
  against every implementation — recorded `nvidia-smi` output, NVML over a fake binding,
  `NullGpuReader` and `ScriptedGpuReader` — plus an equivalence test pinning both NVIDIA backends
  to identical output for the same device, and a live test asserting the same against real
  hardware. The two backends are interchangeable, not merely similar.
- `NvmlBinding`, the published protocol for the NVML surface SweatMeter calls, so the whole backend
  is testable without the library or a GPU; `load_nvml_binding()` and `nvml_binding_available()`.
- `requirements/ci.lock` and `requirements/release.lock`: exact, hash-verified pins for this
  repository's own CI and release pipeline, required by Packaging and Release Standards §4 and
  Security Standards §11, with `requirements/README.md` covering their purpose and regeneration.
- `sweatmeter.testing` re-exports `NullHostReader` and `NullGpuReader`, so all four doubles named
  by the suite testing standards are importable from one module. Their definitions stay in
  `sweatmeter.platform`, because the spec's degradation path constructs them in production code.
- Degradation tests for every remaining collector branch: readers returning the wrong type,
  out-of-range, non-finite and non-numeric sensor values, malformed and duplicate GPU rows and
  static profiles, and a reader whose diagnostic surface is not a mapping.
- Sampler tests for the two documented behaviours that had none: missed deadlines are skipped
  rather than replayed as a burst, and an invalid injected monotonic clock falls back safely.
- `FaultInjectingReader` tests for every documented `ValidationError` and for its delegation of
  availability and wrapped-reader diagnostics.
- An autouse test guard that fails any non-live test opening a network connection, proving the
  spec's no-network claim rather than asserting it in prose.
- CI: tier-3 Windows and macOS early-warning jobs, a gate that keeps platform branching inside the
  factory and reader modules, a nightly performance and live hardware job, and an install-check
  step that runs the spec's five-line standalone script against the built wheel.
- A performance test that measures the real `nvidia-smi` snapshot path against its 120 ms ceiling
  and skips when no GPU is present.

### Fixed
- `machine_profile()` now records `machine_profile: collector_error` when internal assembly fails.
  The diagnostic existed but was unreachable, so a wholly failed profile was indistinguishable
  from a machine that simply reports little about itself.
- Removed an unreachable guard in the throttle heuristic: `_substantial_clock_drop` now skips
  unavailable clock readings itself instead of receiving a pre-filtered sequence.
- `test_snapshot_normalizes_gpu_order_and_rejects_duplicate_index` did not contain a duplicate
  index and never exercised the rejection it named.

### Changed
- `_safe_call` in `sweatmeter.safe` returns the failure alongside the degraded value, so the NVML
  backend can classify NVML status codes into reasons without opening a second broad `except`
  anywhere in the package. `_safe` now delegates to it, keeping exactly one such catch (Coding
  Standards §6).
- `pynvml` joins the `dev` extra: importing it needs no GPU, and CI must exercise both backends of
  the conformance suite.
- CI and the release workflow install the locked sets instead of re-resolving on every run, and
  build with `--no-isolation` so the build backend comes from `release.lock`. The 3.14 and tier-3
  early-warning jobs still resolve from ranges: the former because pinning versions without 3.14
  wheels defeats the warning, the latter because a Linux-resolved lock omits win32-only
  dependencies that a hash-checked install would then refuse.
- CI installs the built distribution rather than an editable checkout, per Packaging Standards §4.
  Coverage is configured by importable name with a `paths` mapping so it measures the package
  wherever it is installed; the previous source-path configuration reported 0% against a
  non-editable install.
- The `security` job runs `pip-audit` against both lockfiles. It previously ran bare, auditing an
  environment that contained only `pip-audit` itself.
- Log records use stable, low-cardinality event names with structured `extra` fields, per the
  coding and observability standards. DEBUG-only logging under `sweatmeter.*` is unchanged.
- Contract tests moved to `tests/contract/`, matching the testing-standards layout.
- Refreshed the curated `docs/` snapshot from the central documentation set (adds ADR-0030 and the
  amended BaseAiCore spec, traceability matrix and roadmap).
- `docs/performance-validation.md` reports both GPU backends measured on real hardware, and the
  evidence behind the split-query design. The previous text stated the GPU was unavailable to the
  validation host, which is no longer true. The `nvidia-smi` path's 40 ms target stays recorded as
  missed (57.6 ms, inside its 120 ms ceiling) rather than quietly rewritten; installing the extra
  is the supported way to meet it.

### Security
- `pytest` moved from `>=8,<9` to `>=9.0.3,<10`, excluding PYSEC-2026-1845 (vulnerable
  `/tmp/pytest-of-{user}` handling, local denial of service or privilege escalation, affecting
  pytest through 9.0.2). Found by auditing the new lockfile; the suite passes unchanged on
  pytest 9.

## [0.3.0] — 2026-08-22

Phases 3–4 of the [development plan](docs/packages/sweatmeter/development-plan.md): complete
collection, deterministic sampling, derived per-device metrics, and publication readiness.

### Added
- `TelemetryCollector` with non-raising live snapshots, field-level degradation reasons, static
  `MachineProfile` assembly, stable fingerprints, and normalized GPU ordering.
- Public host/GPU factories, `NullHostReader`/`NullGpuReader`, and explicit Windows/macOS tier-3
  stubs that raise `UnsupportedPlatformError` instead of returning zeros.
- Supported `sweatmeter.testing` scripted and fault-injecting readers for downstream deterministic
  tests without monkeypatching operating-system boundaries.
- Restartable `TelemetrySampler` with monotonic scheduling, callback isolation, callback and
  iterator consumption, staleness age, optional bounded history, and bounded clean shutdown.
- `TelemetryWindow` per-device peak VRAM, mean power, maximum temperature, actual-timestamp energy
  estimates, supported-sample counts, and three-state explained throttling verdicts.
- Linux current-process RSS readings and cached CPU-temperature sensor discovery.
- Quickstart, platform-support, performance-validation, live-smoke, lifecycle-stress, and complete
  performance-budget coverage.

### Changed
- Package version is `0.3.0`; README now documents the completed public API.

### Performance
- Recorded median no-GPU snapshot latency of 0.241 ms, sampler CPU of 0.242% of one core, 22,551
  bytes peak traced sampler allocation, and 0.000% median synthetic workload degradation on the
  validation host. See `docs/performance-validation.md` for scope and limitations.

## [0.2.0] — 2026-08-22

Phase 2 of the [development plan](docs/packages/sweatmeter/development-plan.md): NVIDIA GPU readers.

### Added
- `NvidiaSmiReader` with live availability probes, per-device samples, static GPU profiles, CUDA
  and compute-capability discovery, and diagnostic unavailability reasons.
- Immutable `GpuSample` values and the `GpuReader` protocol.
- Name-keyed defensive CSV parsing for reordered, missing, extra, malformed, and unsupported fields.
- NVIDIA driver 580.173.02 fixtures, including deterministic single- and two-GPU coverage.

### Security
- Resolve `nvidia-smi` explicitly, invoke it without a shell, and bound every call by both a timeout
  and a pre-parse output limit.

## [0.1.0] — 2026-08-22

Phase 1 of the [development plan](docs/packages/sweatmeter/development-plan.md): Linux host readers.

### Added
- Pure, fixture-testable parsers for `/proc/stat`, `/proc/meminfo`, `/proc/loadavg`,
  `/proc/diskstats`, and `/proc/cpuinfo`.
- Stateful CPU-utilization and disk-throughput deltas with honest first-sample and counter-wrap
  behavior.
- CPU temperature readers for thermal zones with an hwmon fallback, plus `/sys/block` static
  device discovery.
- Immutable `MemoryReading`, `DiskThroughput`, and `HostFacts` values and the `HostReader` protocol.
- The `_safe` operating-system boundary helper, which logs ordinary failures at DEBUG while
  preserving `KeyboardInterrupt` and `SystemExit`.

### Changed
- The package version is now `0.1.0`, matching the completed first phase.
- The package coverage gate is 95%, matching the shared-package testing standard.

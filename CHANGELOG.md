# Changelog

All notable changes to `sweatmeter` are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/), pre-1.0 per
`docs/standards/packaging-and-release-standards.md` §3.

## [Unreleased]

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

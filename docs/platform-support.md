# Platform support

What SweatMeter can read on each platform, and what it does where it cannot. The tier vocabulary
is the suite's (Cross-Platform Standards, in the suite documentation repository;
[spec §16](packages/sweatmeter/spec.md) is the contract this page restates for operators).

| Tier | Platform | What you get |
|---|---|---|
| **1 — Supported** | Linux x86-64 (Ubuntu 24.04+/Debian 13+/Fedora 41+ or equivalent) | Everything: `/proc` and `/sys` host telemetry, hwmon/thermal sensors, NVIDIA GPU telemetry via `nvidia-smi` or NVML, machine profiles. CI-tested, release-blocking. |
| **2 — Best effort** | Linux ARM64 | Should work; built and unit-tested in CI (including ARM `cpuinfo` parsing); hardware-specific telemetry is unverified on real boards. |
| **3 — Interface only** | Windows 11, macOS 14+ | Interfaces and stubs only. `create_host_reader()` raises `UnsupportedPlatformError` naming what is missing; a consumer constructs `NullHostReader` and degrades. **Not** advertised as supported. |

## What "degrades" means here

Nothing raises out of `TelemetryCollector.snapshot()` or `machine_profile()`, and nothing is ever
reported as `0` because it could not be read: an absent tool, an unreadable source, a
malformed value or an unsupported sensor becomes BaseAiCore's `UNSUPPORTED` (ADR-0016), and the
reason is retrievable per field from `snapshot.unavailable_reasons()` — "power sensor
unavailable" rather than a bare dash.

* `NullHostReader` answers every field `UNSUPPORTED` with reason `platform_unsupported` — never
  zeros.
* `NullGpuReader` reports no devices, which is different from a device with zeroed readings.

## GPU telemetry is platform-independent where the tool is

`GpuReader` works wherever its backend does, on any tier:

* **`NvidiaSmiReader`** — runs `nvidia-smi` per sample; works wherever the binary is on `PATH`,
  Windows included.
* **`NvmlGpuReader`** — in-process NVML via the optional `pynvml` extra
  (`pip install "sweatmeter[pynvml]"`); lower overhead, same readings. Both backends are pinned to
  identical output for the same device by the conformance suite (ADR-0021 §7).

`create_gpu_reader()` picks NVML when the binding is importable and healthy, else `nvidia-smi`,
else `NullGpuReader` — each step recorded, none of them a guess.

## Parsers are tested everywhere

Every `/proc`, `/sys` and `nvidia-smi` parser is tested from fixture text on every platform, so
Linux parsing stays fully covered in CI regardless of the runner's own hardware. The live suite
(`pytest -m live`) is the only part that reads the machine it runs on.

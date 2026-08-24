# SweatMeter

CPU/RAM/GPU/VRAM/thermal/power telemetry and static machine profiling; degrades honestly to Unsupported rather than fabricating a value.

**Status:** Phases 1–4 implemented (`sweatmeter 0.4.0`). Linux host readers, NVIDIA telemetry,
non-raising snapshots, stable machine profiles, bounded background sampling, per-device window
statistics, telemetry-derived energy estimates, and explained throttling verdicts are available.

Part of the **Local AI Suite**.

## Install

```bash
pip install sweatmeter

# Optional NVML backend: reads NVIDIA GPUs in-process instead of running `nvidia-smi` per sample.
pip install "sweatmeter[pynvml]"
```

## Quickstart

```python
from sweatmeter import TelemetryCollector

snapshot = TelemetryCollector().snapshot()
print("CPU:", snapshot.cpu_percent, "RAM:", snapshot.ram_used_bytes)
for gpu in snapshot.gpus:
    print(f"GPU {gpu.index}:", gpu.utilization_percent, "VRAM:", gpu.vram_used_bytes)
```

`snapshot()` and `machine_profile()` isolate ordinary sensor failures and do not raise. An absent
tool, unreadable source, malformed value, or unsupported sensor degrades honestly: collections
become empty and individual measurements become BaseAiCore's explicit `UNSUPPORTED` value, never
zero. Every degraded snapshot field has a reason in `snapshot.unavailable_reasons()`.

Background sampling and per-device derived metrics stay bounded and explicit:

```python
from sweatmeter import TelemetrySampler, TelemetryWindow

collector = TelemetryCollector()
with TelemetrySampler(collector, interval_seconds=1.0, buffer_size=60) as sampler:
    run_work()  # your workload

window = TelemetryWindow(sampler.buffered())
print("GPU 0 peak VRAM:", window.peak_vram_bytes(0))
print("GPU 0 energy estimate (J):", window.energy_joules(0))
print("Power samples used:", window.supported_sample_count("energy_joules", 0))
```

Two GPU backends read the same devices and return identical values: the always-available
`nvidia-smi` command, and NVML through the optional `pynvml` extra, which is selected automatically
when installed and removes the per-sample subprocess. One conformance suite runs against both.

Energy is always a telemetry-derived estimate, never hardware instrumentation. No derived method
aggregates devices: each GPU figure takes a `gpu_index` and describes only that device.

## Documentation

Project documentation lives under [`docs/`](docs/README.md). Start with [`docs/README.md`](docs/README.md).

| Read this | For |
|---|---|
| [docs/packages/sweatmeter/spec.md](docs/packages/sweatmeter/spec.md) | Purpose, scope, non-goals, public contracts, configuration, acceptance criteria |
| [docs/packages/sweatmeter/development-plan.md](docs/packages/sweatmeter/development-plan.md) | The phased build plan: goals, work, tests, acceptance criteria per phase |
| [docs/quickstart.md](docs/quickstart.md) | Snapshot, profile, sampler, deterministic-test, and derived-metric examples |

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest -m "not live and not performance"
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full workflow and [`SECURITY.md`](SECURITY.md) for
how to report a vulnerability.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

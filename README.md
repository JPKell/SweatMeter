# SweatMeter

CPU/RAM/GPU/VRAM/thermal/power telemetry and static machine profiling; degrades honestly to Unsupported rather than fabricating a value.

**Status:** Phases 1–2 implemented. Linux host readers provide CPU, memory, load, disk, temperature,
and static host facts. The NVIDIA reader provides per-device utilization, VRAM, temperatures, power,
fan, clocks, throttle reasons, and static GPU facts through an injectable, bounded `nvidia-smi`
boundary. Collection/background sampling and derived metrics remain scheduled for later phases in
the [development plan](docs/packages/sweatmeter/development-plan.md).

Part of the **Local AI Suite** — see [docs/architecture/executive-summary.md](docs/architecture/executive-summary.md)
for how SweatMeter fits with the suite's other applications and packages.

## Install

```bash
pip install sweatmeter
```

## Quickstart

```python
from baseaicore import is_supported
from sweatmeter import LinuxHostReader, NvidiaSmiReader

reader = LinuxHostReader()
reader.cpu_percent()  # prime the cumulative CPU counters
cpu_percent = reader.cpu_percent()
memory = reader.memory()

if is_supported(cpu_percent):
    print(f"CPU: {cpu_percent:.1f}%")
print(f"Memory: {memory.used_bytes} / {memory.total_bytes} bytes")

for gpu in NvidiaSmiReader().sample():
    print(f"GPU {gpu.index}: {gpu.utilization_percent}% / {gpu.vram_used_bytes} bytes VRAM")
```

Every source and command runner is injectable, so these parsers can be tested without using the host
or a GPU on the test machine. An absent tool, unreadable source, malformed value, or unsupported
sensor degrades honestly: collections become empty and individual measurements become BaseAiCore's
explicit `UNSUPPORTED` value, never zero.

## Documentation

This repository carries its own copy of the relevant suite documentation under [`docs/`](docs/README.md),
so it can be read and implemented independently of the other eight suite repositories. Start with
[`docs/README.md`](docs/README.md).

| Read this | For |
|---|---|
| [docs/packages/sweatmeter/spec.md](docs/packages/sweatmeter/spec.md) | Purpose, scope, non-goals, public contracts, configuration, acceptance criteria |
| [docs/packages/sweatmeter/development-plan.md](docs/packages/sweatmeter/development-plan.md) | The phased build plan: goals, work, tests, acceptance criteria per phase |
| [docs/standards/](docs/standards/) | Coding, testing, security, API, database and packaging standards every phase follows |
| [docs/adr/](docs/adr/README.md) | The architectural decisions this design rests on |

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

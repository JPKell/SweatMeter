# SweatMeter Quickstart

SweatMeter reports unavailable telemetry as BaseAiCore's explicit `UNSUPPORTED` sentinel. It never
turns a missing sensor into zero, and every derived GPU figure describes one device.

## Install

```bash
python -m pip install sweatmeter

# Optional, and worth it wherever you sample continuously: the NVML backend reads NVIDIA GPUs
# in-process instead of running `nvidia-smi` twice per snapshot.
python -m pip install "sweatmeter[pynvml]"
```

## One snapshot

```python
from baseaicore import is_supported
from sweatmeter import TelemetryCollector

collector = TelemetryCollector()
snapshot = collector.snapshot()

if is_supported(snapshot.cpu_percent):
    print(f"CPU: {snapshot.cpu_percent:.1f}%")
if is_supported(snapshot.ram_used_bytes):
    print(f"RAM used: {snapshot.ram_used_bytes} bytes")

for gpu in snapshot.gpus:
    print(f"GPU {gpu.index}: {gpu.utilization_percent}%")
    print(f"GPU {gpu.index} VRAM used: {gpu.vram_used_bytes} bytes")

for field, reason in snapshot.unavailable_reasons().items():
    print(f"{field}: unavailable ({reason})")
```

The first CPU and disk-rate readings are normally `UNSUPPORTED`: Linux exposes cumulative counters,
so a second reading is required before a rate exists.

## Choosing a GPU backend

```python
from sweatmeter import GpuBackend, create_gpu_reader

gpu = create_gpu_reader()                            # NVML when installed, else nvidia-smi
gpu = create_gpu_reader(prefer=GpuBackend.PYNVML)    # explicit; raises if the extra is absent
gpu = create_gpu_reader(prefer=GpuBackend.NVIDIA_SMI)
```

Both backends read the same devices and return identical `GpuSample` and `GpuProfile` values, so
switching is a performance decision and never a data one — one conformance suite runs against both
to keep it that way. The NVML backend holds the library open; call `close()` or use it as a context
manager when a process outlives its telemetry.

## Stable machine profile

```python
profile = collector.machine_profile()
print(profile.machine_fingerprint)
print(profile.cpu_model, profile.ram_bytes)
for gpu in profile.gpus:
    print(gpu.index, gpu.name, gpu.uuid, gpu.vram_total_bytes)
```

Driver, CUDA, OS/kernel, Python, and storage facts are recorded, but excluded from the machine
fingerprint. Upgrading a driver therefore does not create a new machine identity.

## Background sampling

```python
from sweatmeter import TelemetrySampler

with TelemetrySampler(collector, interval_seconds=1.0, buffer_size=120) as sampler:
    run_workload()

latest = sampler.latest()
print("latest age:", sampler.latest_age_seconds())
samples = sampler.buffered()
```

The worker starts immediately, follows monotonic deadlines, skips missed deadlines rather than
bursting, isolates callback exceptions, and stops on every context-manager exit. The optional ring
buffer is bounded; without `buffer_size`, only the newest snapshot is retained.

## Derived per-device metrics

```python
from baseaicore import is_supported
from sweatmeter import TelemetryWindow

window = TelemetryWindow(samples)
gpu_index = 0
energy = window.energy_joules(gpu_index)

print("peak VRAM:", window.peak_vram_bytes(gpu_index))
print("mean power:", window.mean_power_watts(gpu_index))
print("maximum temperature:", window.max_temperature_c(gpu_index))
print("energy estimate:", energy)
print("integrated intervals:", window.supported_sample_count("energy_joules", gpu_index))
print("throttling:", window.suspected_throttling(gpu_index))
```

`energy_joules()` is a telemetry-derived estimate: each supported power sample is multiplied by the
actual time until the next snapshot. Unsupported intervals are excluded rather than counted as zero.
The final sample has no following duration and contributes no energy. Results for multiple GPUs are
queried separately and are never summed by SweatMeter.

## Deterministic consumer tests

```python
from sweatmeter import TelemetryCollector
from sweatmeter.testing import HostReading, ScriptedGpuReader, ScriptedHostReader

host = ScriptedHostReader([HostReading(cpu_percent=25.0)])
gpu = ScriptedGpuReader([[]])
snapshot = TelemetryCollector(host=host, gpu=gpu).snapshot()
assert snapshot.cpu_percent == 25.0
```

`FaultInjectingReader(wrapped, fail="memory")` can prove a consumer handles one failed reader
operation without monkeypatching. Scripts raise on exhaustion so stale readings are never silently
repeated. `NullHostReader` and `NullGpuReader` are importable from `sweatmeter.testing` as well as
from `sweatmeter`, because they serve both as test doubles and as the production degradation path.

Support varies by platform and backend; measure overhead in the target environment.

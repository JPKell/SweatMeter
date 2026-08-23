# Platform Support

SweatMeter 0.4.0 isolates platform-specific access behind `HostReader` and `GpuReader`. Unsupported
environments remain usable through explicit null readers and never receive fabricated zero values.

| Surface | Linux x86-64 | Linux ARM64 | Windows 11 | macOS 14+ |
|---|---|---|---|---|
| CPU utilization/load | Supported via `/proc` | Best effort; fixture-tested | Tier-3 stub | Tier-3 stub |
| RAM and process RSS | Supported via `/proc` | Best effort; fixture-tested | Tier-3 stub | Tier-3 stub |
| CPU temperature | `/sys` thermal/hwmon, sensor path cached and rediscovered on failure | Hardware-dependent | Tier-3 stub | Tier-3 stub |
| Disk throughput | Supported via `/proc/diskstats` | Best effort; fixture-tested | Tier-3 stub | Tier-3 stub |
| Static host profile | Supported | Best effort | Tier-3 stub | Tier-3 stub |
| NVIDIA GPU (`nvidia-smi`) | When driver and device are available | Same | `nvidia-smi.exe` when available | Normally unavailable |
| NVIDIA GPU (NVML, `pynvml` extra) | When the extra and the driver are installed | Same | Same | Normally unavailable |
| Window metrics | Platform-independent | Platform-independent | Platform-independent | Platform-independent |

## Support tiers

- Tier 1: Linux x86-64. Release-blocking and covered by injected `/proc`, `/sys`, and NVIDIA output
  fixtures plus a real-host live smoke test.
- Tier 2: Linux ARM64. The parsers accept ARM CPU layouts, but hardware-specific thermal coverage is
  best effort.
- Tier 3: Windows and macOS. `create_host_reader()` returns `WindowsHostReader` or
  `DarwinHostReader`; every method raises a useful `UnsupportedPlatformError`. Construct
  `NullHostReader()` when the application can continue without host telemetry.

## Degradation

```python
from baseaicore import UnsupportedPlatformError
from sweatmeter import NullHostReader, TelemetryCollector, create_host_reader

try:
    host = create_host_reader()
except UnsupportedPlatformError:
    host = NullHostReader()

snapshot = TelemetryCollector(host=host).snapshot()
```

`create_gpu_reader()` selects NVML when the optional `pynvml` extra is importable and the
`nvidia-smi` command otherwise. Neither choice touches a device at construction time, and neither
changes the values a consumer sees: the two backends are held to one conformance suite.

`NullHostReader` returns `UNSUPPORTED` for every measurement with reason
`platform_unsupported`. `NullGpuReader` reports unavailable and returns empty live/static device
collections. A missing or failing `nvidia-smi` is retried on the next operation and records whether
the tool was absent, timed out, failed, or returned malformed data.

## Privileges and side effects

Collection is read-only. It reads `/proc` and `/sys`, and invokes a resolved `nvidia-smi` executable
with an explicit argument list, `shell=False`, captured UTF-8 output, a timeout, and an output-size
limit. The NVML backend calls only NVML's read operations — it never sets a clock, a fan curve or a
power limit — and starts no process at all. SweatMeter performs no network access, fan/clock/power
control, or persistence.

Future native Windows readers require PDH/WMI; macOS readers require `host_processor_info`, `sysctl`,
and potentially privileged SMC/powermetrics access. Those APIs are deliberately not approximated by
the current tier-3 stubs.

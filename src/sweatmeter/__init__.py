"""Honest, injectable host and NVIDIA telemetry for the Local AI Suite.

The public surface includes platform readers, complete non-raising snapshots, static machine
profiles, and bounded background sampling. Derived windows arrive in the next development phase.
"""

from sweatmeter.__about__ import __version__
from sweatmeter.collector import TelemetryCollector
from sweatmeter.platform import (
    GpuBackend,
    NullGpuReader,
    NullHostReader,
    create_gpu_reader,
    create_host_reader,
)
from sweatmeter.readers import (
    DarwinHostReader,
    GpuReader,
    HostReader,
    LinuxHostReader,
    NvidiaSmiReader,
    ParsedCell,
    SubprocessRunner,
    WindowsHostReader,
    parse_cpuinfo,
    parse_diskstats,
    parse_loadavg,
    parse_meminfo,
    parse_nvidia_csv,
    parse_proc_stat,
    read_block_devices,
    read_hwmon,
    read_thermal_zones,
)
from sweatmeter.sampler import TelemetrySampler
from sweatmeter.types import (
    DiskThroughput,
    GpuSample,
    HostFacts,
    MemoryReading,
    ReportedText,
    TelemetrySnapshot,
)

__all__ = [
    "DarwinHostReader",
    "DiskThroughput",
    "GpuBackend",
    "GpuReader",
    "GpuSample",
    "HostFacts",
    "HostReader",
    "LinuxHostReader",
    "MemoryReading",
    "NullGpuReader",
    "NullHostReader",
    "NvidiaSmiReader",
    "ParsedCell",
    "ReportedText",
    "SubprocessRunner",
    "TelemetryCollector",
    "TelemetrySampler",
    "TelemetrySnapshot",
    "WindowsHostReader",
    "__version__",
    "create_gpu_reader",
    "create_host_reader",
    "parse_cpuinfo",
    "parse_diskstats",
    "parse_loadavg",
    "parse_meminfo",
    "parse_nvidia_csv",
    "parse_proc_stat",
    "read_block_devices",
    "read_hwmon",
    "read_thermal_zones",
]

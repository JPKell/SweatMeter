"""Honest, injectable host and NVIDIA telemetry for the Local AI Suite.

Phases 1 and 2 provide Linux host and NVIDIA GPU readers. The collector, background sampler, and
derived windows arrive in later development-plan phases.
"""

from sweatmeter.__about__ import __version__
from sweatmeter.readers import (
    GpuReader,
    HostReader,
    LinuxHostReader,
    NvidiaSmiReader,
    ParsedCell,
    SubprocessRunner,
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
from sweatmeter.types import DiskThroughput, GpuSample, HostFacts, MemoryReading, ReportedText

__all__ = [
    "DiskThroughput",
    "GpuReader",
    "GpuSample",
    "HostFacts",
    "HostReader",
    "LinuxHostReader",
    "MemoryReading",
    "NvidiaSmiReader",
    "ParsedCell",
    "ReportedText",
    "SubprocessRunner",
    "__version__",
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

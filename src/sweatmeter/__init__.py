"""Honest, injectable host telemetry for the Local AI Suite.

Phase 1 provides Linux CPU, memory, load, disk, temperature, and static host readers. GPU telemetry,
the collector, background sampler, and derived windows arrive in later development-plan phases.
"""

from sweatmeter.__about__ import __version__
from sweatmeter.readers import (
    HostReader,
    LinuxHostReader,
    parse_cpuinfo,
    parse_diskstats,
    parse_loadavg,
    parse_meminfo,
    parse_proc_stat,
    read_block_devices,
    read_hwmon,
    read_thermal_zones,
)
from sweatmeter.types import DiskThroughput, HostFacts, MemoryReading, ReportedText

__all__ = [
    "DiskThroughput",
    "HostFacts",
    "HostReader",
    "LinuxHostReader",
    "MemoryReading",
    "ReportedText",
    "__version__",
    "parse_cpuinfo",
    "parse_diskstats",
    "parse_loadavg",
    "parse_meminfo",
    "parse_proc_stat",
    "read_block_devices",
    "read_hwmon",
    "read_thermal_zones",
]

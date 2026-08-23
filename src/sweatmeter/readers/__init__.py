"""Platform reader interfaces, implementations, and explicit tier-3 stubs."""

from sweatmeter.readers.darwin import DarwinHostReader
from sweatmeter.readers.linux import (
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
from sweatmeter.readers.nvidia import (
    NvidiaSmiReader,
    ParsedCell,
    SubprocessRunner,
    parse_nvidia_csv,
)
from sweatmeter.readers.nvml import (
    NvmlBinding,
    NvmlGpuReader,
    load_nvml_binding,
    nvml_binding_available,
)
from sweatmeter.readers.protocols import GpuReader, HostReader
from sweatmeter.readers.windows import WindowsHostReader

__all__ = [
    "DarwinHostReader",
    "GpuReader",
    "HostReader",
    "LinuxHostReader",
    "NvidiaSmiReader",
    "NvmlBinding",
    "NvmlGpuReader",
    "ParsedCell",
    "SubprocessRunner",
    "WindowsHostReader",
    "load_nvml_binding",
    "nvml_binding_available",
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

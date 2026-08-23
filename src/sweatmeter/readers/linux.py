"""Linux host telemetry from injectable ``/proc`` text and ``/sys`` roots.

The parsers are deliberately small and independent of the machine running the tests. Stateful
rates live in :class:`LinuxHostReader`; the parsing functions consume fixture text and never raise
for malformed content.
"""

from __future__ import annotations

import math
import platform as stdlib_platform
import re
import socket
import time
from collections.abc import Callable, Collection
from pathlib import Path

from baseaicore import UNSUPPORTED, Measurement, StorageDevice, Unsupported, is_supported

from sweatmeter.safe import _safe
from sweatmeter.types import DiskThroughput, HostFacts, MemoryReading, ReportedText

type TextReader = Callable[[Path], str]
type MonotonicClock = Callable[[], float]

__all__ = [
    "LinuxHostReader",
    "parse_cpuinfo",
    "parse_diskstats",
    "parse_loadavg",
    "parse_meminfo",
    "parse_proc_stat",
    "read_block_devices",
    "read_hwmon",
    "read_thermal_zones",
]

_KIB = 1024
_SECTOR_BYTES = 512
_PROC_STAT_MIN_FIELDS = 5
_PROC_STAT_IDLE_INDEX = 3
_PROC_STAT_IOWAIT_INDEX = 4
_DISKSTATS_NAME_INDEX = 2
_DISKSTATS_MIN_FIELDS = 10
_DISKSTATS_READ_SECTORS_INDEX = 5
_DISKSTATS_WRITE_SECTORS_INDEX = 9
_MIN_TEMPERATURE_C = -273.15
_MAX_TEMPERATURE_C = 1000.0
_WHOLE_DEVICE = re.compile(
    r"^(?:sd[a-z]+|hd[a-z]+|vd[a-z]+|xvd[a-z]+|nvme\d+n\d+|mmcblk\d+|dasd[a-z]+)$"
)
_PREFERRED_THERMAL_ZONE_TYPES = (
    "x86_pkg_temp",
    "cpu_thermal",
    "cpu-thermal",
    "coretemp",
    "k10temp",
    "acpitz",
)
_CPU_HWMON_DRIVERS = frozenset(
    {"coretemp", "k10temp", "zenpower", "cpu_thermal", "cpu-thermal", "acpitz"}
)
_NON_CPU_HWMON_DRIVERS = frozenset({"amdgpu", "nouveau", "nvidia", "nvme"})


def _read_utf8(path: Path) -> str:
    """Read a kernel text pseudo-file with deterministic decoding."""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_proc_stat(text: str) -> tuple[int, int] | Unsupported:
    """Parse aggregate CPU counters from ``/proc/stat``.

    Returns ``(total_jiffies, idle_jiffies)``. Guest counters are excluded from the total because
    Linux already includes them in the user/nice counters; summing every column would double-count
    guest time. Per-core lines are ignored because the aggregate ``cpu`` line already represents
    all cores.
    """
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] != "cpu":
            continue
        if len(fields) < _PROC_STAT_MIN_FIELDS:
            return UNSUPPORTED
        try:
            counters = tuple(int(value) for value in fields[1:])
        except ValueError:
            return UNSUPPORTED
        if any(counter < 0 for counter in counters):
            return UNSUPPORTED

        # user, nice, system, idle, iowait, irq, softirq, steal; guest fields follow.
        accounting_counters = counters[:8]
        total = sum(accounting_counters)
        idle = counters[_PROC_STAT_IDLE_INDEX] + (
            counters[_PROC_STAT_IOWAIT_INDEX] if len(counters) > _PROC_STAT_IOWAIT_INDEX else 0
        )
        return (total, idle)
    return UNSUPPORTED


def _meminfo_values(text: str) -> dict[str, int]:
    """Return valid ``/proc/meminfo`` quantities normalized from KiB to bytes."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.split()
        if not fields:
            continue
        try:
            amount = int(fields[0])
        except ValueError:
            continue
        if amount < 0 or (len(fields) > 1 and fields[1].lower() != "kb"):
            continue
        values[key.strip()] = amount * _KIB
    return values


def parse_meminfo(text: str) -> MemoryReading:
    """Parse total, available, and used RAM from ``/proc/meminfo``.

    Kernels before Linux 3.14 may omit ``MemAvailable``. In that case the documented fallback is
    ``MemFree + Buffers + Cached + SReclaimable - Shmem``, clamped to the known total. It is an
    approximation, but it is made from reported kernel values rather than a fabricated zero.
    """
    values = _meminfo_values(text)
    total: Measurement = values.get("MemTotal", UNSUPPORTED)
    available: Measurement = values.get("MemAvailable", UNSUPPORTED)

    if available is UNSUPPORTED and "MemFree" in values:
        fallback: int | float = (
            values["MemFree"]
            + values.get("Buffers", 0)
            + values.get("Cached", 0)
            + values.get("SReclaimable", 0)
            - values.get("Shmem", 0)
        )
        fallback = max(0, fallback)
        if is_supported(total):
            fallback = min(fallback, total)
        available = fallback

    used: Measurement = UNSUPPORTED
    if is_supported(total) and is_supported(available) and available <= total:
        used = total - available

    return MemoryReading(total_bytes=total, available_bytes=available, used_bytes=used)


def parse_loadavg(text: str) -> Measurement:
    """Parse the one-minute load average from ``/proc/loadavg``."""
    fields = text.split()
    if not fields:
        return UNSUPPORTED
    try:
        load = float(fields[0])
    except ValueError:
        return UNSUPPORTED
    return load if math.isfinite(load) and load >= 0 else UNSUPPORTED


def parse_diskstats(
    text: str, *, devices: Collection[str] | None = None
) -> tuple[int, int] | Unsupported:
    """Parse cumulative whole-device byte counters from ``/proc/diskstats``.

    Args:
        text: Complete ``/proc/diskstats`` content.
        devices: Optional exact allowlist of whole block-device names. When omitted, known Linux
            physical-device naming patterns are used. Partitions and virtual loop/device-mapper
            devices are excluded to prevent double-counting.

    Returns:
        Cumulative ``(bytes_read, bytes_written)``, or ``UNSUPPORTED`` when a selected device row
        is malformed or no selected whole device is present.
    """
    selected = set(devices) if devices is not None else None
    seen: set[str] = set()
    read_sectors = 0
    write_sectors = 0

    for line in text.splitlines():
        fields = line.split()
        if len(fields) <= _DISKSTATS_NAME_INDEX:
            continue
        name = fields[_DISKSTATS_NAME_INDEX]
        is_selected = name in selected if selected is not None else _WHOLE_DEVICE.fullmatch(name)
        if not is_selected:
            continue
        if len(fields) < _DISKSTATS_MIN_FIELDS:
            return UNSUPPORTED
        try:
            device_read_sectors = int(fields[_DISKSTATS_READ_SECTORS_INDEX])
            device_write_sectors = int(fields[_DISKSTATS_WRITE_SECTORS_INDEX])
        except ValueError:
            return UNSUPPORTED
        if device_read_sectors < 0 or device_write_sectors < 0:
            return UNSUPPORTED
        seen.add(name)
        read_sectors += device_read_sectors
        write_sectors += device_write_sectors

    if not seen or (selected is not None and seen != selected):
        return UNSUPPORTED
    return (read_sectors * _SECTOR_BYTES, write_sectors * _SECTOR_BYTES)


def parse_cpuinfo(text: str) -> HostFacts:
    """Parse CPU model and physical/logical core counts from ``/proc/cpuinfo``.

    X86 physical cores are distinct ``(physical id, core id)`` pairs, so SMT siblings collapse to
    one core while identically numbered cores in different sockets remain distinct. ARM fixtures
    commonly have no ``model name``; ``Hardware`` and ``Processor`` are accepted as model fallbacks.
    A physical count is not guessed when the kernel exposes no topology identifiers.
    """
    model_candidates: dict[str, str] = {}
    logical_count = 0
    physical_core_ids: set[tuple[str, str]] = set()

    for block in re.split(r"\n\s*\n", text.strip()):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator and value.strip():
                fields[key.strip()] = value.strip()
        if "processor" in fields:
            logical_count += 1
        for model_key in ("model name", "Processor", "Hardware", "Model"):
            if model_key in fields:
                model_candidates.setdefault(model_key, fields[model_key])
        physical_id = fields.get("physical id")
        core_id = fields.get("core id")
        if physical_id is not None and core_id is not None:
            physical_core_ids.add((physical_id, core_id))

    model: ReportedText = UNSUPPORTED
    for key in ("model name", "Hardware", "Processor", "Model"):
        candidate = model_candidates.get(key)
        if candidate is not None:
            model = candidate
            break

    physical: Measurement = len(physical_core_ids) if physical_core_ids else UNSUPPORTED
    logical: Measurement = logical_count if logical_count else UNSUPPORTED
    return HostFacts(cpu_model=model, physical_cores=physical, logical_cores=logical)


def _children(root: Path, pattern: str) -> tuple[Path, ...]:
    """Return sorted matching children, degrading an unreadable directory to empty."""
    children = _safe(lambda: tuple(root.glob(pattern)), ())
    return tuple(sorted(children))


def _optional_text(path: Path, read_text: TextReader) -> str | None:
    """Read and normalize optional kernel text."""
    value = _safe(lambda: read_text(path), None)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _millidegrees(text: str | None) -> float | None:
    """Convert a Linux millidegree value to a plausible Celsius reading."""
    if text is None:
        return None
    try:
        value = float(text) / 1000.0
    except ValueError:
        return None
    if not math.isfinite(value) or not _MIN_TEMPERATURE_C <= value <= _MAX_TEMPERATURE_C:
        return None
    return value


def read_thermal_zones(root: Path, *, read_text: TextReader = _read_utf8) -> Measurement:
    """Read the preferred CPU temperature from a ``thermal_zone*`` tree.

    Permission errors, missing files, and malformed individual zones are skipped. Known CPU/package
    zone types take precedence over generic ACPI zones and directory order.
    """
    candidates: list[tuple[int, str, float]] = []
    for zone in _children(root, "thermal_zone*"):
        zone_type = (_optional_text(zone / "type", read_text) or "").lower()
        temperature = _millidegrees(_optional_text(zone / "temp", read_text))
        if temperature is None:
            continue
        try:
            priority = _PREFERRED_THERMAL_ZONE_TYPES.index(zone_type)
        except ValueError:
            priority = len(_PREFERRED_THERMAL_ZONE_TYPES)
        candidates.append((priority, zone.name, temperature))
    if not candidates:
        return UNSUPPORTED
    return min(candidates)[2]


def read_hwmon(root: Path, *, read_text: TextReader = _read_utf8) -> Measurement:
    """Read a CPU temperature from a ``hwmon*`` tree.

    CPU driver names and package/Tctl/Tdie labels are preferred. Sensors explicitly belonging to
    GPU or NVMe drivers are ignored; an unknown driver remains a last-resort candidate because some
    kernel/platform combinations omit ``name``.
    """
    candidates: list[tuple[int, str, float]] = []
    for hwmon in _children(root, "hwmon*"):
        driver = (_optional_text(hwmon / "name", read_text) or "").lower()
        if driver in _NON_CPU_HWMON_DRIVERS:
            continue
        for temperature_file in _children(hwmon, "temp*_input"):
            temperature = _millidegrees(_optional_text(temperature_file, read_text))
            if temperature is None:
                continue
            label_file = temperature_file.with_name(
                temperature_file.name.removesuffix("_input") + "_label"
            )
            label = (_optional_text(label_file, read_text) or "").lower()
            cpu_label = any(token in label for token in ("package", "tctl", "tdie", "cpu"))
            priority = 0 if driver in _CPU_HWMON_DRIVERS or cpu_label else 1
            candidates.append((priority, str(temperature_file), temperature))
    if not candidates:
        return UNSUPPORTED
    return min(candidates)[2]


def read_block_devices(
    root: Path, *, read_text: TextReader = _read_utf8
) -> tuple[StorageDevice, ...]:
    """Read physical whole-device facts from a ``/sys/block`` tree.

    Unreadable attributes degrade independently: the device still appears, its size becomes
    ``UNSUPPORTED``, and optional model/rotational values become ``None`` as required by
    BaseAiCore's ``StorageDevice`` contract.
    """
    devices: list[StorageDevice] = []
    for device in _children(root, "*"):
        if _WHOLE_DEVICE.fullmatch(device.name) is None:
            continue

        size_text = _optional_text(device / "size", read_text)
        size_bytes: Measurement = UNSUPPORTED
        if size_text is not None:
            try:
                sectors = int(size_text)
            except ValueError:
                sectors = -1
            if sectors >= 0:
                size_bytes = sectors * _SECTOR_BYTES

        rotational_text = _optional_text(device / "queue" / "rotational", read_text)
        rotational = (
            {"0": False, "1": True}.get(rotational_text) if rotational_text is not None else None
        )
        model = _optional_text(device / "device" / "model", read_text)
        devices.append(
            StorageDevice(
                name=device.name,
                size_bytes=size_bytes,
                model=model,
                rotational=rotational,
            )
        )
    return tuple(devices)


def _reported_text(fn: Callable[[], str]) -> ReportedText:
    """Return non-empty normalized system text or ``UNSUPPORTED``."""
    value = _safe(fn)
    if not isinstance(value, str):
        return UNSUPPORTED
    stripped = value.strip()
    return stripped if stripped else UNSUPPORTED


class LinuxHostReader:
    """Read Linux host metrics from injectable ``/proc`` and ``/sys`` roots.

    CPU utilization and disk throughput are deltas. Their first calls return ``UNSUPPORTED`` rather
    than the misleading value zero. Kernel counter rollback/wrap also returns ``UNSUPPORTED`` for
    that interval and installs a fresh baseline for the following call.
    """

    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        sys_root: Path = Path("/sys"),
        read_text: TextReader = _read_utf8,
        clock: MonotonicClock = time.monotonic,
        disk_devices: Collection[str] | None = None,
    ) -> None:
        """Configure injectable kernel-data roots, text reader, and monotonic clock."""
        self._proc_root = proc_root
        self._sys_root = sys_root
        self._read_text = read_text
        self._clock = clock
        self._disk_devices = frozenset(disk_devices) if disk_devices is not None else None
        self._previous_cpu: tuple[int, int] | None = None
        self._previous_disk: tuple[float, int, int] | None = None

    def _proc_text(self, name: str) -> str:
        return self._read_text(self._proc_root / name)

    def cpu_percent(self) -> Measurement:
        """Return aggregate utilization since the preceding valid CPU-counter call."""
        parsed = _safe(lambda: parse_proc_stat(self._proc_text("stat")))
        if not is_supported(parsed):
            return UNSUPPORTED

        previous, self._previous_cpu = self._previous_cpu, parsed
        if previous is None:
            return UNSUPPORTED
        total_delta = parsed[0] - previous[0]
        idle_delta = parsed[1] - previous[1]
        if total_delta <= 0 or idle_delta < 0 or idle_delta > total_delta:
            return UNSUPPORTED
        return min(100.0, max(0.0, (total_delta - idle_delta) * 100.0 / total_delta))

    def load_average_1m(self) -> Measurement:
        """Return Linux's one-minute runnable-task load average."""
        return _safe(lambda: parse_loadavg(self._proc_text("loadavg")))

    def memory(self) -> MemoryReading:
        """Return RAM quantities, degrading an unreadable source field-by-field."""
        return _safe(lambda: parse_meminfo(self._proc_text("meminfo")), MemoryReading())

    def cpu_temperature(self) -> Measurement:
        """Read thermal zones first, then fall back to hwmon."""
        thermal = read_thermal_zones(
            self._sys_root / "class" / "thermal", read_text=self._read_text
        )
        if is_supported(thermal):
            return thermal
        return read_hwmon(self._sys_root / "class" / "hwmon", read_text=self._read_text)

    def disk_throughput(self) -> DiskThroughput:
        """Return disk rates since the preceding valid counter sample."""
        parsed = _safe(
            lambda: parse_diskstats(self._proc_text("diskstats"), devices=self._disk_devices)
        )
        now = _safe(self._clock)
        if not is_supported(parsed) or not isinstance(now, (int, float)) or not math.isfinite(now):
            return DiskThroughput()

        previous, self._previous_disk = self._previous_disk, (float(now), *parsed)
        if previous is None:
            return DiskThroughput()
        elapsed = float(now) - previous[0]
        if elapsed <= 0:
            return DiskThroughput()

        read_rate: Measurement = UNSUPPORTED
        write_rate: Measurement = UNSUPPORTED
        if parsed[0] >= previous[1]:
            read_rate = (parsed[0] - previous[1]) / elapsed
        if parsed[1] >= previous[2]:
            write_rate = (parsed[1] - previous[2]) / elapsed
        return DiskThroughput(read_rate, write_rate)

    def static_facts(self) -> HostFacts:
        """Return static CPU, RAM, OS, runtime, and block-device facts."""
        cpu = _safe(lambda: parse_cpuinfo(self._proc_text("cpuinfo")), HostFacts())
        memory = self.memory()
        return HostFacts(
            hostname=_reported_text(socket.gethostname),
            os_name=_reported_text(stdlib_platform.system),
            os_version=_reported_text(stdlib_platform.version),
            kernel=_reported_text(stdlib_platform.release),
            architecture=_reported_text(stdlib_platform.machine),
            cpu_model=cpu.cpu_model,
            physical_cores=cpu.physical_cores,
            logical_cores=cpu.logical_cores,
            ram_bytes=memory.total_bytes,
            storage=read_block_devices(self._sys_root / "block", read_text=self._read_text),
            python_version=_reported_text(stdlib_platform.python_version),
        )

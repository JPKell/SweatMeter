"""NVIDIA telemetry from bounded, injectable ``nvidia-smi`` CSV commands.

The command boundary is read-only and never uses a shell. CSV rows are converted to mappings keyed
by the requested NVIDIA field names before any metric is interpreted, so reordered queries and
optional columns cannot silently move a value into the wrong measurement.
"""

from __future__ import annotations

import csv
import logging
import math
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from baseaicore import (
    UNSUPPORTED,
    GpuProfile,
    GpuVendor,
    Measurement,
    Unsupported,
    ValidationError,
)

from sweatmeter.types import GpuSample

type ParsedCell = str | Unsupported
type ExecutableResolver = Callable[[str], str | None]

__all__ = ["NvidiaSmiReader", "ParsedCell", "SubprocessRunner", "parse_nvidia_csv"]

_LOGGER = logging.getLogger(__name__)
_MIB_BYTES = 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 5.0
_DEFAULT_MAX_OUTPUT_BYTES = 1_000_000
_CUDA_VERSION = re.compile(r"\bCUDA Version:\s*([0-9]+(?:\.[0-9]+)*)\b")
_UNAVAILABLE_MARKERS = frozenset(
    {
        "",
        "n/a",
        "[n/a]",
        "not available",
        "[not available]",
        "not supported",
        "[not supported]",
        "unsupported",
        "[unsupported]",
    }
)

_PROBE_FIELDS = ("index",)
_STATIC_FIELDS = ("index", "name", "uuid", "memory.total", "driver_version")
_COMPUTE_FIELDS = ("index", "compute_cap")
_SAMPLE_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "temperature.memory",
    "power.draw",
    "power.limit",
    "fan.speed",
    "clocks.current.sm",
    "clocks.current.memory",
)
_THROTTLE_FIELDS = (
    ("gpu_idle", "clocks_throttle_reasons.gpu_idle"),
    ("applications_clocks_setting", "clocks_throttle_reasons.applications_clocks_setting"),
    ("sw_power_cap", "clocks_throttle_reasons.sw_power_cap"),
    ("hw_slowdown", "clocks_throttle_reasons.hw_slowdown"),
    ("hw_thermal_slowdown", "clocks_throttle_reasons.hw_thermal_slowdown"),
    ("hw_power_brake_slowdown", "clocks_throttle_reasons.hw_power_brake_slowdown"),
    ("sync_boost", "clocks_throttle_reasons.sync_boost"),
    ("sw_thermal_slowdown", "clocks_throttle_reasons.sw_thermal_slowdown"),
)
_THROTTLE_COLUMNS = ("index", *(column for _name, column in _THROTTLE_FIELDS))


class SubprocessRunner(Protocol):
    """Callable shape used to inject the ``nvidia-smi`` process boundary in tests."""

    def __call__(  # noqa: PLR0913 — exact subprocess.run controls are part of the boundary contract
        self,
        args: Sequence[str],
        *,
        capture_output: bool,
        check: bool,
        encoding: str,
        errors: str,
        shell: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Run one explicitly bounded command and return its completed-process record."""
        ...


@dataclass(frozen=True, slots=True)
class _ParsedRows:
    """Parsed rows plus the count of CSV lines rejected as malformed."""

    rows: tuple[dict[str, ParsedCell], ...]
    malformed_count: int


def _parse_nvidia_csv(output: str, columns: Sequence[str]) -> _ParsedRows:
    """Parse independent CSV lines and retain every line that can be decoded."""
    parsed: list[dict[str, ParsedCell]] = []
    malformed_count = 0
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            cells = next(csv.reader([line], skipinitialspace=True, strict=True))
        except (csv.Error, StopIteration):
            malformed_count += 1
            continue

        normalized: list[ParsedCell] = []
        for cell in cells:
            stripped = cell.strip()
            normalized.append(
                UNSUPPORTED if stripped.casefold() in _UNAVAILABLE_MARKERS else stripped
            )
        # A short row simply lacks its trailing fields; extra output fields are ignored. The
        # resulting mapping is always consumed by name, never by a hard-coded numeric position.
        parsed.append(dict(zip(columns, normalized, strict=False)))
    return _ParsedRows(tuple(parsed), malformed_count)


def parse_nvidia_csv(output: str, columns: Sequence[str]) -> tuple[dict[str, ParsedCell], ...]:
    """Parse headerless ``nvidia-smi`` CSV into field-name mappings.

    Args:
        output: Text emitted by ``--format=csv,noheader,nounits``.
        columns: Field names in the exact order used in the corresponding ``--query-gpu`` option.
            Any query order is valid. Missing trailing cells remain absent from the mapping and
            extra trailing cells are ignored.

    Returns:
        Every independently parseable, non-blank row. NVIDIA's unavailable markers become the
        explicit ``UNSUPPORTED`` sentinel; one malformed line does not discard other rows.
    """
    return _parse_nvidia_csv(output, columns).rows


class NvidiaSmiReader:
    """Read per-device NVIDIA telemetry without fabricating unavailable values.

    The executable is resolved with ``shutil.which`` for every operation, and every command uses
    an argument list, ``shell=False``, a timeout, captured UTF-8 text, and an output-size limit
    before parsing. Failures affect only the current call; the next call resolves and invokes the
    tool again. Optional static fields and throttle reasons use separate commands so their absence
    cannot erase otherwise-valid GPU data.

    ``unavailable_reasons()`` describes degradation from the most recent public operation. Reader
    instances retain only that diagnostic mapping and are intended to be owned by one collector;
    concurrent use of one instance is not supported.

    Args:
        executable: Executable name or explicit path to resolve. Defaults to ``nvidia-smi``.
        runner: Injectable subprocess-compatible callable. Tests never need a GPU or process.
        resolver: Injectable executable resolver. Defaults to ``shutil.which``.
        timeout_seconds: Positive finite timeout supplied to every command.
        max_output_bytes: Positive byte limit applied before any command output is parsed.

    Raises:
        ValidationError: If a constructor setting is blank, non-positive, or non-finite.
    """

    def __init__(
        self,
        *,
        executable: str = "nvidia-smi",
        runner: SubprocessRunner = subprocess.run,
        resolver: ExecutableResolver = shutil.which,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        """Configure the executable boundary and its runtime and output bounds."""
        if not isinstance(executable, str) or not executable.strip():
            raise ValidationError(
                "NvidiaSmiReader.executable must be a non-empty command name or path.",
                details={"field": "executable"},
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValidationError(
                "NvidiaSmiReader.timeout_seconds must be a positive finite number; "
                f"got {timeout_seconds!r}.",
                details={"field": "timeout_seconds", "value": timeout_seconds},
            )
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0
        ):
            raise ValidationError(
                "NvidiaSmiReader.max_output_bytes must be a positive integer; "
                f"got {max_output_bytes!r}.",
                details={"field": "max_output_bytes", "value": max_output_bytes},
            )

        self._executable = executable
        self._runner = runner
        self._resolver = resolver
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes
        self._reasons: dict[str, str] = {}

    def unavailable_reasons(self) -> Mapping[str, str]:
        """Return a copy of degradation reasons from the most recent public operation."""
        return dict(sorted(self._reasons.items()))

    def available(self) -> bool:
        """Return whether at least one valid NVIDIA GPU index can be queried now.

        The result is a live probe, not a cached executable check. An installed tool with no
        working driver or visible GPU is therefore unavailable, and a later call retries it.
        """
        self._reasons.clear()
        rows = self._query(_PROBE_FIELDS, reason_key="gpu")
        indexed = self._rows_by_index(rows, scope="gpu.probe")
        if indexed:
            return True
        self._reasons.setdefault("gpu", "malformed_csv" if rows else "no_nvidia_gpus")
        return False

    def static_info(self) -> tuple[GpuProfile, ...]:
        """Return static identity and capacity information for every visible NVIDIA GPU.

        CUDA version comes from the tool banner and compute capability from a separate query.
        Either optional command may fail while the core GPU profiles remain available.
        """
        self._reasons.clear()
        core_rows = self._rows_by_index(
            self._query(_STATIC_FIELDS, reason_key="gpu"), scope="gpu.static"
        )
        if not core_rows:
            self._reasons.setdefault("gpu", "malformed_csv" if self._reasons else "no_nvidia_gpus")
            return ()

        compute_rows = self._rows_by_index(
            self._query(_COMPUTE_FIELDS, reason_key="gpu.compute_capability"),
            scope="gpu.compute",
        )
        cuda_version = self._read_cuda_version()

        profiles: list[GpuProfile] = []
        for index, row in sorted(core_rows.items()):
            compute_row = compute_rows.get(index, {})
            profiles.append(
                GpuProfile(
                    index=index,
                    name=self._optional_text(row, "name", index=index, field="name"),
                    uuid=self._optional_text(row, "uuid", index=index, field="uuid"),
                    vram_total_bytes=self._measurement(
                        row,
                        "memory.total",
                        index=index,
                        field="vram_total_bytes",
                        mib_to_bytes=True,
                    ),
                    driver_version=self._optional_text(
                        row, "driver_version", index=index, field="driver_version"
                    ),
                    cuda_version=cuda_version,
                    compute_capability=self._optional_text(
                        compute_row,
                        "compute_cap",
                        index=index,
                        field="compute_capability",
                    ),
                    vendor=GpuVendor.NVIDIA,
                )
            )
        return tuple(profiles)

    def sample(self) -> tuple[GpuSample, ...]:
        """Return one live telemetry sample per visible NVIDIA GPU.

        A failed core query returns an empty tuple for this call. Parseable rows from partially
        malformed output are retained, and each missing, unsupported, malformed, or out-of-range
        sensor becomes ``UNSUPPORTED`` independently. The next call always retries the command.
        """
        self._reasons.clear()
        core_rows = self._rows_by_index(
            self._query(_SAMPLE_FIELDS, reason_key="gpu"), scope="gpu.sample"
        )
        if not core_rows:
            self._reasons.setdefault("gpu", "malformed_csv" if self._reasons else "no_nvidia_gpus")
            return ()

        throttle_rows = self._rows_by_index(
            self._query(_THROTTLE_COLUMNS, reason_key="gpu.throttle_reasons"),
            scope="gpu.throttle",
        )

        samples: list[GpuSample] = []
        for index, row in sorted(core_rows.items()):
            reasons, reasons_available = self._throttle_reasons(
                throttle_rows.get(index), index=index
            )
            samples.append(
                self._sample_from_row(
                    row,
                    index=index,
                    reasons=reasons,
                    reasons_available=reasons_available,
                )
            )
        return tuple(samples)

    def _invoke(  # noqa: PLR0911 — each boundary failure has a distinct reason code
        self, arguments: Sequence[str]
    ) -> tuple[str | None, str | None]:
        """Run one resolved command and return either bounded output or a reason code."""
        try:
            executable = self._resolver(self._executable)
        except PermissionError:
            _LOGGER.debug("NVIDIA executable resolution was denied", exc_info=True)
            return (None, "permission_denied")
        except OSError:
            _LOGGER.debug("NVIDIA executable resolution failed", exc_info=True)
            return (None, "nvidia_smi_not_found")
        if not executable:
            _LOGGER.debug("NVIDIA telemetry unavailable: %s was not found", self._executable)
            return (None, "nvidia_smi_not_found")

        command = [executable, *arguments]
        try:
            completed = self._runner(
                command,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                shell=False,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            _LOGGER.debug("NVIDIA telemetry command timed out", exc_info=True)
            return (None, "nvidia_smi_timeout")
        except FileNotFoundError:
            _LOGGER.debug("NVIDIA telemetry executable disappeared", exc_info=True)
            return (None, "nvidia_smi_not_found")
        except PermissionError:
            _LOGGER.debug("NVIDIA telemetry command was denied", exc_info=True)
            return (None, "permission_denied")
        except (OSError, subprocess.SubprocessError):
            _LOGGER.debug("NVIDIA telemetry command could not run", exc_info=True)
            return (None, "nvidia_smi_unavailable")

        if completed.returncode != 0:
            _LOGGER.debug("NVIDIA telemetry command exited %d", completed.returncode)
            return (None, "nvidia_smi_failed")
        output = completed.stdout or ""
        if len(output.encode("utf-8", errors="replace")) > self._max_output_bytes:
            _LOGGER.debug("NVIDIA telemetry output exceeded %d bytes", self._max_output_bytes)
            return (None, "nvidia_smi_output_too_large")
        return (output, None)

    def _query(
        self, fields: Sequence[str], *, reason_key: str
    ) -> tuple[dict[str, ParsedCell], ...]:
        """Run one named-field CSV query and record command or parse degradation."""
        output, failure = self._invoke(
            (f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")
        )
        if output is None:
            self._reasons[reason_key] = failure or "nvidia_smi_unavailable"
            return ()
        parsed = _parse_nvidia_csv(output, fields)
        if parsed.malformed_count:
            self._reasons.setdefault(reason_key, "malformed_csv")
        return parsed.rows

    def _rows_by_index(
        self, rows: Sequence[Mapping[str, ParsedCell]], *, scope: str
    ) -> dict[int, Mapping[str, ParsedCell]]:
        """Key valid rows by GPU index and reject malformed or duplicate identities."""
        indexed: dict[int, Mapping[str, ParsedCell]] = {}
        for row_number, row in enumerate(rows):
            raw_index = row.get("index")
            try:
                index = int(raw_index) if isinstance(raw_index, str) else -1
            except ValueError:
                index = -1
            if index < 0:
                self._reasons[f"{scope}.row.{row_number}.index"] = "malformed_value"
                continue
            if index in indexed:
                self._reasons[f"{scope}.row.{row_number}.index"] = "duplicate_gpu_index"
                continue
            indexed[index] = row
        return indexed

    def _optional_text(
        self,
        row: Mapping[str, ParsedCell],
        column: str,
        *,
        index: int,
        field: str,
    ) -> str | None:
        """Read optional text and record why it is missing."""
        value = row.get(column)
        reason_key = f"gpu.{index}.{field}"
        if value is None:
            self._reasons[reason_key] = "field_missing"
            return None
        if not isinstance(value, str):
            self._reasons[reason_key] = "sensor_unsupported"
            return None
        return value

    def _measurement(  # noqa: PLR0911, PLR0913 — every field failure keeps its own reason
        self,
        row: Mapping[str, ParsedCell],
        column: str,
        *,
        index: int,
        field: str,
        maximum: float | None = None,
        mib_to_bytes: bool = False,
    ) -> Measurement:
        """Read one non-negative finite numeric sensor with optional normalization."""
        value = row.get(column)
        reason_key = f"gpu.{index}.{field}"
        if value is None:
            self._reasons[reason_key] = "field_missing"
            return UNSUPPORTED
        if value is UNSUPPORTED:
            self._reasons[reason_key] = "sensor_unsupported"
            return UNSUPPORTED
        try:
            number = float(value)
        except ValueError:
            self._reasons[reason_key] = "malformed_value"
            return UNSUPPORTED
        if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
            self._reasons[reason_key] = "out_of_range"
            return UNSUPPORTED
        if not mib_to_bytes:
            return number
        bytes_value = number * _MIB_BYTES
        if not math.isfinite(bytes_value):
            self._reasons[reason_key] = "out_of_range"
            return UNSUPPORTED
        return int(bytes_value)

    def _read_cuda_version(self) -> str | None:
        """Read the CUDA version advertised in the ``nvidia-smi`` banner."""
        output, failure = self._invoke(())
        if output is None:
            self._reasons["gpu.cuda_version"] = failure or "nvidia_smi_unavailable"
            return None
        match = _CUDA_VERSION.search(output)
        if match is None:
            self._reasons["gpu.cuda_version"] = "field_missing"
            return None
        return match.group(1)

    def _throttle_reasons(
        self, row: Mapping[str, ParsedCell] | None, *, index: int
    ) -> tuple[tuple[str, ...], bool]:
        """Return active throttle reasons and whether a none-active result is knowable."""
        if row is None:
            self._reasons[f"gpu.{index}.throttle_reasons"] = "sensor_unsupported"
            return ((), False)

        active: list[str] = []
        recognized = False
        for name, column in _THROTTLE_FIELDS:
            value = row.get(column)
            if not isinstance(value, str):
                continue
            state = value.casefold()
            if state == "active":
                recognized = True
                active.append(name)
            elif state == "not active":
                recognized = True
        if not recognized:
            self._reasons[f"gpu.{index}.throttle_reasons"] = "sensor_unsupported"
        return (tuple(active), recognized)

    def _sample_from_row(
        self,
        row: Mapping[str, ParsedCell],
        *,
        index: int,
        reasons: tuple[str, ...],
        reasons_available: bool,
    ) -> GpuSample:
        """Build one normalized live sample from a field-name mapping."""
        return GpuSample(
            index=index,
            uuid=self._optional_text(row, "uuid", index=index, field="uuid"),
            utilization_percent=self._measurement(
                row, "utilization.gpu", index=index, field="utilization_percent", maximum=100
            ),
            memory_utilization_percent=self._measurement(
                row,
                "utilization.memory",
                index=index,
                field="memory_utilization_percent",
                maximum=100,
            ),
            vram_used_bytes=self._measurement(
                row, "memory.used", index=index, field="vram_used_bytes", mib_to_bytes=True
            ),
            vram_total_bytes=self._measurement(
                row, "memory.total", index=index, field="vram_total_bytes", mib_to_bytes=True
            ),
            temperature_c=self._measurement(
                row, "temperature.gpu", index=index, field="temperature_c"
            ),
            memory_temperature_c=self._measurement(
                row, "temperature.memory", index=index, field="memory_temperature_c"
            ),
            power_watts=self._measurement(row, "power.draw", index=index, field="power_watts"),
            power_limit_watts=self._measurement(
                row, "power.limit", index=index, field="power_limit_watts"
            ),
            fan_percent=self._measurement(
                row, "fan.speed", index=index, field="fan_percent", maximum=100
            ),
            core_clock_mhz=self._measurement(
                row, "clocks.current.sm", index=index, field="core_clock_mhz"
            ),
            memory_clock_mhz=self._measurement(
                row, "clocks.current.memory", index=index, field="memory_clock_mhz"
            ),
            throttle_reasons=reasons,
            throttle_reasons_available=reasons_available,
        )

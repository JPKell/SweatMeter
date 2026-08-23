"""Fixture and failure-path tests for the Phase 2 NVIDIA reader."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import pytest
from baseaicore import UNSUPPORTED, GpuProfile, GpuVendor, ValidationError

from sweatmeter import (
    GpuReader,
    GpuSample,
    NvidiaSmiReader,
    ParsedCell,
    SubprocessRunner,
    parse_nvidia_csv,
)


@dataclass(frozen=True, slots=True)
class RunnerCall:
    args: tuple[str, ...]
    capture_output: bool
    check: bool
    encoding: str
    errors: str
    shell: bool
    text: bool
    timeout: float


class ScriptedRunner:
    def __init__(
        self, outcomes: Sequence[subprocess.CompletedProcess[str] | BaseException]
    ) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[RunnerCall] = []

    def __call__(  # noqa: PLR0913 — mirrors the production runner protocol exactly
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
        self.calls.append(
            RunnerCall(tuple(args), capture_output, check, encoding, errors, shell, text, timeout)
        )
        outcome = next(self._outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _completed(
    stdout: str = "", *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["nvidia-smi"], returncode, stdout, stderr)


def _found(_executable: str) -> str:
    return "/opt/nvidia/bin/nvidia-smi"


def _reader(
    *outcomes: subprocess.CompletedProcess[str] | BaseException,
    resolver: Callable[[str], str | None] = _found,
    timeout_seconds: float = 2.5,
    max_output_bytes: int = 1_000_000,
) -> tuple[NvidiaSmiReader, ScriptedRunner]:
    runner = ScriptedRunner(outcomes)
    return (
        NvidiaSmiReader(
            runner=runner,
            resolver=resolver,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        ),
        runner,
    )


def _fixture(root: Path, name: str) -> str:
    return (root / "nvidia" / name).read_text(encoding="utf-8")


def test_parse_nvidia_csv_maps_values_by_requested_name_in_any_order() -> None:
    rows = parse_nvidia_csv(
        '55.5, 2, "future, value", 80, ignored trailing value\n',
        ("power.draw", "index", "future.column", "utilization.gpu"),
    )

    assert rows == (
        {
            "power.draw": "55.5",
            "index": "2",
            "future.column": "future, value",
            "utilization.gpu": "80",
        },
    )


def test_parse_nvidia_csv_normalizes_markers_and_tolerates_missing_columns() -> None:
    rows = parse_nvidia_csv(
        "0, [N/A], [Not Supported], , value\n",
        ("index", "one", "two", "three", "four", "missing"),
    )

    assert rows[0]["index"] == "0"
    assert rows[0]["one"] is UNSUPPORTED
    assert rows[0]["two"] is UNSUPPORTED
    assert rows[0]["three"] is UNSUPPORTED
    assert rows[0]["four"] == "value"
    assert "missing" not in rows[0]


def test_parse_nvidia_csv_retains_valid_rows_around_malformed_lines() -> None:
    rows = parse_nvidia_csv('"unterminated\n\n1, usable\n', ("index", "value"))

    assert rows == ({"index": "1", "value": "usable"},)


def test_static_info_uses_versioned_driver_fixture(telemetry_fixtures: Path) -> None:
    reader, _runner = _reader(
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-static.csv")),
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-compute.csv")),
        _completed(_fixture(telemetry_fixtures, "580.173.02-banner.txt")),
    )

    assert reader.static_info() == (
        GpuProfile(
            index=0,
            name="NVIDIA GeForce RTX 5060 Ti",
            uuid="GPU-fixture-0000",
            vram_total_bytes=16_311 * 1024 * 1024,
            driver_version="580.173.02",
            cuda_version="13.0",
            compute_capability="12.0",
            vendor=GpuVendor.NVIDIA,
        ),
    )
    assert reader.unavailable_reasons() == {}


def test_single_gpu_sample_normalizes_units_and_honestly_marks_na(
    telemetry_fixtures: Path,
) -> None:
    reader, _runner = _reader(
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-sample.csv")),
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-throttle.csv")),
    )

    sample = reader.sample()[0]

    assert sample.index == 0
    assert sample.uuid == "GPU-fixture-0000"
    assert sample.utilization_percent == 2.0
    assert sample.memory_utilization_percent == 10.0
    assert sample.vram_used_bytes == 655 * 1024 * 1024
    assert sample.vram_total_bytes == 16_311 * 1024 * 1024
    assert sample.temperature_c == 29.0
    assert sample.memory_temperature_c is UNSUPPORTED
    assert sample.power_watts == 9.99
    assert isinstance(sample.power_watts, float)
    assert sample.power_limit_watts == 180.0
    assert sample.fan_percent == 0.0
    assert sample.core_clock_mhz == 570.0
    assert sample.memory_clock_mhz == 405.0
    assert sample.throttle_reasons == ()
    assert sample.throttle_reasons_available is True
    assert reader.unavailable_reasons() == {"gpu.0.memory_temperature_c": "sensor_unsupported"}


def test_two_gpu_output_is_sorted_and_keeps_reasons_per_device(
    telemetry_fixtures: Path,
) -> None:
    sample_lines = _fixture(telemetry_fixtures, "580.173.02-two-gpu-sample.csv").splitlines()
    reason_lines = _fixture(telemetry_fixtures, "580.173.02-two-gpu-throttle.csv").splitlines()
    reader, _runner = _reader(
        _completed("\n".join(reversed(sample_lines))),
        _completed("\n".join(reversed(reason_lines))),
    )

    samples = reader.sample()

    assert [sample.index for sample in samples] == [0, 1]
    assert samples[1].vram_total_bytes == 24_576 * 1024 * 1024
    assert samples[1].power_watts == 210.5
    assert samples[1].throttle_reasons == ("sw_power_cap", "hw_thermal_slowdown")
    assert samples[1].throttle_reasons_available is True


def test_missing_power_sensor_is_unsupported_with_reason_never_zero(
    telemetry_fixtures: Path,
) -> None:
    output = _fixture(telemetry_fixtures, "580.173.02-single-sample.csv").replace(
        "9.99, 180.00", "[N/A], 180.00"
    )
    reader, _runner = _reader(
        _completed(output),
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-throttle.csv")),
    )

    sample = reader.sample()[0]

    assert sample.power_watts is UNSUPPORTED
    assert sample.power_limit_watts == 180.0
    assert reader.unavailable_reasons()["gpu.0.power_watts"] == "sensor_unsupported"


def test_percentages_outside_zero_to_one_hundred_degrade_independently() -> None:
    reader, _runner = _reader(
        _completed("0, GPU-fixture, 101, -1, 1, 2, 30, 31, 12, 20, 100.1, 300, 400\n"),
        _completed("0, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A\n"),
    )

    sample = reader.sample()[0]

    assert sample.utilization_percent is UNSUPPORTED
    assert sample.memory_utilization_percent is UNSUPPORTED
    assert sample.fan_percent is UNSUPPORTED
    assert sample.power_watts == 12.0
    assert sample.throttle_reasons_available is False
    assert reader.unavailable_reasons()["gpu.0.utilization_percent"] == "out_of_range"


def test_truncated_valid_row_retains_parseable_fields() -> None:
    reader, _runner = _reader(
        _completed("0, GPU-fixture, 25\n"),
        _completed("0, Not Active\n"),
    )

    sample = reader.sample()[0]

    assert sample.uuid == "GPU-fixture"
    assert sample.utilization_percent == 25.0
    assert sample.memory_utilization_percent is UNSUPPORTED
    assert sample.vram_used_bytes is UNSUPPORTED
    assert sample.throttle_reasons == ()
    assert sample.throttle_reasons_available is True
    assert reader.unavailable_reasons()["gpu.0.vram_used_bytes"] == "field_missing"


def test_malformed_csv_and_index_do_not_discard_a_valid_short_row() -> None:
    reader, _runner = _reader(
        _completed('"unterminated\nbad, GPU-bad, 1\n0, GPU-ok, 25\n'),
        _completed("0, Not Active\n"),
    )

    samples = reader.sample()

    assert len(samples) == 1
    assert samples[0].uuid == "GPU-ok"
    assert samples[0].utilization_percent == 25.0
    reasons = reader.unavailable_reasons()
    assert reasons["gpu"] == "malformed_csv"
    assert reasons["gpu.sample.row.0.index"] == "malformed_value"


def test_duplicate_and_negative_gpu_indices_are_rejected() -> None:
    reader, _runner = _reader(
        _completed("-1, GPU-negative, 1\n0, GPU-first, 2\n0, GPU-duplicate, 3\n"),
        _completed("0, Not Active\n"),
    )

    samples = reader.sample()

    assert len(samples) == 1
    assert samples[0].uuid == "GPU-first"
    assert reader.unavailable_reasons()["gpu.sample.row.2.index"] == "duplicate_gpu_index"


def test_zero_gpu_output_is_empty_and_available_is_false() -> None:
    reader, _runner = _reader(_completed(""), _completed(""))

    assert reader.available() is False
    assert reader.unavailable_reasons() == {"gpu": "no_nvidia_gpus"}
    assert reader.sample() == ()
    assert reader.unavailable_reasons() == {"gpu": "no_nvidia_gpus"}


def test_available_accepts_a_valid_index_and_rejects_a_malformed_probe() -> None:
    reader, _runner = _reader(_completed("0\n"), _completed("not-an-index\n"))

    assert reader.available() is True
    assert reader.unavailable_reasons() == {}
    assert reader.available() is False
    assert reader.unavailable_reasons()["gpu"] == "malformed_csv"


def test_missing_executable_never_calls_runner() -> None:
    reader, runner = _reader(resolver=lambda _executable: None)

    assert reader.available() is False
    assert reader.unavailable_reasons() == {"gpu": "nvidia_smi_not_found"}
    assert reader.sample() == ()
    assert reader.static_info() == ()
    assert runner.calls == []


def test_nonzero_exit_affects_one_sample_and_next_call_retries(
    telemetry_fixtures: Path,
) -> None:
    reader, runner = _reader(
        _completed(returncode=9, stderr="driver unavailable"),
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-sample.csv")),
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-throttle.csv")),
    )

    assert reader.sample() == ()
    assert reader.unavailable_reasons() == {"gpu": "nvidia_smi_failed"}
    assert len(reader.sample()) == 1
    assert len(runner.calls) == 3


def test_timeout_affects_one_sample_and_next_call_retries(telemetry_fixtures: Path) -> None:
    timeout = subprocess.TimeoutExpired(["nvidia-smi"], 2.5)
    reader, _runner = _reader(
        timeout,
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-sample.csv")),
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-throttle.csv")),
    )

    assert reader.sample() == ()
    assert reader.unavailable_reasons() == {"gpu": "nvidia_smi_timeout"}
    assert len(reader.sample()) == 1


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (FileNotFoundError("removed after resolution"), "nvidia_smi_not_found"),
        (PermissionError("denied"), "permission_denied"),
        (OSError("exec failed"), "nvidia_smi_unavailable"),
        (subprocess.SubprocessError("broken"), "nvidia_smi_unavailable"),
    ],
)
def test_process_start_failures_are_recorded(error: BaseException, reason: str) -> None:
    reader, _runner = _reader(error)

    assert reader.sample() == ()
    assert reader.unavailable_reasons() == {"gpu": reason}


def test_resolver_os_failure_is_recorded() -> None:
    def fail_resolver(_executable: str) -> str:
        raise OSError("path lookup failed")

    reader, _runner = _reader(resolver=fail_resolver)

    assert reader.available() is False
    assert reader.unavailable_reasons() == {"gpu": "nvidia_smi_not_found"}


def test_resolver_permission_failure_and_empty_result_are_recorded() -> None:
    def denied_resolver(_executable: str) -> str:
        raise PermissionError("path lookup denied")

    denied, _runner = _reader(resolver=denied_resolver)
    empty, _runner = _reader(resolver=lambda _executable: "")

    assert denied.sample() == ()
    assert denied.unavailable_reasons() == {"gpu": "permission_denied"}
    assert empty.sample() == ()
    assert empty.unavailable_reasons() == {"gpu": "nvidia_smi_not_found"}


def test_output_limit_counts_utf8_bytes_before_parsing() -> None:
    reader, _runner = _reader(_completed("ééé\n"), max_output_bytes=5)

    assert reader.sample() == ()
    assert reader.unavailable_reasons() == {"gpu": "nvidia_smi_output_too_large"}


def test_every_command_is_explicit_shell_free_utf8_and_timed(
    telemetry_fixtures: Path,
) -> None:
    reader, runner = _reader(
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-sample.csv")),
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-throttle.csv")),
        timeout_seconds=3.25,
    )

    reader.sample()

    assert len(runner.calls) == 2
    for call in runner.calls:
        assert call.args[0] == "/opt/nvidia/bin/nvidia-smi"
        assert call.args[1].startswith("--query-gpu=")
        assert call.args[2] == "--format=csv,noheader,nounits"
        assert call.capture_output is True
        assert call.check is False
        assert call.encoding == "utf-8"
        assert call.errors == "replace"
        assert call.shell is False
        assert call.text is True
        assert call.timeout == 3.25


def test_failed_throttle_query_preserves_core_metrics(telemetry_fixtures: Path) -> None:
    reader, _runner = _reader(
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-sample.csv")),
        _completed(returncode=2),
    )

    sample = reader.sample()[0]

    assert sample.utilization_percent == 2.0
    assert sample.throttle_reasons == ()
    assert sample.throttle_reasons_available is False
    assert reader.unavailable_reasons()["gpu.throttle_reasons"] == "nvidia_smi_failed"


def test_unrecognized_throttle_text_means_availability_is_unknown(
    telemetry_fixtures: Path,
) -> None:
    reader, _runner = _reader(
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-sample.csv")),
        _completed("0, Unknown, Unknown, Unknown, Unknown, Unknown, Unknown, Unknown, Unknown\n"),
    )

    sample = reader.sample()[0]

    assert sample.throttle_reasons == ()
    assert sample.throttle_reasons_available is False
    assert reader.unavailable_reasons()["gpu.0.throttle_reasons"] == "sensor_unsupported"


def test_optional_static_commands_can_fail_without_erasing_profile(
    telemetry_fixtures: Path,
) -> None:
    reader, _runner = _reader(
        _completed(_fixture(telemetry_fixtures, "580.173.02-single-static.csv")),
        _completed(returncode=2),
        subprocess.TimeoutExpired(["nvidia-smi"], 2.5),
    )

    profile = reader.static_info()[0]

    assert profile.name == "NVIDIA GeForce RTX 5060 Ti"
    assert profile.vram_total_bytes == 16_311 * 1024 * 1024
    assert profile.compute_capability is None
    assert profile.cuda_version is None
    assert reader.unavailable_reasons()["gpu.compute_capability"] == "nvidia_smi_failed"
    assert reader.unavailable_reasons()["gpu.cuda_version"] == "nvidia_smi_timeout"


def test_missing_cuda_banner_field_and_static_sensor_markers_are_recorded() -> None:
    reader, _runner = _reader(
        _completed("0, [Not Supported], [N/A], bad-vram, [N/A]\n"),
        _completed("0, [N/A]\n"),
        _completed("NVIDIA-SMI banner without toolkit version\n"),
    )

    profile = reader.static_info()[0]

    assert profile.name is None
    assert profile.uuid is None
    assert profile.vram_total_bytes is UNSUPPORTED
    assert profile.driver_version is None
    assert profile.compute_capability is None
    assert profile.cuda_version is None
    reasons = reader.unavailable_reasons()
    assert reasons["gpu.0.name"] == "sensor_unsupported"
    assert reasons["gpu.0.vram_total_bytes"] == "malformed_value"
    assert reasons["gpu.cuda_version"] == "field_missing"


def test_missing_static_columns_are_none_or_unsupported() -> None:
    reader, _runner = _reader(
        _completed("0, Fixture GPU\n"),
        _completed("0\n"),
        _completed("CUDA Version: 12.8\n"),
    )

    profile = reader.static_info()[0]

    assert profile.name == "Fixture GPU"
    assert profile.uuid is None
    assert profile.vram_total_bytes is UNSUPPORTED
    assert profile.driver_version is None
    assert profile.compute_capability is None
    assert reader.unavailable_reasons()["gpu.0.uuid"] == "field_missing"


def test_extreme_mib_value_degrades_instead_of_overflowing() -> None:
    reader, _runner = _reader(
        _completed("0, Fixture GPU, GPU-fixture, 1e308, 580.173.02\n"),
        _completed("0, 12.0\n"),
        _completed("CUDA Version: 13.0\n"),
    )

    profile = reader.static_info()[0]

    assert profile.vram_total_bytes is UNSUPPORTED
    assert reader.unavailable_reasons()["gpu.0.vram_total_bytes"] == "out_of_range"


def test_unavailable_reason_result_is_a_defensive_copy() -> None:
    reader, _runner = _reader(_completed(""))
    reader.sample()
    reasons = reader.unavailable_reasons()

    assert isinstance(reasons, dict)
    reasons["gpu"] = "tampered"
    assert reader.unavailable_reasons()["gpu"] == "no_nvidia_gpus"


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"executable": "  "}, "executable"),
        ({"executable": None}, "executable"),
        ({"timeout_seconds": 0.0}, "timeout_seconds"),
        ({"timeout_seconds": True}, "timeout_seconds"),
        ({"timeout_seconds": "5"}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"max_output_bytes": 0}, "max_output_bytes"),
        ({"max_output_bytes": False}, "max_output_bytes"),
        ({"max_output_bytes": 1.5}, "max_output_bytes"),
    ],
)
def test_constructor_rejects_unbounded_settings(kwargs: dict[str, object], field: str) -> None:
    with pytest.raises(ValidationError) as captured:
        NvidiaSmiReader(**kwargs)  # type: ignore[arg-type]  # deliberate runtime validation matrix

    assert captured.value.details["field"] == field


def test_process_control_exception_is_not_swallowed() -> None:
    reader, _runner = _reader(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        reader.sample()


def test_public_gpu_types_and_protocol_annotations_resolve() -> None:
    reader: GpuReader = NvidiaSmiReader(resolver=lambda _executable: None)

    assert reader.available() is False
    assert get_type_hints(GpuReader.sample)["return"] == Sequence[GpuSample]
    assert ParsedCell.__value__ is not None
    assert SubprocessRunner.__name__ == "SubprocessRunner"

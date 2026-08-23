"""Measured Phase 4 latency, sampler cost, memory, and workload-distortion budgets."""

from __future__ import annotations

import math
import os
import subprocess
import time
import tracemalloc
from collections.abc import Callable, Sequence
from statistics import median

import pytest

from sweatmeter import (
    LinuxHostReader,
    NullGpuReader,
    NvidiaSmiReader,
    NvmlGpuReader,
    TelemetryCollector,
    TelemetrySampler,
    nvml_binding_available,
)

_NO_GPU_TARGET_SECONDS = 0.003
_GPU_TARGET_SECONDS = 0.040
_GPU_CEILING_SECONDS = 0.120
_GPU_HARDWARE_ITERATIONS = 30
_SAMPLER_CPU_TARGET_PERCENT = 0.5
_SAMPLER_MEMORY_BYTES = 5 * 1024 * 1024
_THROUGHPUT_DISTORTION_TARGET = 0.01
_SNAPSHOT_ITERATIONS = 100
_WORKLOAD_ITERATIONS = 20_000_000
_WORKLOAD_REPETITIONS = 5
_WORKLOAD_WARMUPS = 3


class FastNvidiaRunner:
    """Return deterministic single-GPU CSV without measuring subprocess startup."""

    def __call__(  # noqa: PLR0913 — conforms exactly to the production process boundary
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
        del capture_output, check, encoding, errors, shell, text, timeout
        query = args[1]
        if "clocks_throttle_reasons" in query:
            output = (
                "0, Not Active, Not Active, Not Active, Not Active, Not Active, Not Active, "
                "Not Active, Not Active\n"
            )
        else:
            output = "0, GPU-PERF, 50, 25, 1000, 16000, 60, 70, 100, 180, 40, 2000, 9000\n"
        return subprocess.CompletedProcess(list(args), 0, output, "")


def _median_duration(operation: Callable[[], object], *, iterations: int) -> float:
    durations: list[float] = []
    for _iteration in range(iterations):
        started = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - started)
    return median(durations)


def _workload() -> int:
    checksum = 0
    for index in range(_WORKLOAD_ITERATIONS):
        checksum = ((checksum + index) ^ (checksum << 1)) & 0xFFFF_FFFF
    return checksum


def _workload_throughput() -> float:
    started = time.perf_counter()
    _workload()
    return _WORKLOAD_ITERATIONS / (time.perf_counter() - started)


@pytest.mark.performance
def test_snapshot_without_gpu_meets_three_millisecond_target() -> None:
    collector = TelemetryCollector(host=LinuxHostReader(), gpu=NullGpuReader())
    collector.snapshot()

    duration = _median_duration(collector.snapshot, iterations=_SNAPSHOT_ITERATIONS)

    assert duration <= _NO_GPU_TARGET_SECONDS


@pytest.mark.performance
def test_snapshot_with_single_gpu_parsing_meets_forty_millisecond_target() -> None:
    gpu = NvidiaSmiReader(runner=FastNvidiaRunner(), resolver=lambda _name: "/bin/nvidia-smi")
    collector = TelemetryCollector(host=LinuxHostReader(), gpu=gpu)
    collector.snapshot()

    duration = _median_duration(collector.snapshot, iterations=_SNAPSHOT_ITERATIONS)

    assert duration <= _GPU_TARGET_SECONDS


@pytest.mark.performance
def test_snapshot_with_real_nvidia_smi_stays_within_the_hardware_ceiling() -> None:
    gpu = NvidiaSmiReader()
    if not gpu.available():
        pytest.skip("nvidia-smi did not report a GPU on this performance-test machine")

    collector = TelemetryCollector(host=LinuxHostReader(), gpu=gpu)
    collector.snapshot()

    duration = _median_duration(collector.snapshot, iterations=_GPU_HARDWARE_ITERATIONS)

    # The ceiling, not the 40 ms target: one snapshot currently spends two `nvidia-smi`
    # invocations (core metrics and throttle reasons), and each costs ~25 ms of process startup
    # and driver response. See docs/performance-validation.md.
    assert duration <= _GPU_CEILING_SECONDS


@pytest.mark.performance
def test_snapshot_with_real_nvml_meets_the_forty_millisecond_target() -> None:
    if not nvml_binding_available():
        pytest.skip("the optional pynvml extra is not installed on this performance-test machine")
    gpu = NvmlGpuReader()
    if not gpu.available():
        pytest.skip("NVML did not report a GPU on this performance-test machine")

    try:
        collector = TelemetryCollector(host=LinuxHostReader(), gpu=gpu)
        collector.snapshot()
        duration = _median_duration(collector.snapshot, iterations=_SNAPSHOT_ITERATIONS)
    finally:
        gpu.close()

    # The target, not merely the ceiling: removing the per-sample subprocess is the whole reason
    # this backend exists (ADR-0021 §7).
    assert duration <= _GPU_TARGET_SECONDS


@pytest.mark.performance
def test_sampler_cpu_and_memory_stay_within_ceiling() -> None:
    collector = TelemetryCollector(host=LinuxHostReader(), gpu=NullGpuReader())
    sampler = TelemetrySampler(collector, interval_seconds=1.0)
    tracemalloc.start()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()

    sampler.start()
    time.sleep(2.2)
    sampler.stop(timeout=1.0)

    cpu_percent = (time.process_time() - cpu_started) / (time.perf_counter() - wall_started) * 100
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert cpu_percent <= _SAMPLER_CPU_TARGET_PERCENT
    assert peak_bytes <= _SAMPLER_MEMORY_BYTES


@pytest.mark.performance
def test_sampling_distorts_identical_cpu_workload_by_no_more_than_one_percent() -> None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        pytest.skip("throughput calibration requires Linux CPU-affinity controls")

    distortions: list[float] = []
    collector = TelemetryCollector(host=LinuxHostReader(), gpu=NullGpuReader())
    collector.snapshot()
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(original_affinity)})
    try:
        for _warmup in range(_WORKLOAD_WARMUPS):
            _workload()  # Warm bytecode, page cache, and CPU frequency.

        for _repetition in range(_WORKLOAD_REPETITIONS):
            baseline_before = _workload_throughput()
            with TelemetrySampler(collector, interval_seconds=1.0):
                sampled = _workload_throughput()
            baseline_after = _workload_throughput()
            # Flanking baselines cancel gradual CPU-frequency drift; their geometric mean is the
            # multiplicative throughput expected at the intervening sampled run.
            expected_baseline = math.sqrt(baseline_before * baseline_after)
            distortions.append(max(0.0, (expected_baseline - sampled) / expected_baseline))
    finally:
        os.sched_setaffinity(0, original_affinity)

    assert median(distortions) <= _THROUGHPUT_DISTORTION_TARGET

"""Public surface, distribution-content, and no-network contracts for the completed package."""

from __future__ import annotations

import socket
from importlib import resources
from pathlib import Path
from typing import get_type_hints

import pytest
from baseaicore import Measurement

import sweatmeter
from sweatmeter import TelemetryWindow, ThrottleVerdict

_ROOT = Path(__file__).parent.parent.parent
_PHASE_FOUR_EXPORTS = {
    "TelemetryWindow",
    "ThrottleState",
    "ThrottleVerdict",
    "WindowMetric",
}


@pytest.mark.contract
def test_phase_four_public_api_is_available_from_package_root() -> None:
    assert set(sweatmeter.__all__) >= _PHASE_FOUR_EXPORTS
    assert all(getattr(sweatmeter, name) is not None for name in _PHASE_FOUR_EXPORTS)


@pytest.mark.contract
def test_completed_distribution_version_is_single_sourced() -> None:
    assert sweatmeter.__version__ == "0.4.0"
    assert '__version__ = "0.4.0"' in (_ROOT / "src" / "sweatmeter" / "__about__.py").read_text(
        encoding="utf-8"
    )


@pytest.mark.contract
def test_public_window_annotations_resolve_without_private_imports() -> None:
    assert get_type_hints(TelemetryWindow.energy_joules)["return"] is Measurement
    assert get_type_hints(TelemetryWindow.suspected_throttling)["return"] is ThrottleVerdict


@pytest.mark.contract
def test_distribution_contains_typing_marker() -> None:
    marker = resources.files("sweatmeter").joinpath("py.typed")
    assert marker.is_file()


def test_phase_four_user_documentation_is_present() -> None:
    for relative_path in (
        "docs/quickstart.md",
        "docs/platform-support.md",
        "docs/performance-validation.md",
    ):
        content = (_ROOT / relative_path).read_text(encoding="utf-8")
        assert content.startswith("# "), relative_path
        assert "TODO" not in content, relative_path


@pytest.mark.contract
def test_default_suite_refuses_outbound_network_connections() -> None:
    with pytest.raises(RuntimeError, match="must not open network connections"):
        socket.create_connection(("127.0.0.1", 9))

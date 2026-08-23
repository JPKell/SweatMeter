"""Shared fixtures for portable SweatMeter tests."""

from pathlib import Path

import pytest


@pytest.fixture
def telemetry_fixtures() -> Path:
    """Return the root of captured, platform-independent kernel-data fixtures."""
    return Path(__file__).parent / "fixtures" / "telemetry"

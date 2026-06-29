# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SEARCH_R1_SCRIPTS = Path(__file__).resolve().parents[4] / "contrib" / "recipes" / "search_r1" / "scripts"
sys.path.insert(0, str(SEARCH_R1_SCRIPTS))

import gpu_keepalive as gk  # noqa: E402


def test_parse_target_util_fraction() -> None:
    assert gk.parse_target_util("0.05") == 0.05
    assert gk.parse_target_util("0.5") == 0.5


def test_parse_target_util_percent() -> None:
    assert gk.parse_target_util("5") == 0.05
    assert gk.parse_target_util("35") == 0.35
    assert gk.parse_target_util("5%") == 0.05


def test_duty_cycle_sleep_seconds() -> None:
    assert gk.duty_cycle_sleep_seconds(0.05, 0.05) == pytest.approx(0.95)
    assert gk.duty_cycle_sleep_seconds(0.1, 0.1) == pytest.approx(0.9)
    assert gk.duty_cycle_sleep_seconds(0.05, 0.0) == 0.0
    assert gk.duty_cycle_sleep_seconds(0.05, 1.0) == 0.0


def test_parse_skip_devices() -> None:
    assert gk.parse_skip_devices("") == set()
    assert gk.parse_skip_devices("0, 2 3") == {0, 2, 3}

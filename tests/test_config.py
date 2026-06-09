"""Smoke tests for config loading (DoD item 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from delivery_sim.config.loader import load_scenario
from delivery_sim.config.schema import ScenarioConfig

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def test_load_example_scenario_returns_scenario_config() -> None:
    """load_scenario('scenarios/example.yaml') returns a valid ScenarioConfig."""
    config = load_scenario(SCENARIOS_DIR / "example.yaml")
    assert isinstance(config, ScenarioConfig)


def test_example_scenario_fields() -> None:
    config = load_scenario(SCENARIOS_DIR / "example.yaml")
    assert config.name == "example_scenario"
    assert config.seed == 42
    assert config.dt == 1.0
    assert config.max_steps == 1000


def test_example_scenario_world() -> None:
    config = load_scenario(SCENARIOS_DIR / "example.yaml")
    assert config.world.width == 1000.0
    assert config.world.height == 1000.0


def test_example_scenario_stores() -> None:
    config = load_scenario(SCENARIOS_DIR / "example.yaml")
    assert len(config.stores) == 2
    names = {s.name for s in config.stores}
    assert names == {"warehouse_a", "warehouse_b"}


def test_example_scenario_couriers() -> None:
    config = load_scenario(SCENARIOS_DIR / "example.yaml")
    assert len(config.couriers) == 1
    assert config.couriers[0].courier_type == "BikeCourier"
    assert config.couriers[0].count == 3


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario(SCENARIOS_DIR / "does_not_exist.yaml")


def test_invalid_yaml_raises_validation_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: test\ndt: -1.0\n")  # dt must be > 0
    with pytest.raises(Exception):  # pydantic.ValidationError
        load_scenario(bad)

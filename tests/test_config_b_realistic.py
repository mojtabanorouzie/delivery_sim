"""B1 — Config schema amendment tests for increment B-realistic.

Verifies that:
- All new fields are additive with defaults (existing YAMLs load unchanged).
- New fields validate correctly (reject bad values).
- The balanced.yaml scenario loads with default B-realistic values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from delivery_sim.config.schema import (
    DemandConfig,
    ProfileBreakpoint,
    ScenarioConfig,
    StoreConfig,
)

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


# ---------------------------------------------------------------------------
# Existing scenario backward compatibility
# ---------------------------------------------------------------------------

class TestExistingScenariosLoadUnchanged:
    def test_balanced_yaml_loads(self) -> None:
        from delivery_sim.config.loader import load_scenario
        cfg = load_scenario(SCENARIOS_DIR / "balanced.yaml")
        assert isinstance(cfg, ScenarioConfig)

    def test_balanced_has_default_return_rate(self) -> None:
        from delivery_sim.config.loader import load_scenario
        cfg = load_scenario(SCENARIOS_DIR / "balanced.yaml")
        assert cfg.return_rate == 0.0

    def test_balanced_stores_have_default_prep_time(self) -> None:
        from delivery_sim.config.loader import load_scenario
        cfg = load_scenario(SCENARIOS_DIR / "balanced.yaml")
        for store in cfg.stores:
            assert store.prep_time == 30.0

    def test_balanced_demand_has_default_intensity(self) -> None:
        from delivery_sim.config.loader import load_scenario
        cfg = load_scenario(SCENARIOS_DIR / "balanced.yaml")
        assert cfg.demand.intensity == 1.0

    def test_balanced_demand_has_empty_profile(self) -> None:
        from delivery_sim.config.loader import load_scenario
        cfg = load_scenario(SCENARIOS_DIR / "balanced.yaml")
        assert cfg.demand.profile == []

    def test_example_yaml_loads(self) -> None:
        from delivery_sim.config.loader import load_scenario
        cfg = load_scenario(SCENARIOS_DIR / "example.yaml")
        assert isinstance(cfg, ScenarioConfig)
        assert cfg.return_rate == 0.0


# ---------------------------------------------------------------------------
# StoreConfig: prep_time
# ---------------------------------------------------------------------------

class TestStoreConfigPrepTime:
    def test_default_is_30(self) -> None:
        s = StoreConfig(name="s", x=0.0, y=0.0)
        assert s.prep_time == 30.0

    def test_explicit_value_stored(self) -> None:
        s = StoreConfig(name="s", x=0.0, y=0.0, prep_time=60.0)
        assert s.prep_time == 60.0

    def test_zero_prep_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StoreConfig(name="s", x=0.0, y=0.0, prep_time=0.0)

    def test_negative_prep_time_rejected(self) -> None:
        with pytest.raises(ValidationError):
            StoreConfig(name="s", x=0.0, y=0.0, prep_time=-5.0)


# ---------------------------------------------------------------------------
# DemandConfig: intensity
# ---------------------------------------------------------------------------

class TestDemandConfigIntensity:
    def test_default_intensity_is_1(self) -> None:
        d = DemandConfig(generator_type="PoissonDemandGenerator")
        assert d.intensity == 1.0

    def test_intensity_stored(self) -> None:
        d = DemandConfig(generator_type="PoissonDemandGenerator", intensity=2.0)
        assert d.intensity == 2.0

    def test_zero_intensity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DemandConfig(generator_type="PoissonDemandGenerator", intensity=0.0)

    def test_negative_intensity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DemandConfig(generator_type="PoissonDemandGenerator", intensity=-1.0)


# ---------------------------------------------------------------------------
# DemandConfig: profile breakpoints
# ---------------------------------------------------------------------------

class TestDemandConfigProfile:
    def test_empty_profile_by_default(self) -> None:
        d = DemandConfig(generator_type="DailyProfileDemandGenerator")
        assert d.profile == []

    def test_valid_profile_stored(self) -> None:
        d = DemandConfig(
            generator_type="DailyProfileDemandGenerator",
            profile=[
                {"time_fraction": 0.0, "rate_factor": 0.3},
                {"time_fraction": 0.5, "rate_factor": 1.0},
                {"time_fraction": 1.0, "rate_factor": 0.3},
            ],
        )
        assert len(d.profile) == 3
        assert d.profile[0].time_fraction == 0.0
        assert d.profile[1].rate_factor == 1.0

    def test_profile_breakpoint_time_fraction_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            ProfileBreakpoint(time_fraction=1.5, rate_factor=1.0)

    def test_profile_breakpoint_negative_time_fraction(self) -> None:
        with pytest.raises(ValidationError):
            ProfileBreakpoint(time_fraction=-0.1, rate_factor=1.0)

    def test_profile_breakpoint_negative_rate_factor(self) -> None:
        with pytest.raises(ValidationError):
            ProfileBreakpoint(time_fraction=0.5, rate_factor=-0.1)


# ---------------------------------------------------------------------------
# DemandConfig: burst parameters
# ---------------------------------------------------------------------------

class TestDemandConfigBurst:
    def test_burst_defaults(self) -> None:
        d = DemandConfig(generator_type="BurstDemandGenerator")
        assert d.burst_rate_factor == 5.0
        assert d.burst_duration_fraction == 0.1
        assert d.burst_interval_fraction == 0.3

    def test_burst_rate_factor_must_exceed_1(self) -> None:
        with pytest.raises(ValidationError):
            DemandConfig(
                generator_type="BurstDemandGenerator",
                burst_rate_factor=1.0,
            )

    def test_burst_duration_fraction_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DemandConfig(
                generator_type="BurstDemandGenerator",
                burst_duration_fraction=0.0,
            )

    def test_burst_interval_fraction_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            DemandConfig(
                generator_type="BurstDemandGenerator",
                burst_interval_fraction=0.0,
            )

    def test_burst_windows_overlap_rejected(self) -> None:
        # duration=0.7, interval=0.4 → sum=1.1 > 1.0
        with pytest.raises(ValidationError, match="burst_duration_fraction"):
            DemandConfig(
                generator_type="BurstDemandGenerator",
                burst_duration_fraction=0.7,
                burst_interval_fraction=0.4,
            )

    def test_burst_windows_sum_exactly_1_accepted(self) -> None:
        # sum == 1.0 is the boundary (duration + interval = 1.0 is valid)
        d = DemandConfig(
            generator_type="BurstDemandGenerator",
            burst_duration_fraction=0.5,
            burst_interval_fraction=0.5,
        )
        assert d.burst_duration_fraction + d.burst_interval_fraction == 1.0

    def test_burst_duration_must_be_lt_1(self) -> None:
        with pytest.raises(ValidationError):
            DemandConfig(
                generator_type="BurstDemandGenerator",
                burst_duration_fraction=1.0,
            )


# ---------------------------------------------------------------------------
# ScenarioConfig: return_rate
# ---------------------------------------------------------------------------

class TestScenarioConfigReturnRate:
    def _minimal(self, **kwargs: object) -> ScenarioConfig:
        return ScenarioConfig(name="test", **kwargs)  # type: ignore[arg-type]

    def test_default_return_rate_is_zero(self) -> None:
        cfg = self._minimal()
        assert cfg.return_rate == 0.0

    def test_valid_return_rate_stored(self) -> None:
        cfg = self._minimal(return_rate=0.01)
        assert cfg.return_rate == 0.01

    def test_return_rate_1_accepted(self) -> None:
        cfg = self._minimal(return_rate=1.0)
        assert cfg.return_rate == 1.0

    def test_negative_return_rate_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._minimal(return_rate=-0.01)

    def test_return_rate_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._minimal(return_rate=1.001)


# ---------------------------------------------------------------------------
# Round-trip: YAML with B-realistic fields
# ---------------------------------------------------------------------------

class TestBRealisticYamlRoundtrip:
    def test_inline_b_realistic_yaml(self, tmp_path: Path) -> None:
        yaml_text = """\
name: b_realistic_test
seed: 7
dt: 1.0
max_steps: 500
return_rate: 0.02
stores:
  - name: depot
    x: 500.0
    y: 500.0
    capacity: 5
    prep_time: 45.0
    coverage_radius: 600.0
couriers:
  - courier_type: BikeCourier
    count: 10
    speed: 1.5
demand:
  generator_type: PoissonDemandGenerator
  rate: 0.02
  intensity: 1.5
routing:
  model_type: euclidean
"""
        p = tmp_path / "b_realistic_test.yaml"
        p.write_text(yaml_text, encoding="utf-8")
        from delivery_sim.config.loader import load_scenario
        cfg = load_scenario(p)
        assert cfg.return_rate == 0.02
        assert cfg.stores[0].prep_time == 45.0
        assert cfg.demand.intensity == 1.5

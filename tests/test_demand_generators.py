"""B8 — Time-varying demand generators.

D-1: PoissonDemandGenerator backward compat — existing tests still pass,
     reset(rng) without horizon kwarg must not raise.
D-2: DailyProfileDemandGenerator.reset(horizon=0) raises ValueError (V-3).
D-3: DailyProfileDemandGenerator current_intensity matches breakpoints.
D-4: BurstDemandGenerator.reset(horizon=0) raises ValueError (V-3).
D-5: BurstDemandGenerator current_intensity alternates correctly.
D-6: Both generators produce more orders in high-intensity phases than low.
D-7: Simulator integration — DailyProfile and Burst generators run to
     completion without error.
D-8: Determinism — same (config, seed) with a time-varying generator.
"""

from __future__ import annotations

import numpy as np
import pytest

from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    ProfileBreakpoint,
    ScenarioConfig,
    StoreConfig,
)
from delivery_sim.engine.simulator import Simulator
from delivery_sim.entities.demand_generator import (
    BurstDemandGenerator,
    DailyProfileDemandGenerator,
    PoissonDemandGenerator,
)
from delivery_sim.metrics.collector import KPICollector

# ---------------------------------------------------------------------------
# D-1: PoissonDemandGenerator backward compat
# ---------------------------------------------------------------------------

class TestPoissonBackwardCompat:
    def test_reset_without_horizon_does_not_raise(self) -> None:
        gen = PoissonDemandGenerator(rate=0.03, dt=1.0)
        rng = np.random.default_rng(0)
        gen.reset(rng)  # no horizon kwarg — must not raise

    def test_reset_with_horizon_zero_does_not_raise(self) -> None:
        gen = PoissonDemandGenerator(rate=0.03, dt=1.0)
        rng = np.random.default_rng(0)
        gen.reset(rng, horizon=0.0)  # Poisson is stationary — no validation needed

    def test_current_intensity_constant(self) -> None:
        gen = PoissonDemandGenerator(rate=0.03, dt=1.0)
        for t in [0.0, 500.0, 2999.0]:
            assert gen.current_intensity(t) == 1.0

    def test_accepts_intensity_kwarg(self) -> None:
        gen = PoissonDemandGenerator(rate=0.03, dt=1.0, intensity=2.0)
        assert gen.intensity == 2.0

    def test_accepts_extra_kwargs(self) -> None:
        gen = PoissonDemandGenerator(
            rate=0.03, dt=1.0,
            profile=[],
            burst_rate_factor=5.0,
            burst_duration_fraction=0.1,
            burst_interval_fraction=0.3,
        )
        assert gen.rate == 0.03


# ---------------------------------------------------------------------------
# D-2: DailyProfileDemandGenerator.reset horizon=0 raises ValueError (V-3)
# ---------------------------------------------------------------------------

class TestDailyProfileV3:
    def _gen(self) -> DailyProfileDemandGenerator:
        return DailyProfileDemandGenerator(
            rate=0.03, dt=1.0,
            profile=[
                ProfileBreakpoint(time_fraction=0.0, rate_factor=0.5),
                ProfileBreakpoint(time_fraction=0.5, rate_factor=2.0),
                ProfileBreakpoint(time_fraction=1.0, rate_factor=0.5),
            ],
        )

    def test_horizon_zero_raises(self) -> None:
        gen = self._gen()
        with pytest.raises(ValueError, match="horizon > 0"):
            gen.reset(np.random.default_rng(0), horizon=0.0)

    def test_horizon_negative_raises(self) -> None:
        gen = self._gen()
        with pytest.raises(ValueError, match="horizon > 0"):
            gen.reset(np.random.default_rng(0), horizon=-1.0)

    def test_valid_horizon_does_not_raise(self) -> None:
        gen = self._gen()
        gen.reset(np.random.default_rng(0), horizon=3000.0)


# ---------------------------------------------------------------------------
# D-3: DailyProfileDemandGenerator current_intensity matches breakpoints
# ---------------------------------------------------------------------------

class TestDailyProfileIntensity:
    def _gen(self, profile_points: list[tuple[float, float]]) -> DailyProfileDemandGenerator:
        profile = [
            ProfileBreakpoint(time_fraction=tf, rate_factor=rf)
            for tf, rf in profile_points
        ]
        gen = DailyProfileDemandGenerator(rate=1.0, dt=1.0, profile=profile, intensity=1.0)
        gen.reset(np.random.default_rng(0), horizon=1000.0)
        return gen

    def test_at_first_breakpoint(self) -> None:
        gen = self._gen([(0.0, 0.5), (0.5, 2.0), (1.0, 0.5)])
        assert gen.current_intensity(0.0) == pytest.approx(0.5)

    def test_at_last_breakpoint(self) -> None:
        gen = self._gen([(0.0, 0.5), (0.5, 2.0), (1.0, 0.5)])
        assert gen.current_intensity(1000.0) == pytest.approx(0.5)

    def test_at_midpoint_breakpoint(self) -> None:
        gen = self._gen([(0.0, 0.5), (0.5, 2.0), (1.0, 0.5)])
        assert gen.current_intensity(500.0) == pytest.approx(2.0)

    def test_interpolation_between_breakpoints(self) -> None:
        gen = self._gen([(0.0, 0.0), (1.0, 4.0)])
        # At t=250 (tf=0.25) → factor = 0 + 0.25*(4-0) = 1.0
        assert gen.current_intensity(250.0) == pytest.approx(1.0)

    def test_below_first_breakpoint_clamps(self) -> None:
        gen = self._gen([(0.2, 3.0), (1.0, 1.0)])
        # t=0 → tf=0.0 < 0.2 → clamp to first factor
        assert gen.current_intensity(0.0) == pytest.approx(3.0)

    def test_empty_profile_returns_intensity(self) -> None:
        gen = DailyProfileDemandGenerator(rate=1.0, dt=1.0, profile=[], intensity=2.5)
        gen.reset(np.random.default_rng(0), horizon=1000.0)
        assert gen.current_intensity(500.0) == pytest.approx(2.5)

    def test_intensity_multiplier(self) -> None:
        gen = DailyProfileDemandGenerator(
            rate=1.0, dt=1.0, intensity=0.5,
            profile=[ProfileBreakpoint(time_fraction=0.0, rate_factor=2.0),
                     ProfileBreakpoint(time_fraction=1.0, rate_factor=2.0)],
        )
        gen.reset(np.random.default_rng(0), horizon=1000.0)
        # intensity=0.5, factor=2.0 → 1.0
        assert gen.current_intensity(500.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# D-4: BurstDemandGenerator.reset horizon=0 raises ValueError (V-3)
# ---------------------------------------------------------------------------

class TestBurstV3:
    def _gen(self) -> BurstDemandGenerator:
        return BurstDemandGenerator(
            rate=0.03, dt=1.0,
            burst_rate_factor=5.0,
            burst_duration_fraction=0.1,
            burst_interval_fraction=0.3,
        )

    def test_horizon_zero_raises(self) -> None:
        gen = self._gen()
        with pytest.raises(ValueError, match="horizon > 0"):
            gen.reset(np.random.default_rng(0), horizon=0.0)

    def test_horizon_negative_raises(self) -> None:
        gen = self._gen()
        with pytest.raises(ValueError, match="horizon > 0"):
            gen.reset(np.random.default_rng(0), horizon=-100.0)

    def test_valid_horizon_does_not_raise(self) -> None:
        gen = self._gen()
        gen.reset(np.random.default_rng(0), horizon=3000.0)


# ---------------------------------------------------------------------------
# D-5: BurstDemandGenerator current_intensity alternates correctly
# ---------------------------------------------------------------------------

class TestBurstIntensity:
    def _gen(self, burst_factor: float = 5.0,
             dur: float = 0.1, interval: float = 0.3,
             horizon: float = 1000.0) -> BurstDemandGenerator:
        gen = BurstDemandGenerator(
            rate=1.0, dt=1.0,
            burst_rate_factor=burst_factor,
            burst_duration_fraction=dur,
            burst_interval_fraction=interval,
            intensity=1.0,
        )
        gen.reset(np.random.default_rng(0), horizon=horizon)
        return gen

    def test_at_cycle_start_is_burst(self) -> None:
        gen = self._gen(horizon=1000.0)
        # burst_len = 0.1 * 1000 = 100, cycle_len = 400
        # t=0 → pos=0 < 100 → burst
        assert gen.current_intensity(0.0) == pytest.approx(5.0)

    def test_just_before_burst_end_is_burst(self) -> None:
        gen = self._gen(horizon=1000.0)
        assert gen.current_intensity(99.0) == pytest.approx(5.0)

    def test_at_burst_end_is_quiet(self) -> None:
        gen = self._gen(horizon=1000.0)
        # pos=100 → quiet (>=100)
        assert gen.current_intensity(100.0) == pytest.approx(1.0)

    def test_end_of_cycle_is_quiet(self) -> None:
        gen = self._gen(horizon=1000.0)
        # cycle_len=400, t=399 → pos=399 ≥ 100 → quiet
        assert gen.current_intensity(399.0) == pytest.approx(1.0)

    def test_second_cycle_burst(self) -> None:
        gen = self._gen(horizon=1000.0)
        # t=400 → pos=0 → burst
        assert gen.current_intensity(400.0) == pytest.approx(5.0)

    def test_intensity_multiplier(self) -> None:
        gen = BurstDemandGenerator(
            rate=1.0, dt=1.0,
            burst_rate_factor=5.0,
            burst_duration_fraction=0.1,
            burst_interval_fraction=0.3,
            intensity=2.0,
        )
        gen.reset(np.random.default_rng(0), horizon=1000.0)
        assert gen.current_intensity(0.0) == pytest.approx(10.0)   # burst: 2.0*5
        assert gen.current_intensity(200.0) == pytest.approx(2.0)  # quiet: 2.0*1


# ---------------------------------------------------------------------------
# D-6: Time-varying generators produce more orders in high-intensity windows
# ---------------------------------------------------------------------------

class TestHighLowIntensityOrdering:
    def test_daily_profile_peak_rate_higher_than_trough(self) -> None:
        rng = np.random.default_rng(999)
        horizon = 10000.0

        # Profile: low at t_fraction=0.0, high at t_fraction=0.5
        profile = [
            ProfileBreakpoint(time_fraction=0.0, rate_factor=0.1),
            ProfileBreakpoint(time_fraction=0.5, rate_factor=5.0),
            ProfileBreakpoint(time_fraction=1.0, rate_factor=0.1),
        ]
        gen = DailyProfileDemandGenerator(
            rate=0.01, dt=1.0, profile=profile, intensity=1.0,
        )
        gen.reset(rng, horizon=horizon)

        # Count next_event inter-arrivals near trough (t≈0) vs peak (t≈5000)
        trough_delays = [gen.next_event(0.0, rng)[0] for _ in range(50)]  # type: ignore[index]
        peak_delays   = [gen.next_event(5000.0, rng)[0] for _ in range(50)]  # type: ignore[index]

        mean_trough_delay = sum(d - 0.0 for d in trough_delays) / 50
        mean_peak_delay   = sum(d - 5000.0 for d in peak_delays) / 50

        # Peak has higher rate → shorter delays
        assert mean_peak_delay < mean_trough_delay

    def test_burst_has_shorter_delays_in_burst_window(self) -> None:
        rng = np.random.default_rng(7)
        horizon = 10000.0
        gen = BurstDemandGenerator(
            rate=0.01, dt=1.0,
            burst_rate_factor=10.0,
            burst_duration_fraction=0.1,
            burst_interval_fraction=0.3,
            intensity=1.0,
        )
        gen.reset(rng, horizon=horizon)
        # t=0 → burst; t=2000 → quiet (cycle_len=4000, burst_len=1000)
        burst_delays = [gen.next_event(0.0, rng)[0] for _ in range(50)]  # type: ignore[index]
        quiet_delays = [gen.next_event(2000.0, rng)[0] for _ in range(50)]  # type: ignore[index]
        mean_burst = sum(d - 0.0 for d in burst_delays) / 50
        mean_quiet = sum(d - 2000.0 for d in quiet_delays) / 50
        assert mean_burst < mean_quiet


# ---------------------------------------------------------------------------
# D-7: Simulator integration — time-varying generators complete without error
# ---------------------------------------------------------------------------

def _base_cfg(**demand_overrides: object) -> ScenarioConfig:
    return ScenarioConfig(
        name="dg_test",
        seed=42,
        dt=1.0,
        max_steps=3000,
        stores=[StoreConfig(
            name="s", x=500.0, y=500.0,
            capacity=20, prep_time=30.0, coverage_radius=1000.0,
        )],
        couriers=[CourierConfig(courier_type="BikeCourier", count=10, speed=2.0)],
        demand=DemandConfig(**demand_overrides),  # type: ignore[arg-type]
    )


class TestSimulatorIntegrationDemand:
    def test_daily_profile_runs_to_completion(self) -> None:
        cfg = _base_cfg(
            generator_type="DailyProfileDemandGenerator",
            rate=0.03,
            profile=[
                ProfileBreakpoint(time_fraction=0.0, rate_factor=0.5),
                ProfileBreakpoint(time_fraction=0.5, rate_factor=3.0),
                ProfileBreakpoint(time_fraction=1.0, rate_factor=0.5),
            ],
        )
        sim = Simulator(cfg)
        col = KPICollector()
        sim.attach_collector(col)
        sim.run()
        assert col.summary()["total_orders"] > 0

    def test_burst_runs_to_completion(self) -> None:
        cfg = _base_cfg(
            generator_type="BurstDemandGenerator",
            rate=0.03,
            burst_rate_factor=5.0,
            burst_duration_fraction=0.1,
            burst_interval_fraction=0.3,
        )
        sim = Simulator(cfg)
        col = KPICollector()
        sim.attach_collector(col)
        sim.run()
        assert col.summary()["total_orders"] > 0

    def test_poisson_reset_accepts_horizon_kwarg(self) -> None:
        cfg = _base_cfg(generator_type="PoissonDemandGenerator", rate=0.03)
        sim = Simulator(cfg)
        col = KPICollector()
        sim.attach_collector(col)
        sim.run()  # uses horizon kwarg internally; must not raise
        assert col.summary()["total_orders"] > 0


# ---------------------------------------------------------------------------
# D-8: Determinism with time-varying generators
# ---------------------------------------------------------------------------

class TestDemandGeneratorDeterminism:
    def _run_kpis(self, cfg: ScenarioConfig) -> dict:
        sim = Simulator(cfg)
        col = KPICollector()
        sim.attach_collector(col)
        sim.run()
        return col.summary()

    def test_daily_profile_determinism(self) -> None:
        cfg = _base_cfg(
            generator_type="DailyProfileDemandGenerator",
            rate=0.03,
            profile=[
                ProfileBreakpoint(time_fraction=0.0, rate_factor=0.5),
                ProfileBreakpoint(time_fraction=1.0, rate_factor=2.0),
            ],
        )
        assert self._run_kpis(cfg) == self._run_kpis(cfg)

    def test_burst_determinism(self) -> None:
        cfg = _base_cfg(
            generator_type="BurstDemandGenerator",
            rate=0.03,
            burst_rate_factor=4.0,
            burst_duration_fraction=0.15,
            burst_interval_fraction=0.25,
        )
        assert self._run_kpis(cfg) == self._run_kpis(cfg)

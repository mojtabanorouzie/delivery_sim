"""Heavy-seed determinism — adversarial reproducibility verification.

Tests that the combination of:
  - saturated courier queue (heavy demand, few couriers)
  - return_rate > 0 (returns_rng draws every delivery attempt)
  - time-varying BurstDemandGenerator (gen_rng + burst phase logic)

produces byte-identical results across two independent runs with the same seed.

"Byte-identical" means:
  - every order's status, all timestamps dict entries, delivery_cost
  - every courier's final phase
  - all KPI summary values

This is the adversarial case: it exercises all three new randomness streams
(gen_rng, returns_rng, burst phase) simultaneously under high contention.
"""

from __future__ import annotations

from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    ProfileBreakpoint,
    ScenarioConfig,
    StoreConfig,
)
from delivery_sim.engine.simulator import Simulator
from delivery_sim.entities.order import OrderStatus
from delivery_sim.metrics.collector import KPICollector


def _adversarial_cfg(seed: int = 42) -> ScenarioConfig:
    """Heavy + returns + burst demand — all new randomness interacting at once.

    Parameters chosen to:
    - Saturate the store queue: capacity=2 forces simultaneous enqueues
    - Fire many return draws: return_rate=0.15 with high order volume
    - Exercise BurstDemandGenerator: alternates high/low demand phases
    - High demand vs few couriers: demand >> throughput → courier contention
    """
    return ScenarioConfig(
        name="adversarial_heavy",
        seed=seed,
        dt=1.0,
        max_steps=5000,
        return_rate=0.15,
        stores=[StoreConfig(
            name="s",
            x=500.0, y=500.0,
            capacity=2,        # deliberately low → queue contention
            prep_time=60.0,
            coverage_radius=900.0,
        )],
        couriers=[CourierConfig(courier_type="BikeCourier", count=6, speed=2.0)],
        demand=DemandConfig(
            generator_type="BurstDemandGenerator",
            rate=0.035,
            burst_rate_factor=4.0,
            burst_duration_fraction=0.15,
            burst_interval_fraction=0.25,
        ),
    )


def _run_full(cfg: ScenarioConfig) -> tuple[KPICollector, Simulator]:
    sim = Simulator(cfg)
    col = KPICollector()
    sim.attach_collector(col)
    sim.run()
    return col, sim


def _order_fingerprint(sim: Simulator) -> dict:
    """Extract per-order state: status, all timestamps, delivery_cost."""
    assert sim.world is not None
    return {
        order_id: {
            "status": o.status.name,
            "timestamps": {s.name: t for s, t in sorted(o.timestamps.items(),
                                                         key=lambda x: x[0].value)},
            "delivery_cost": o.delivery_cost,
            "assigned_courier_id": o.assigned_courier_id,
        }
        for order_id, o in sorted(sim.world.active_orders.items())
    }


def _courier_fingerprint(sim: Simulator) -> dict:
    """Extract final courier phases."""
    assert sim.world is not None
    return dict(sorted(sim.world.courier_phase.items()))


class TestHeavySeedDeterminism:
    def test_byte_identical_order_histories(self) -> None:
        """Two runs with identical config+seed produce identical per-order data."""
        cfg = _adversarial_cfg(seed=42)
        _, sim_a = _run_full(cfg)
        _, sim_b = _run_full(cfg)

        fp_a = _order_fingerprint(sim_a)
        fp_b = _order_fingerprint(sim_b)

        assert fp_a == fp_b, (
            "Order histories diverged between two runs with identical seed. "
            f"Differing orders: {sorted(set(fp_a) ^ set(fp_b)) or 'same keys, different values'}"
        )

    def test_byte_identical_kpis(self) -> None:
        """Two runs produce identical KPI summary."""
        cfg = _adversarial_cfg(seed=42)
        col_a, _ = _run_full(cfg)
        col_b, _ = _run_full(cfg)
        assert col_a.summary() == col_b.summary()

    def test_byte_identical_courier_phases(self) -> None:
        """Courier phases at end of episode are identical across runs."""
        cfg = _adversarial_cfg(seed=42)
        _, sim_a = _run_full(cfg)
        _, sim_b = _run_full(cfg)
        assert _courier_fingerprint(sim_a) == _courier_fingerprint(sim_b)

    def test_returned_orders_exist_in_adversarial_run(self) -> None:
        """Confirm returns_rng is actually firing: adversarial run produces RETURNED orders."""
        cfg = _adversarial_cfg(seed=42)
        col, sim = _run_full(cfg)
        assert sim.world is not None
        returned = [
            o for o in sim.world.active_orders.values()
            if o.status == OrderStatus.RETURNED
        ]
        assert len(returned) > 0, (
            "Expected RETURNED orders with return_rate=0.15 on heavy load, got none. "
            "returns_rng may not be drawing."
        )
        assert col.summary()["returned_orders"] == len(returned)

    def test_store_queue_saturated_in_adversarial_run(self) -> None:
        """Confirm store queue actually forms (capacity=2 + heavy demand)."""
        cfg = _adversarial_cfg(seed=42)
        col, _ = _run_full(cfg)
        kpis = col.summary()
        assert kpis["max_store_queue_depth"] > 0, (
            "Expected store queue to form with capacity=2 + heavy demand. "
            "Queue contention not exercised."
        )

    def test_determinism_across_multiple_seeds(self) -> None:
        """Adversarial scenario is deterministic for 5 different seeds."""
        for seed in [7, 13, 99, 333, 1001]:
            cfg = _adversarial_cfg(seed=seed)
            col_a, sim_a = _run_full(cfg)
            col_b, sim_b = _run_full(cfg)
            assert col_a.summary() == col_b.summary(), (
                f"KPI mismatch for seed={seed}"
            )
            assert _order_fingerprint(sim_a) == _order_fingerprint(sim_b), (
                f"Order history mismatch for seed={seed}"
            )

    def test_different_seeds_produce_different_histories(self) -> None:
        """Two different seeds must not produce the same order history (sanity check)."""
        _, sim_42  = _run_full(_adversarial_cfg(seed=42))
        _, sim_999 = _run_full(_adversarial_cfg(seed=999))
        fp_42  = _order_fingerprint(sim_42)
        fp_999 = _order_fingerprint(sim_999)
        # Different seeds → different demand streams → different histories
        assert fp_42 != fp_999


class TestHeavySeedDeterminismWithDailyProfile:
    """Same adversarial test with DailyProfileDemandGenerator instead of Burst."""

    def _profile_cfg(self, seed: int = 42) -> ScenarioConfig:
        return ScenarioConfig(
            name="adversarial_daily",
            seed=seed,
            dt=1.0,
            max_steps=5000,
            return_rate=0.15,
            stores=[StoreConfig(
                name="s",
                x=500.0, y=500.0,
                capacity=2,
                prep_time=60.0,
                coverage_radius=900.0,
            )],
            couriers=[CourierConfig(courier_type="BikeCourier", count=6, speed=2.0)],
            demand=DemandConfig(
                generator_type="DailyProfileDemandGenerator",
                rate=0.035,
                profile=[
                    ProfileBreakpoint(time_fraction=0.0, rate_factor=0.2),
                    ProfileBreakpoint(time_fraction=0.5, rate_factor=4.0),
                    ProfileBreakpoint(time_fraction=1.0, rate_factor=0.2),
                ],
            ),
        )

    def test_byte_identical_with_daily_profile(self) -> None:
        cfg = self._profile_cfg(seed=77)
        col_a, sim_a = _run_full(cfg)
        col_b, sim_b = _run_full(cfg)
        assert col_a.summary() == col_b.summary()
        assert _order_fingerprint(sim_a) == _order_fingerprint(sim_b)

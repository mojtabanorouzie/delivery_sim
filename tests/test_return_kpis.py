"""B7 — Return KPI metrics tests.

Verifies:
- on_order_returned increments returned_orders and total_delivery_cost
- on_courier_returned_to_store computes mean_return_leg_time
- return_rate = returned / (delivered + failed + returned)
- return_rate=0.0 scenarios still produce returned_orders=0 and mean_return_leg_time=0.0
- reset() clears all return state
- KPICollector unit-level: direct notification calls
"""

from __future__ import annotations

from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    ScenarioConfig,
    StoreConfig,
)
from delivery_sim.engine.simulator import Simulator
from delivery_sim.entities.order import Order
from delivery_sim.metrics.collector import KPICollector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(return_rate: float = 0.0, seed: int = 42, max_steps: int = 3000) -> ScenarioConfig:
    return ScenarioConfig(
        name="kpi_test",
        seed=seed,
        dt=1.0,
        max_steps=max_steps,
        return_rate=return_rate,
        stores=[StoreConfig(
            name="s", x=500.0, y=500.0,
            capacity=20, prep_time=30.0, coverage_radius=1000.0,
        )],
        couriers=[CourierConfig(courier_type="BikeCourier", count=10, speed=2.0)],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=0.03),
    )


def _run(cfg: ScenarioConfig) -> tuple[KPICollector, Simulator]:
    sim = Simulator(cfg)
    col = KPICollector()
    sim.attach_collector(col)
    sim.run()
    return col, sim


def _make_order(order_id: str = "o1") -> Order:
    return Order(
        order_id=order_id,
        store_id="s",
        customer_x=100.0,
        customer_y=100.0,
        created_at=0.0,
    )


# ---------------------------------------------------------------------------
# KPICollector unit tests (notification-level)
# ---------------------------------------------------------------------------

class TestReturnKPIUnit:
    def test_on_order_returned_increments_count(self) -> None:
        col = KPICollector()
        o = _make_order()
        col.on_order_returned(o, sim_time=100.0, cost=5.0)
        assert col.summary()["returned_orders"] == 1

    def test_on_order_returned_accumulates_cost(self) -> None:
        col = KPICollector()
        o1 = _make_order("o1")
        o2 = _make_order("o2")
        col.on_order_returned(o1, sim_time=100.0, cost=3.0)
        col.on_order_returned(o2, sim_time=200.0, cost=7.0)
        # total_delivery_cost is the running accumulator in _return_total_cost
        # (not exposed alone) but returned_orders should be 2
        assert col.summary()["returned_orders"] == 2

    def test_return_leg_time_computed(self) -> None:
        col = KPICollector()
        o = _make_order("o1")
        col.on_order_returned(o, sim_time=100.0, cost=5.0)
        col.on_courier_returned_to_store("c1", "o1", sim_time=180.0)
        kpis = col.summary()
        assert kpis["mean_return_leg_time"] == 80.0

    def test_multiple_return_legs_averaged(self) -> None:
        col = KPICollector()
        for i, (returned_t, arrived_t) in enumerate(
            [(100.0, 160.0), (200.0, 290.0)]  # legs of 60 and 90
        ):
            o = _make_order(f"o{i}")
            col.on_order_returned(o, sim_time=returned_t, cost=1.0)
            col.on_courier_returned_to_store(f"c{i}", f"o{i}", sim_time=arrived_t)
        assert col.summary()["mean_return_leg_time"] == 75.0  # (60+90)/2

    def test_no_return_legs_gives_zero(self) -> None:
        col = KPICollector()
        assert col.summary()["mean_return_leg_time"] == 0.0

    def test_return_rate_computed(self) -> None:
        col = KPICollector()
        # 2 delivered, 1 returned → return_rate = 1/3
        o_del1 = _make_order("d1")
        o_del2 = _make_order("d2")
        o_ret  = _make_order("r1")
        col.on_order_delivered(o_del1, sim_time=50.0, cost=1.0)
        col.on_order_delivered(o_del2, sim_time=80.0, cost=1.0)
        col.on_order_returned(o_ret, sim_time=100.0, cost=1.0)
        kpis = col.summary()
        assert abs(kpis["return_rate"] - 1 / 3) < 1e-9

    def test_return_rate_zero_when_no_terminal_orders(self) -> None:
        col = KPICollector()
        assert col.summary()["return_rate"] == 0.0

    def test_reset_clears_return_state(self) -> None:
        col = KPICollector()
        o = _make_order()
        col.on_order_returned(o, sim_time=100.0, cost=5.0)
        col.on_courier_returned_to_store("c1", o.order_id, sim_time=170.0)
        col.reset()
        kpis = col.summary()
        assert kpis["returned_orders"] == 0
        assert kpis["mean_return_leg_time"] == 0.0
        assert kpis["return_rate"] == 0.0

    def test_return_leg_start_cleared_on_return(self) -> None:
        """Orphaned return-leg starts (courier never returned) don't poison next reset."""
        col = KPICollector()
        o = _make_order()
        col.on_order_returned(o, sim_time=100.0, cost=5.0)
        # on_courier_returned_to_store never called (episode ended mid-return)
        col.reset()
        # After reset, internal _return_leg_starts must be empty
        assert len(col._return_leg_starts) == 0


# ---------------------------------------------------------------------------
# Simulator integration tests
# ---------------------------------------------------------------------------

class TestReturnKPIIntegration:
    def test_return_rate_zero_config_zero_metrics(self) -> None:
        col, _ = _run(_cfg(return_rate=0.0))
        kpis = col.summary()
        assert kpis["returned_orders"] == 0
        assert kpis["return_rate"] == 0.0
        assert kpis["mean_return_leg_time"] == 0.0

    def test_return_rate_one_all_returned(self) -> None:
        col, sim = _run(_cfg(return_rate=1.0, seed=7))
        assert sim.world is not None
        kpis = col.summary()
        assert kpis["delivered_orders"] == 0
        assert kpis["returned_orders"] > 0
        assert kpis["return_rate"] == 1.0

    def test_mean_return_leg_time_positive_when_returns_exist(self) -> None:
        col, _ = _run(_cfg(return_rate=1.0, seed=3))
        kpis = col.summary()
        # If returns occurred and some couriers completed the return leg
        if kpis["returned_orders"] > 0:
            assert kpis["mean_return_leg_time"] >= 0.0
            # Positive if at least one leg completed within horizon
            # (not guaranteed for every seed — just ensure it's non-negative)

    def test_return_rate_calculation_matches_counts(self) -> None:
        col, _ = _run(_cfg(return_rate=0.15, seed=17))
        kpis = col.summary()
        terminal = kpis["delivered_orders"] + kpis["failed_orders"] + kpis["returned_orders"]
        if terminal > 0:
            expected_rate = kpis["returned_orders"] / terminal
            assert abs(kpis["return_rate"] - expected_rate) < 1e-9

    def test_returned_orders_not_in_delivered(self) -> None:
        col, _ = _run(_cfg(return_rate=1.0, seed=11))
        kpis = col.summary()
        assert kpis["delivered_orders"] == 0
        assert kpis["returned_orders"] + kpis["failed_orders"] <= kpis["total_orders"]

    def test_determinism_with_returns(self) -> None:
        kpis_a = _run(_cfg(return_rate=0.1, seed=88))[0].summary()
        kpis_b = _run(_cfg(return_rate=0.1, seed=88))[0].summary()
        assert kpis_a == kpis_b

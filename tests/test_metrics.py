"""
Tests for KPICollector (Step 4).

Test inventory
--------------
1. KPI correctness: counts and delivery times match world.active_orders.
2. Cost correctness: total_delivery_cost matches manual calculation.
3. dt-independence: same run at different dt cadences yields identical KPIs.
4. Collector non-interference: with vs without collector produces identical
   order histories and delivery counts.
5. FAILED / uncovered orders counted correctly.
6. Reproducibility: same (config, seed) => identical summary().
7. Degenerate: zero deliveries => all metrics well-defined (no div-by-zero).
8. Degenerate: zero couriers => utilisation = 0.0, no crash.
9. Courier utilisation in [0, 1].
10. SLA violations counted when delivery_time > sla_seconds.
"""

from __future__ import annotations

import math

import pytest

import delivery_sim.entities  # noqa: F401 — trigger @register decorators
import delivery_sim.routing  # noqa: F401 — trigger @register decorators
from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    RoutingConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.engine.event_queue import Event
from delivery_sim.engine.simulator import Simulator
from delivery_sim.entities.order import Order, OrderStatus
from delivery_sim.metrics.collector import KPICollector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(
    *,
    seed: int = 42,
    rate: float = 1.0,
    num_couriers: int = 2,
    max_steps: int = 200,
    dt: float = 1.0,
    speed: float = 10.0,
    coverage_radius: float = 1500.0,
    cost_per_unit: float = 0.01,
) -> ScenarioConfig:
    """Minimal scenario — one store at (0, 0), couriers spawn at origin."""
    return ScenarioConfig(
        name="test_metrics",
        seed=seed,
        dt=dt,
        max_steps=max_steps,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[StoreConfig(
            name="s1", x=0.0, y=0.0,
            capacity=20, coverage_radius=coverage_radius,
        )],
        couriers=[CourierConfig(
            courier_type="BikeCourier",
            count=num_couriers,
            speed=speed,
            cost_per_unit=cost_per_unit,
        )],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=rate),
        routing=RoutingConfig(model_type="euclidean"),
    )


def _run_with_collector(config: ScenarioConfig, **kw: float) -> tuple[Simulator, KPICollector]:
    """Run simulator and return (sim, collector) after episode."""
    collector = KPICollector(**kw)
    sim = Simulator(config)
    sim.attach_collector(collector)
    sim.run()
    return sim, collector


# ---------------------------------------------------------------------------
# Test 1 — KPI count correctness
# ---------------------------------------------------------------------------


def test_kpi_counts_match_world_active_orders() -> None:
    """total / delivered / failed counts in summary() must match world state."""
    config = make_config(seed=1, rate=2.0, max_steps=150)
    sim, collector = _run_with_collector(config)

    assert sim.world is not None
    orders = list(sim.world.active_orders.values())
    expected_total = len(orders)
    expected_delivered = sum(1 for o in orders if o.status == OrderStatus.DELIVERED)
    expected_failed = sum(1 for o in orders if o.status == OrderStatus.FAILED)

    kpis = collector.summary()
    assert kpis["total_orders"] == expected_total
    assert kpis["delivered_orders"] == expected_delivered
    assert kpis["failed_orders"] == expected_failed
    assert expected_delivered > 0, "test requires at least one delivery"


def test_delivery_rate_equals_fraction() -> None:
    """delivery_rate == delivered / total."""
    config = make_config(seed=2, rate=1.5, max_steps=100)
    _, collector = _run_with_collector(config)
    kpis = collector.summary()
    expected_rate = (
        kpis["delivered_orders"] / kpis["total_orders"]
        if kpis["total_orders"] > 0
        else 0.0
    )
    assert kpis["delivery_rate"] == pytest.approx(expected_rate)


def test_delivery_times_match_order_timestamps() -> None:
    """mean_delivery_time must equal the mean of (DELIVERED_t - CREATED_t)."""
    config = make_config(seed=3, rate=2.0, max_steps=150)
    sim, collector = _run_with_collector(config)

    assert sim.world is not None
    times = [
        o.timestamps[OrderStatus.DELIVERED] - o.timestamps[OrderStatus.CREATED]
        for o in sim.world.active_orders.values()
        if o.status == OrderStatus.DELIVERED
    ]
    assert times, "test requires at least one delivered order"
    expected_mean = sum(times) / len(times)

    kpis = collector.summary()
    assert kpis["mean_delivery_time"] == pytest.approx(expected_mean, rel=1e-9)


def test_p95_delivery_time_no_less_than_p50() -> None:
    """p95 >= p50 for any non-empty delivery set."""
    config = make_config(seed=4, rate=2.0, max_steps=150)
    _, collector = _run_with_collector(config)
    kpis = collector.summary()
    if kpis["delivered_orders"] > 0:
        assert kpis["p95_delivery_time"] >= kpis["p50_delivery_time"]


# ---------------------------------------------------------------------------
# Test 2 — Cost correctness
# ---------------------------------------------------------------------------


def test_cost_matches_manual_calculation() -> None:
    """total_delivery_cost must match cost_per_unit * distance for each leg.

    Store and courier spawn are both at (0, 0).  For the *first* assignment of
    each courier after episode reset the courier sits at the origin, which
    coincides with the store, so leg-1 distance is exactly 0 and
    total_cost_for_that_order = cost_per_unit * distance(store, customer).

    The test uses two assertions:

    1. Lower-bound (always applicable): total_cost >= sum of leg-2 costs
       (leg-1 >= 0, so adding it can only increase the total).

    2. Exact equality when every courier made exactly one delivery in the
       episode (leg-1 = 0 for all deliveries → cost = leg-2 cost only).

    horizon = 200 s >> prep_time (30 s) + max_travel_time, so orders always
    complete; no skip guard is needed.
    """
    import math as _math
    from collections import Counter

    config = make_config(
        seed=99, rate=0.5, max_steps=200, speed=50.0,
        coverage_radius=2000.0, cost_per_unit=0.05,
    )
    sim, collector = _run_with_collector(config)
    assert sim.world is not None

    delivered = [
        o for o in sim.world.active_orders.values()
        if o.status == OrderStatus.DELIVERED
    ]
    assert delivered, "horizon=200s must produce at least one delivered order"

    kpis = collector.summary()
    assert kpis["total_delivery_cost"] > 0.0

    store = sim.world.stores[0]
    leg2_sum = sum(
        0.05 * _math.hypot(o.customer_x - store.x, o.customer_y - store.y)
        for o in delivered
    )

    # Assertion 1 — lower bound (holds regardless of how many trips per courier)
    assert kpis["total_delivery_cost"] >= leg2_sum - 1e-9

    # Assertion 2 — exact equality when each courier made at most 1 delivery
    deliveries_per_courier = Counter(o.assigned_courier_id for o in delivered)
    if max(deliveries_per_courier.values()) == 1:
        assert kpis["total_delivery_cost"] == pytest.approx(leg2_sum, rel=1e-9)


def test_cost_nonnegative() -> None:
    config = make_config(seed=10, rate=2.0, max_steps=100)
    _, collector = _run_with_collector(config)
    assert collector.summary()["total_delivery_cost"] >= 0.0


# ---------------------------------------------------------------------------
# Test 3 — dt-independence  *** THE KEY TEST ***
# ---------------------------------------------------------------------------


def test_dt_independence_same_kpis_at_different_cadences() -> None:
    """Identical KPIs at dt=1.0 (200 steps) and dt=0.1 (2000 steps).

    Both configs produce horizon = 200.0 s, same seed, same events.
    This test proves the event-driven design: metrics are determined by event
    timestamps, not by the observer cadence.
    """
    horizon = 200.0

    config_dt1 = make_config(seed=7, rate=1.5, dt=1.0,
                             max_steps=int(horizon / 1.0))
    config_dt01 = make_config(seed=7, rate=1.5, dt=0.1,
                              max_steps=int(horizon / 0.1))

    _, col1 = _run_with_collector(config_dt1)
    _, col2 = _run_with_collector(config_dt01)

    kpis1 = col1.summary()
    kpis2 = col2.summary()

    for key in (
        "total_orders", "delivered_orders", "failed_orders",
        "sla_violations",
    ):
        assert kpis1[key] == kpis2[key], (
            f"dt-independence violated for {key!r}: "
            f"dt=1.0 → {kpis1[key]}, dt=0.1 → {kpis2[key]}"
        )

    for key in (
        "delivery_rate", "mean_delivery_time", "p50_delivery_time",
        "p95_delivery_time", "mean_pickup_latency",
        "courier_utilization", "total_delivery_cost",
    ):
        assert kpis1[key] == pytest.approx(kpis2[key], rel=1e-9, abs=1e-12), (
            f"dt-independence violated for {key!r}: "
            f"dt=1.0 → {kpis1[key]:.6f}, dt=0.1 → {kpis2[key]:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Collector non-interference
# ---------------------------------------------------------------------------


def test_collector_does_not_alter_order_histories() -> None:
    """A run with a collector attached must produce identical order histories
    (status, timestamps, assigned_courier_id) to a headless run."""
    config = make_config(seed=42, rate=2.0, max_steps=100)

    sim_headless = Simulator(config)
    sim_headless.run()

    sim_observed = Simulator(config)
    sim_observed.attach_collector(KPICollector())
    sim_observed.run()

    assert sim_headless.world is not None
    assert sim_observed.world is not None

    def history(sim: Simulator) -> dict:  # type: ignore[type-arg]
        assert sim.world is not None
        return {
            oid: (o.status, dict(o.timestamps), o.assigned_courier_id)
            for oid, o in sim.world.active_orders.items()
        }

    assert history(sim_headless) == history(sim_observed)


def test_collector_does_not_alter_delivery_count() -> None:
    config = make_config(seed=55, rate=2.0, max_steps=150)

    def count_delivered(sim: Simulator) -> int:
        assert sim.world is not None
        return sum(
            1 for o in sim.world.active_orders.values()
            if o.status == OrderStatus.DELIVERED
        )

    sim_a = Simulator(config)
    sim_a.run()

    sim_b = Simulator(config)
    sim_b.attach_collector(KPICollector())
    sim_b.run()

    assert count_delivered(sim_a) == count_delivered(sim_b)


# ---------------------------------------------------------------------------
# Test 5 — FAILED / uncovered orders
# ---------------------------------------------------------------------------


def test_failed_uncovered_orders_counted() -> None:
    """coverage_radius=0.0 → all orders are uncovered → all FAILED.

    failed_orders must equal total_orders, delivered_orders == 0.
    """
    config = make_config(
        seed=11, rate=2.0, max_steps=50, coverage_radius=0.0,
    )
    _, collector = _run_with_collector(config)
    kpis = collector.summary()

    assert kpis["delivered_orders"] == 0
    assert kpis["failed_orders"] == kpis["total_orders"]
    assert kpis["total_orders"] > 0, "need orders to test FAILED path"


def test_failed_orders_not_in_delivery_times() -> None:
    """Failed orders must not appear in delivery_time statistics."""
    config = make_config(
        seed=12, rate=2.0, max_steps=50, coverage_radius=0.0,
    )
    _, collector = _run_with_collector(config)
    kpis = collector.summary()
    assert kpis["mean_delivery_time"] == pytest.approx(0.0)
    assert kpis["p50_delivery_time"] == pytest.approx(0.0)
    assert kpis["p95_delivery_time"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 6 — Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_same_seed_identical_kpis() -> None:
    """Two runs with the same (config, seed) must produce identical summary()."""
    config = make_config(seed=99, rate=1.5, max_steps=100)

    _, col1 = _run_with_collector(config)
    _, col2 = _run_with_collector(config)

    assert col1.summary() == col2.summary()


def test_reproducibility_different_seeds_differ() -> None:
    """Different seeds should (almost certainly) produce different KPIs."""
    _, col1 = _run_with_collector(make_config(seed=1, rate=2.0, max_steps=100))
    _, col2 = _run_with_collector(make_config(seed=2, rate=2.0, max_steps=100))
    assert col1.summary() != col2.summary()


# ---------------------------------------------------------------------------
# Test 7 — Degenerate: zero deliveries
# ---------------------------------------------------------------------------


def test_degenerate_zero_demand_no_div_by_zero() -> None:
    """rate=0 → no orders → all KPI stats must be 0.0 with no exception."""
    config = make_config(rate=0.0, max_steps=100)
    _, collector = _run_with_collector(config)
    kpis = collector.summary()

    assert kpis["total_orders"] == 0
    assert kpis["delivered_orders"] == 0
    assert kpis["failed_orders"] == 0
    assert kpis["delivery_rate"] == pytest.approx(0.0)
    assert kpis["mean_delivery_time"] == pytest.approx(0.0)
    assert kpis["p50_delivery_time"] == pytest.approx(0.0)
    assert kpis["p95_delivery_time"] == pytest.approx(0.0)
    assert kpis["mean_pickup_latency"] == pytest.approx(0.0)
    assert kpis["courier_utilization"] == pytest.approx(0.0)
    assert kpis["total_delivery_cost"] == pytest.approx(0.0)
    assert kpis["sla_violations"] == 0


def test_degenerate_zero_couriers_no_crash() -> None:
    """Zero couriers → orders created but none delivered; no exception."""
    config = make_config(num_couriers=0, rate=2.0, max_steps=50)
    _, collector = _run_with_collector(config)
    kpis = collector.summary()

    assert kpis["delivered_orders"] == 0
    assert kpis["courier_utilization"] == pytest.approx(0.0)
    assert kpis["total_orders"] > 0


# ---------------------------------------------------------------------------
# Test 8 — Courier utilisation bounds
# ---------------------------------------------------------------------------


def test_courier_utilization_in_unit_interval() -> None:
    """Courier utilisation must be in [0.0, 1.0]."""
    config = make_config(seed=20, rate=3.0, max_steps=100)
    _, collector = _run_with_collector(config)
    util = collector.summary()["courier_utilization"]
    assert 0.0 <= util <= 1.0


def test_courier_utilization_positive_with_deliveries() -> None:
    """When deliveries happen, couriers were busy → utilisation > 0."""
    config = make_config(seed=21, rate=2.0, max_steps=150)
    _, collector = _run_with_collector(config)
    kpis = collector.summary()
    if kpis["delivered_orders"] > 0:
        assert kpis["courier_utilization"] > 0.0


# ---------------------------------------------------------------------------
# Test 9 — SLA violations
# ---------------------------------------------------------------------------


def test_sla_violations_counted_when_tight_threshold() -> None:
    """SLA = 1 second on a run with normal delivery times → all orders violate."""
    config = make_config(seed=30, rate=1.0, max_steps=100)
    _, collector = _run_with_collector(config, sla_seconds=1.0)
    kpis = collector.summary()
    if kpis["delivered_orders"] > 0:
        assert kpis["sla_violations"] == kpis["delivered_orders"]


def test_sla_violations_zero_with_infinite_threshold() -> None:
    """SLA = inf → no violations regardless of delivery time."""
    config = make_config(seed=31, rate=1.0, max_steps=100)
    _, collector = _run_with_collector(config, sla_seconds=math.inf)
    assert collector.summary()["sla_violations"] == 0


# ---------------------------------------------------------------------------
# Test 10 — reset() clears state
# ---------------------------------------------------------------------------


def test_reset_clears_all_state() -> None:
    """After reset(), summary() must be as if no orders were processed."""
    config = make_config(seed=42, rate=2.0, max_steps=100)
    _, collector = _run_with_collector(config)
    assert collector.summary()["total_orders"] > 0

    collector.reset()
    kpis = collector.summary()
    assert kpis["total_orders"] == 0
    assert kpis["delivered_orders"] == 0
    assert kpis["failed_orders"] == 0
    assert kpis["total_delivery_cost"] == pytest.approx(0.0)
    assert kpis["courier_utilization"] == pytest.approx(0.0)
    assert kpis["mean_delivery_time"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 11 — _on_order_terminal structural fix: no cost entry leaks
# ---------------------------------------------------------------------------


class _LeakySimulator(Simulator):
    """Test double that forgets to call _on_order_terminal when an in-flight
    order fails at the customer — simulates the step-5 obligation being missed.

    Overrides _handle_courier_arrived_customer to transition the order to
    FAILED (valid per ALLOWED_TRANSITIONS: IN_TRANSIT → FAILED) and free the
    courier, but deliberately omits the _on_order_terminal() call, leaving
    the accumulated leg cost in _order_leg_cost.  The tripwire in run() must
    fire when it finds a terminal order with a live cost entry.
    """

    def _handle_courier_arrived_customer(self, event: Event) -> None:
        assert self.world is not None
        sim_time = self.clock.elapsed
        courier_id: str = event.payload["courier_id"]
        order_id: str = event.payload["order_id"]
        order = self.world.active_orders[order_id]
        # Fail the order in-transit without cleanup — this is the bug under test.
        order.transition(OrderStatus.FAILED, sim_time)
        self.world.courier_phase[courier_id] = "free"
        # Deliberately NOT calling self._on_order_terminal(order, sim_time).


def test_on_order_terminal_pops_entry_for_assigned_to_failed() -> None:
    """Simulate the step-5 path: courier assigned (cost entry created in
    _order_leg_cost), order then transitions to FAILED.  _on_order_terminal
    must remove the entry so the cost cannot leak.
    """
    config = make_config(seed=1, rate=0.0, max_steps=10)  # no demand; just need world
    sim = Simulator(config)
    sim.reset()

    # Inject a cost entry exactly as _dispatch does for an ASSIGNED order.
    sim._order_leg_cost["order_099999"] = 7.5

    # Build a minimal order that went ASSIGNED → FAILED (a valid transition).
    o = Order(
        order_id="order_099999",
        store_id="s1",
        customer_x=50.0,
        customer_y=50.0,
        created_at=1.0,
    )
    o.transition(OrderStatus.ASSIGNED, 1.5)
    o.transition(OrderStatus.FAILED, 2.0)

    sim._on_order_terminal(o, 2.0)

    assert "order_099999" not in sim._order_leg_cost, (
        "_on_order_terminal must pop the cost entry for a FAILED in-flight order"
    )


def test_tripwire_fires_when_terminal_order_not_cleaned_up() -> None:
    """A handler that transitions an order to FAILED without calling
    _on_order_terminal must trigger the end-of-run tripwire assertion.

    Uses _LeakySimulator, which omits the cleanup call.  The assertion must
    mention _order_leg_cost so developers can trace the failure.
    """
    config = make_config(seed=42, rate=1.0, max_steps=200)
    with pytest.raises(AssertionError, match="_order_leg_cost"):
        _LeakySimulator(config).run()


def test_no_leaked_cost_entries_after_normal_run() -> None:
    """After a normal run(), _order_leg_cost must contain no entries for orders
    that reached a terminal state (DELIVERED or FAILED).  Non-terminal
    in-flight orders may legitimately have entries; those are dropped by
    reset().
    """
    config = make_config(seed=42, rate=2.0, max_steps=200)
    sim = Simulator(config)
    sim.run()

    assert sim.world is not None
    leaked = {
        oid for oid in sim._order_leg_cost
        if oid in sim.world.active_orders
        and sim.world.active_orders[oid].is_terminal
    }
    assert not leaked, (
        f"_order_leg_cost leaked entries for terminal orders: {leaked}"
    )

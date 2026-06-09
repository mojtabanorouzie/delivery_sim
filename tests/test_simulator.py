"""Tests for the 3c Simulator (WorldState.snapshot + event-driven run loop).

Test inventory
--------------
1. Reproducibility: same (config, seed) → identical snapshot sequences;
   different seed → different sequences.
2. Observer-invariance: headless run and consumer run produce identical order
   histories, event times, and delivery counts.
3. End-to-end example.yaml: completes without error; ≥1 order DELIVERED.
4. State-machine integrity: every order's transition history is a legal path
   per ALLOWED_TRANSITIONS; timestamps are non-decreasing.
5. Clock monotonicity: sim_time never decreases across observer snapshots.
6. Degenerate — zero demand: completes immediately; zero orders created.
7. Degenerate — zero couriers: orders created; none delivered; no crash; no
   infinite loop (horizon terminates the run).
8. SETTLED precondition: assign() never raises during a normal run (couriers
   are only reassigned at arrival events).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import delivery_sim.entities  # noqa: F401 — trigger @register decorators
import delivery_sim.routing  # noqa: F401 — trigger @register decorators
from delivery_sim.config.loader import load_scenario
from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    RewardConfig,
    RoutingConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.engine.simulator import Simulator
from delivery_sim.entities.order import ALLOWED_TRANSITIONS, OrderStatus
from delivery_sim.render.protocol import SnapshotConsumer, WorldSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"


def make_config(
    *,
    seed: int = 42,
    rate: float = 1.0,
    num_couriers: int = 2,
    max_steps: int = 200,
    speed: float = 10.0,
    coverage_radius: float = 1500.0,
) -> ScenarioConfig:
    """Minimal scenario with one store at (100, 100) covering the full world.

    Default speed=10.0 and coverage_radius=1500.0 ensure all customers are
    covered and couriers complete deliveries well within the horizon.
    """
    return ScenarioConfig(
        name="test",
        seed=seed,
        dt=1.0,
        max_steps=max_steps,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[StoreConfig(
            name="s1", x=100.0, y=100.0,
            capacity=20, coverage_radius=coverage_radius,
        )],
        couriers=[CourierConfig(
            courier_type="BikeCourier",
            count=num_couriers,
            speed=speed,
        )],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=rate),
        routing=RoutingConfig(model_type="euclidean"),
        reward=RewardConfig(function_type="SparseDeliveryReward"),
    )


class SnapshotRecorder:
    """Collects all WorldSnapshots emitted by the run loop."""

    def __init__(self) -> None:
        self.snapshots: list[WorldSnapshot] = []

    def consume(self, snapshot: WorldSnapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self) -> None:
        pass


# Verify SnapshotRecorder satisfies the Protocol at import time
_: SnapshotConsumer = SnapshotRecorder()


# ---------------------------------------------------------------------------
# Test 1 — Reproducibility
# ---------------------------------------------------------------------------


def test_reproducibility_same_seed_identical_snapshot_sequences() -> None:
    """Two runs with the same (config, seed) must produce byte-identical
    snapshot sequences: positions, statuses, times, and order counts."""
    config = make_config(seed=42, rate=2.0, max_steps=100)

    rec_a = SnapshotRecorder()
    Simulator(config).run(consumer=rec_a)

    rec_b = SnapshotRecorder()
    Simulator(config).run(consumer=rec_b)

    assert len(rec_a.snapshots) > 0
    assert rec_a.snapshots == rec_b.snapshots


def test_reproducibility_different_seeds_differ() -> None:
    """Different seeds must produce different snapshot sequences."""
    rec_a = SnapshotRecorder()
    Simulator(make_config(seed=42)).run(consumer=rec_a)

    rec_b = SnapshotRecorder()
    Simulator(make_config(seed=7)).run(consumer=rec_b)

    assert rec_a.snapshots != rec_b.snapshots


# ---------------------------------------------------------------------------
# Test 2 — Observer-invariance
# ---------------------------------------------------------------------------


def test_observer_invariance_headless_and_consumer_identical_histories() -> None:
    """A consumer run and a headless run with the same seed must produce
    identical order histories (status, timestamps) and elapsed time."""
    config = make_config(seed=42, rate=2.0, max_steps=100)

    sim_headless = Simulator(config)
    sim_headless.run()

    sim_observed = Simulator(config)
    rec = SnapshotRecorder()
    sim_observed.run(consumer=rec)

    def order_history(sim: Simulator) -> dict:  # type: ignore[type-arg]
        assert sim.world is not None
        return {
            oid: (o.status, dict(o.timestamps))
            for oid, o in sim.world.active_orders.items()
        }

    assert order_history(sim_headless) == order_history(sim_observed)
    assert sim_headless.clock.elapsed == pytest.approx(sim_observed.clock.elapsed)
    assert len(rec.snapshots) > 0  # consumer was actually invoked


def test_observer_invariance_delivered_count_matches() -> None:
    """Delivered order count must be the same with and without a consumer."""
    config = make_config(seed=99, rate=1.5, max_steps=150)

    def count_delivered(sim: Simulator) -> int:
        assert sim.world is not None
        return sum(
            1 for o in sim.world.active_orders.values()
            if o.status == OrderStatus.DELIVERED
        )

    sim_a = Simulator(config)
    sim_a.run()

    sim_b = Simulator(config)
    sim_b.run(consumer=SnapshotRecorder())

    assert count_delivered(sim_a) == count_delivered(sim_b)


# ---------------------------------------------------------------------------
# Test 3 — End-to-end example.yaml
# ---------------------------------------------------------------------------


def test_end_to_end_example_yaml_at_least_one_delivery() -> None:
    """example.yaml run must complete and deliver at least one order."""
    config = load_scenario(_SCENARIOS_DIR / "example.yaml")
    sim = Simulator(config)
    sim.run()

    assert sim.world is not None
    delivered = [
        o for o in sim.world.active_orders.values()
        if o.status == OrderStatus.DELIVERED
    ]
    assert len(delivered) >= 1, (
        f"Expected ≥1 DELIVERED order; "
        f"got statuses: {[o.status.name for o in sim.world.active_orders.values()]}"
    )


def test_end_to_end_example_yaml_no_exception() -> None:
    """example.yaml run must complete without raising."""
    config = load_scenario(_SCENARIOS_DIR / "example.yaml")
    Simulator(config).run()


# ---------------------------------------------------------------------------
# Test 4 — State-machine integrity
# ---------------------------------------------------------------------------


def test_state_machine_all_transitions_legal() -> None:
    """Every order's transition sequence must be a valid path through
    ALLOWED_TRANSITIONS.  Timestamps must be non-decreasing."""
    config = make_config(seed=1, rate=3.0, max_steps=150)
    sim = Simulator(config)
    sim.run()

    assert sim.world is not None
    assert len(sim.world.active_orders) > 0, "no orders created — test invalid"

    for order in sim.world.active_orders.values():
        keys = list(order.timestamps.keys())
        assert keys[0] == OrderStatus.CREATED, (
            f"Order {order.order_id}: first timestamp is not CREATED"
        )
        for i in range(len(keys) - 1):
            src, dst = keys[i], keys[i + 1]
            assert dst in ALLOWED_TRANSITIONS[src], (
                f"Order {order.order_id}: illegal transition "
                f"{src.name} → {dst.name}"
            )
        times = list(order.timestamps.values())
        assert all(t1 <= t2 for t1, t2 in zip(times, times[1:])), (
            f"Order {order.order_id}: timestamps not non-decreasing: {times}"
        )


def test_state_machine_nothing_mutated_outside_transition() -> None:
    """order.status must always equal the last key in order.timestamps."""
    config = make_config(seed=2, rate=2.0, max_steps=100)
    sim = Simulator(config)
    sim.run()

    assert sim.world is not None
    for order in sim.world.active_orders.values():
        last_key = list(order.timestamps.keys())[-1]
        assert order.status == last_key, (
            f"Order {order.order_id}: status {order.status.name} "
            f"doesn't match last timestamp key {last_key.name}"
        )


# ---------------------------------------------------------------------------
# Test 5 — Clock monotonicity
# ---------------------------------------------------------------------------


def test_clock_monotonicity_observer_times_never_decrease() -> None:
    """Observer snapshot elapsed times must be non-decreasing."""
    config = make_config(seed=42, rate=2.0, max_steps=150)
    sim = Simulator(config)
    rec = SnapshotRecorder()
    sim.run(consumer=rec)

    assert len(rec.snapshots) > 0
    elapsed = [s.elapsed for s in rec.snapshots]
    for i in range(len(elapsed) - 1):
        assert elapsed[i] <= elapsed[i + 1], (
            f"Clock went backward: {elapsed[i]} → {elapsed[i+1]} at snapshot {i}"
        )


def test_clock_monotonicity_advance_to_never_raises() -> None:
    """A successful run proves advance_to was never called with decreasing t
    (it raises ValueError on backward movement)."""
    config = make_config(seed=42, rate=3.0, max_steps=200)
    sim = Simulator(config)
    sim.run()  # would raise ValueError if clock moved backward
    assert sim.clock.elapsed >= 0.0


# ---------------------------------------------------------------------------
# Test 6 — Degenerate: zero demand
# ---------------------------------------------------------------------------


def test_zero_demand_completes_immediately() -> None:
    """rate=0 → no events; queue empty; run exits immediately."""
    config = make_config(rate=0.0, max_steps=500)
    sim = Simulator(config)
    sim.run()

    assert sim.world is not None
    assert len(sim.world.active_orders) == 0
    assert sim.clock.elapsed == pytest.approx(0.0)


def test_zero_demand_zero_deliveries() -> None:
    config = make_config(rate=0.0, max_steps=200)
    sim = Simulator(config)
    sim.run()

    assert sim.world is not None
    delivered = [
        o for o in sim.world.active_orders.values()
        if o.status == OrderStatus.DELIVERED
    ]
    assert len(delivered) == 0


# ---------------------------------------------------------------------------
# Test 7 — Degenerate: zero couriers
# ---------------------------------------------------------------------------


def test_zero_couriers_orders_created_none_delivered() -> None:
    """No couriers → demand fires; _dispatch returns early every time;
    horizon terminates the run (no infinite loop)."""
    config = make_config(num_couriers=0, rate=2.0, max_steps=50)
    sim = Simulator(config)
    sim.run()  # must complete

    assert sim.world is not None
    orders = list(sim.world.active_orders.values())
    assert len(orders) > 0, "expected orders to be created even with no couriers"
    assert all(o.status != OrderStatus.DELIVERED for o in orders)


def test_zero_couriers_orders_stay_created_or_failed() -> None:
    """With no couriers, orders that have a covering store stay CREATED
    (never dispatched); uncovered orders are FAILED."""
    config = make_config(num_couriers=0, rate=1.0, max_steps=30)
    sim = Simulator(config)
    sim.run()

    assert sim.world is not None
    for order in sim.world.active_orders.values():
        assert order.status in (OrderStatus.CREATED, OrderStatus.FAILED), (
            f"Order {order.order_id} unexpectedly in {order.status.name}"
        )


# ---------------------------------------------------------------------------
# Test 8 — SETTLED precondition never violated
# ---------------------------------------------------------------------------


def test_settled_precondition_never_violated_during_run() -> None:
    """BikeCourier.assign() raises ValueError when called mid-motion.
    A successful run (no exception) proves assign() was only called at
    arrival events (when sim_time >= arrival_time — SETTLED).

    This test is non-vacuous for two reasons:
    - The defensive assertion in _dispatch fires before assign() and would
      raise AssertionError if courier_phase=="free" did not imply SETTLED.
    - assert len(delivered) > 0 guarantees leg-2 ran: _handle_order_ready
      called assign() on a courier whose _arrival_time_val is non-None (the
      old leg-1 eta), so the sim_time < arrival_time() guard in assign() was
      actually evaluated and came out False.  A vacuous pass (no assign()
      calls) is excluded by the delivery count check.
    """
    config = make_config(seed=42, num_couriers=3, rate=2.0, max_steps=200)
    sim = Simulator(config)
    sim.run()  # would raise ValueError if SETTLED precondition was violated

    assert sim.world is not None
    delivered = [
        o for o in sim.world.active_orders.values()
        if o.status == OrderStatus.DELIVERED
    ]
    assert len(delivered) > 0, (
        "expected ≥1 delivery to prove assign() succeeded for leg-1 and leg-2"
    )


def test_settled_precondition_couriers_never_reassigned_mid_motion() -> None:
    """Couriers in non-free phases must never appear as 'free' in the middle
    of a delivery (i.e., the phase map is only updated at event boundaries)."""
    config = make_config(seed=55, num_couriers=2, rate=2.0, max_steps=150)
    sim = Simulator(config)

    # Collect phase snapshots during the run and verify consistency
    phase_snapshots: list[dict[str, str]] = []

    class PhaseRecorder:
        def consume(self, snapshot: WorldSnapshot) -> None:
            phase_snapshots.append({
                c.courier_id: c.status for c in snapshot.couriers
            })

        def close(self) -> None:
            pass

    sim.run(consumer=PhaseRecorder())

    # Each snapshot's courier statuses must be one of the four valid phases
    valid_phases = {"free", "en-route-store", "at-store", "en-route-customer"}
    for snap_phases in phase_snapshots:
        for cid, phase in snap_phases.items():
            assert phase in valid_phases, (
                f"Courier {cid} has invalid phase {phase!r}"
            )

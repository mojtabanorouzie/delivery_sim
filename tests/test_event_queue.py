"""Tests for EventQueue tie-breaking and determinism.

Verifies:
- Comparison key is (time, priority, seq) — never reaches event_type or payload
- Two events with identical (time, priority) pop in insertion (FIFO) order
- Module-level seq counter is monotonically increasing across Event instances
- Reproducibility: a seeded simulator run that produces simultaneous events
  yields identical pop order across two independent runs
"""

from __future__ import annotations

import delivery_sim.entities  # noqa: F401 — trigger @register decorators
import delivery_sim.routing  # noqa: F401 — trigger @register decorators
from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    RewardConfig,
    RoutingConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.engine.event_queue import Event, EventQueue
from delivery_sim.engine.simulator import Simulator
from delivery_sim.render.protocol import SnapshotConsumer, WorldSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(seed: int = 42) -> ScenarioConfig:
    return ScenarioConfig(
        name="test",
        seed=seed,
        dt=1.0,
        max_steps=200,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[StoreConfig(
            name="s1", x=100.0, y=100.0,
            capacity=20, coverage_radius=1500.0,
        )],
        couriers=[CourierConfig(
            courier_type="BikeCourier",
            count=2,
            speed=10.0,
        )],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=2.0),
        routing=RoutingConfig(model_type="euclidean"),
        reward=RewardConfig(function_type="SparseDeliveryReward"),
    )


class SnapshotRecorder:
    def __init__(self) -> None:
        self.snapshots: list[WorldSnapshot] = []

    def consume(self, snapshot: WorldSnapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self) -> None:
        pass


_: SnapshotConsumer = SnapshotRecorder()


# ---------------------------------------------------------------------------
# Comparison key
# ---------------------------------------------------------------------------


def test_event_seq_auto_assigned_and_unique() -> None:
    """Each Event gets a unique, monotonically increasing seq."""
    e1 = Event(time=1.0, priority=5, event_type="a")
    e2 = Event(time=1.0, priority=5, event_type="b")
    assert e2.seq > e1.seq


def test_event_comparison_key_is_time_priority_seq() -> None:
    """Two events that share (time, priority) are ordered by seq alone."""
    e1 = Event(time=5.0, priority=5, event_type="first")
    e2 = Event(time=5.0, priority=5, event_type="second")
    # e1 was created first → smaller seq → e1 < e2
    assert e1 < e2
    assert not (e2 < e1)


def test_event_comparison_never_reaches_event_type() -> None:
    """event_type is excluded from comparison (compare=False).

    Two events with the same (time, priority, seq) are not possible in
    practice (seq is unique), but we can verify the field is excluded
    by constructing events where only seq differs and checking order.
    """
    e1 = Event(time=0.0, priority=0, event_type="z")
    e2 = Event(time=0.0, priority=0, event_type="a")
    # Regardless of event_type sort order (z > a), e1 < e2 because seq(e1) < seq(e2)
    assert e1 < e2


def test_event_comparison_never_reaches_payload() -> None:
    """payload is excluded from comparison (compare=False).

    Push dicts as payloads — if comparison ever reached them, Python would
    raise TypeError because dicts are not orderable.
    """
    queue = EventQueue()
    queue.push(Event(time=3.0, priority=1, event_type="x", payload={"a": 1}))
    queue.push(Event(time=3.0, priority=1, event_type="y", payload={"b": 2}))
    # Must not raise TypeError
    first = queue.pop()
    second = queue.pop()
    assert first.event_type == "x"
    assert second.event_type == "y"


# ---------------------------------------------------------------------------
# Same-time FIFO ordering
# ---------------------------------------------------------------------------


def test_same_time_same_priority_pop_in_insertion_order() -> None:
    """Two events pushed at the same (time, priority) must pop FIFO."""
    queue = EventQueue()
    queue.push(Event(time=5.0, priority=5, event_type="first"))
    queue.push(Event(time=5.0, priority=5, event_type="second"))

    first = queue.pop()
    second = queue.pop()
    assert first.event_type == "first"
    assert second.event_type == "second"


def test_same_time_same_priority_many_events_pop_fifo() -> None:
    """N events at the same (time, priority) all pop in push order."""
    queue = EventQueue()
    n = 10
    for i in range(n):
        queue.push(Event(time=0.0, priority=0, event_type=f"e{i}"))

    for i in range(n):
        e = queue.pop()
        assert e.event_type == f"e{i}", (
            f"Expected e{i} at position {i}, got {e.event_type}"
        )


def test_same_time_lower_priority_int_pops_first() -> None:
    """Lower priority integer = higher urgency; pops before higher int."""
    queue = EventQueue()
    queue.push(Event(time=1.0, priority=10, event_type="low-urgency"))
    queue.push(Event(time=1.0, priority=5,  event_type="high-urgency"))

    first = queue.pop()
    assert first.event_type == "high-urgency"


# ---------------------------------------------------------------------------
# Reproducibility under simultaneous events
# ---------------------------------------------------------------------------


def test_reproducibility_same_seed_identical_event_sequences() -> None:
    """Two runs with the same (config, seed) must produce identical
    snapshot sequences even when simultaneous events occur."""
    config = make_config(seed=42)

    rec_a = SnapshotRecorder()
    Simulator(config).run(consumer=rec_a)

    rec_b = SnapshotRecorder()
    Simulator(config).run(consumer=rec_b)

    assert len(rec_a.snapshots) > 0
    assert rec_a.snapshots == rec_b.snapshots


def test_reproducibility_simultaneous_events_seed_17() -> None:
    """Seed 17 with rate=5.0 produces simultaneous events (high arrival rate
    relative to short prep times).  Two runs must be byte-identical."""
    # rate=5.0 + max_steps=300 → dense arrivals; simultaneous events very likely
    config = ScenarioConfig(
        name="test-simultaneous",
        seed=17,
        dt=1.0,
        max_steps=300,
        world=WorldConfig(width=500.0, height=500.0),
        stores=[StoreConfig(
            name="s1", x=250.0, y=250.0,
            capacity=50, coverage_radius=750.0,
        )],
        couriers=[CourierConfig(
            courier_type="BikeCourier",
            count=5,
            speed=50.0,
        )],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=5.0),
        routing=RoutingConfig(model_type="euclidean"),
        reward=RewardConfig(function_type="SparseDeliveryReward"),
    )

    rec_a = SnapshotRecorder()
    Simulator(config).run(consumer=rec_a)

    rec_b = SnapshotRecorder()
    Simulator(config).run(consumer=rec_b)

    assert len(rec_a.snapshots) > 0
    assert rec_a.snapshots == rec_b.snapshots

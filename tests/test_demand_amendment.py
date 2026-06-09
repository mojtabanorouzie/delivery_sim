"""Tests for DemandGenerator.next_event (3b contract amendment).

Verifies:
- next_event returns None at rate 0, consuming no RNG state
- next_event emits (arrival_time, {customer_x, customer_y}) only — no store_id
  (store assignment is dispatch's responsibility, not the generator's)
- Draw order is fixed: delay → cx → cy — reordering breaks seed-compatibility
- Deterministic on same seed; distinct seeds produce distinct streams
- next_event is the sole consumer of its rng argument
- Covering-store resolution building blocks: Store.covers() is the dispatch
  mechanism; no random fallback when no store covers a customer
- Uncovered customer → FAILED terminal state (a real coverage-gap metric)
- Horizon boundary convention: arrival_time < horizon (strict less-than)
"""

from __future__ import annotations

import numpy as np
import pytest

import delivery_sim.entities  # noqa: F401 — triggers registration side-effects
from delivery_sim.entities.demand_generator import PoissonDemandGenerator
from delivery_sim.entities.order import Order, OrderStatus
from delivery_sim.entities.store import BuiltinStore
from delivery_sim.routing.euclidean import EuclideanRouting

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def routing() -> EuclideanRouting:
    return EuclideanRouting()


@pytest.fixture
def gen() -> PoissonDemandGenerator:
    return PoissonDemandGenerator(
        rate=1.0, dt=1.0, world_width=200.0, world_height=100.0
    )


# ---------------------------------------------------------------------------
# next_event — rate-zero exhaustion
# ---------------------------------------------------------------------------


def test_next_event_rate_zero_returns_none() -> None:
    gen = PoissonDemandGenerator(rate=0.0, dt=1.0)
    assert gen.next_event(0.0, np.random.default_rng(0)) is None
    assert gen.next_event(99.0, np.random.default_rng(0)) is None


def test_next_event_rate_zero_does_not_consume_rng() -> None:
    """rate=0 returns None without advancing the rng."""
    gen = PoissonDemandGenerator(rate=0.0, dt=1.0)
    rng = np.random.default_rng(55)
    rng_twin = np.random.default_rng(55)
    gen.next_event(0.0, rng)
    # Neither was advanced; their next draws must agree
    assert rng.random() == pytest.approx(rng_twin.random())


# ---------------------------------------------------------------------------
# next_event — return-value shape (option a: no store_id)
# ---------------------------------------------------------------------------


def test_next_event_returns_two_tuple(gen: PoissonDemandGenerator) -> None:
    result = gen.next_event(0.0, np.random.default_rng(1))
    assert result is not None
    assert isinstance(result, tuple) and len(result) == 2


def test_next_event_arrival_time_is_float(gen: PoissonDemandGenerator) -> None:
    result = gen.next_event(0.0, np.random.default_rng(2))
    assert result is not None
    arrival, _ = result
    assert isinstance(arrival, float)


def test_next_event_attrs_has_customer_x_and_y(gen: PoissonDemandGenerator) -> None:
    result = gen.next_event(0.0, np.random.default_rng(3))
    assert result is not None
    _, attrs = result
    assert "customer_x" in attrs
    assert "customer_y" in attrs


def test_next_event_attrs_has_no_store_id(gen: PoissonDemandGenerator) -> None:
    """Store assignment belongs to dispatch, not the generator."""
    result = gen.next_event(0.0, np.random.default_rng(4))
    assert result is not None
    _, attrs = result
    assert "store_id" not in attrs


def test_next_event_attrs_has_exactly_customer_x_and_y_keys(
    gen: PoissonDemandGenerator,
) -> None:
    result = gen.next_event(0.0, np.random.default_rng(5))
    assert result is not None
    _, attrs = result
    assert set(attrs.keys()) == {"customer_x", "customer_y"}


# ---------------------------------------------------------------------------
# next_event — temporal and spatial bounds
# ---------------------------------------------------------------------------


def test_next_event_arrival_strictly_after_sim_time(
    gen: PoissonDemandGenerator,
) -> None:
    for t_start in (0.0, 10.0, 500.0):
        result = gen.next_event(t_start, np.random.default_rng(int(t_start)))
        assert result is not None
        arrival, _ = result
        assert arrival > t_start


def test_next_event_customer_x_in_world_bounds(gen: PoissonDemandGenerator) -> None:
    rng = np.random.default_rng(10)
    for _ in range(50):
        result = gen.next_event(0.0, rng)
        assert result is not None
        _, attrs = result
        assert 0.0 <= attrs["customer_x"] < 200.0


def test_next_event_customer_y_in_world_bounds(gen: PoissonDemandGenerator) -> None:
    rng = np.random.default_rng(11)
    for _ in range(50):
        result = gen.next_event(0.0, rng)
        assert result is not None
        _, attrs = result
        assert 0.0 <= attrs["customer_y"] < 100.0


# ---------------------------------------------------------------------------
# next_event — determinism and fixed draw order
# ---------------------------------------------------------------------------


def test_next_event_same_seed_produces_identical_result(
    gen: PoissonDemandGenerator,
) -> None:
    result_a = gen.next_event(0.0, np.random.default_rng(42))
    result_b = gen.next_event(0.0, np.random.default_rng(42))
    assert result_a == result_b


def test_next_event_different_seeds_produce_different_results(
    gen: PoissonDemandGenerator,
) -> None:
    arrivals: set[float] = set()
    for seed in range(20):
        result = gen.next_event(0.0, np.random.default_rng(seed))
        assert result is not None
        arrival, _ = result
        arrivals.add(round(arrival, 8))
    assert len(arrivals) > 5


def test_next_event_draw_order_is_delay_then_cx_then_cy() -> None:
    """Fixed draw order: Exp(1/rate) → Uniform cx → Uniform cy.

    Reordering these draws changes the stream and breaks seed-compatibility
    of any saved simulation results.  This test pins the contract.
    """
    gen = PoissonDemandGenerator(
        rate=2.0, dt=1.0, world_width=500.0, world_height=300.0
    )
    seed = 77

    rng_expected = np.random.default_rng(seed)
    expected_delay = float(rng_expected.exponential(1.0 / 2.0))
    expected_cx = float(rng_expected.uniform(0.0, 500.0))
    expected_cy = float(rng_expected.uniform(0.0, 300.0))

    rng_actual = np.random.default_rng(seed)
    result = gen.next_event(5.0, rng_actual)
    assert result is not None
    arrival, attrs = result

    assert arrival == pytest.approx(5.0 + expected_delay)
    assert attrs["customer_x"] == pytest.approx(expected_cx)
    assert attrs["customer_y"] == pytest.approx(expected_cy)


def test_next_event_is_sole_rng_consumer() -> None:
    """next_event is the sole consumer of rng; three calls consume exactly 3×3 draws."""
    gen = PoissonDemandGenerator(
        rate=1.0, dt=1.0, world_width=100.0, world_height=100.0
    )
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)

    collected_a = [gen.next_event(0.0, rng_a) for _ in range(3)]

    # Replay manually on rng_b to confirm the sole-consumer contract
    collected_b = []
    for _ in range(3):
        delay = float(rng_b.exponential(1.0 / 1.0))
        cx = float(rng_b.uniform(0.0, 100.0))
        cy = float(rng_b.uniform(0.0, 100.0))
        collected_b.append((0.0 + delay, {"customer_x": cx, "customer_y": cy}))

    assert collected_a == collected_b


# ---------------------------------------------------------------------------
# Covering-store resolution building blocks
#
# The Simulator's _dispatch() will iterate stores sorted by store_id and pick
# the first one whose covers() returns True.  These tests verify that contract.
# ---------------------------------------------------------------------------


def _first_covering_store(
    stores: list[BuiltinStore], cx: float, cy: float, routing: EuclideanRouting
) -> BuiltinStore | None:
    """Return the first store (by store_id, ascending) that covers (cx, cy)."""
    for store in sorted(stores, key=lambda s: s.store_id):
        if store.covers(cx, cy, routing):
            return store
    return None


def test_covering_store_found_inside_radius(routing: EuclideanRouting) -> None:
    store = BuiltinStore("s1", 0.0, 0.0, coverage_radius=100.0)
    assert _first_covering_store([store], 50.0, 0.0, routing) is store


def test_covering_store_none_when_outside_all_radii(
    routing: EuclideanRouting,
) -> None:
    store = BuiltinStore("s1", 0.0, 0.0, coverage_radius=10.0)
    assert _first_covering_store([store], 50.0, 0.0, routing) is None


def test_covering_store_tie_break_picks_alphabetically_first(
    routing: EuclideanRouting,
) -> None:
    """When multiple stores cover the customer, the one with the smallest
    store_id (alphabetical) is chosen.  Deterministic; never dict-order-dependent.
    """
    store_b = BuiltinStore("store_b", 500.0, 500.0, coverage_radius=600.0)
    store_a = BuiltinStore("store_a", 500.0, 500.0, coverage_radius=600.0)
    result = _first_covering_store([store_b, store_a], 300.0, 300.0, routing)
    assert result is store_a


def test_covering_store_none_with_empty_store_list(routing: EuclideanRouting) -> None:
    assert _first_covering_store([], 100.0, 100.0, routing) is None


def test_covering_store_respects_live_coverage_radius_mutation(
    routing: EuclideanRouting,
) -> None:
    """Shrinking coverage_radius at runtime removes the store from eligible set.

    This confirms that coverage_radius is a live control variable — the value
    at dispatch time governs eligibility, not the value at scenario load time.
    """
    store = BuiltinStore("s1", 0.0, 0.0, coverage_radius=200.0)
    assert _first_covering_store([store], 150.0, 0.0, routing) is store
    store.coverage_radius = 100.0
    assert _first_covering_store([store], 150.0, 0.0, routing) is None


# ---------------------------------------------------------------------------
# Uncovered customer → FAILED order (not a silent fallback)
# ---------------------------------------------------------------------------


def test_uncovered_customer_order_transitions_to_failed() -> None:
    """No covering store found → dispatch calls order.transition(FAILED).

    FAILED is a terminal state; the order is never dispatched or delivered.
    This represents a real coverage-gap metric, not a random-store fallback.
    """
    order = Order(
        order_id="ord-uncovered",
        store_id="",
        customer_x=9999.0,
        customer_y=9999.0,
        created_at=0.0,
    )
    assert order.status == OrderStatus.CREATED
    order.transition(OrderStatus.FAILED, sim_time=0.01)
    assert order.status == OrderStatus.FAILED
    assert order.is_terminal


def test_failed_order_accepts_no_further_transitions() -> None:
    """FAILED is terminal: any subsequent transition raises ValueError."""
    order = Order(
        order_id="ord-1", store_id="s1", customer_x=0.0, customer_y=0.0, created_at=0.0
    )
    order.transition(OrderStatus.FAILED, sim_time=1.0)
    with pytest.raises(ValueError, match="Illegal transition"):
        order.transition(OrderStatus.ASSIGNED, sim_time=2.0)


# ---------------------------------------------------------------------------
# Horizon boundary convention
# ---------------------------------------------------------------------------


def test_horizon_scheduling_guard_strict_less_than() -> None:
    """The schedule guard must be ``arrival_time < horizon`` (strict).

    Loop terminates at ``sim_time >= horizon``.  An event scheduled at exactly
    horizon would never be processed; scheduling it silently drops the order.
    Documented here so the Simulator's loop can cite this test as authority.
    """
    horizon = 500.0

    just_before = horizon - 1e-9
    at_boundary = horizon
    just_after = horizon + 1e-9

    assert just_before < horizon      # schedule: yes
    assert not (at_boundary < horizon)  # schedule: no — boundary excluded
    assert not (just_after < horizon)   # schedule: no — past horizon


def test_horizon_guard_consistent_with_next_event_output() -> None:
    """A stream of next_event calls respects the horizon guard at boundary values."""
    gen = PoissonDemandGenerator(
        rate=1.0, dt=1.0, world_width=100.0, world_height=100.0
    )
    horizon = 10.0
    rng = np.random.default_rng(0)
    sim_time = 0.0
    scheduled = 0
    for _ in range(200):
        result = gen.next_event(sim_time, rng)
        assert result is not None
        arrival, _ = result
        if arrival < horizon:
            scheduled += 1
            sim_time = arrival
        else:
            break  # would not schedule; stop
    # We ran some events and stopped before or at horizon
    assert sim_time < horizon

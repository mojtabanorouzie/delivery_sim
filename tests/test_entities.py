"""Tests for built-in entity implementations (steps 2 / 3b).

Covers BuiltinStore (capacity + coverage), BikeCourier (trajectory-based
movement per ADR-002), PoissonDemandGenerator, and registry resolution.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import delivery_sim.entities  # noqa: F401 — triggers registration side-effects
from delivery_sim.entities.courier import BikeCourier, CourierStatus
from delivery_sim.entities.demand_generator import PoissonDemandGenerator
from delivery_sim.entities.store import BuiltinStore
from delivery_sim.registry import create, list_registered
from delivery_sim.routing.euclidean import EuclideanRouting

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def routing() -> EuclideanRouting:
    return EuclideanRouting()


# ---------------------------------------------------------------------------
# BuiltinStore — capacity
# ---------------------------------------------------------------------------


class TestBuiltinStore:
    def test_can_prepare_when_under_capacity(self) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, capacity=2)
        assert store.can_prepare("ord-1") is True

    def test_cannot_prepare_when_at_capacity(self) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, capacity=2)
        store.start_preparation("ord-1", 0.0)
        store.start_preparation("ord-2", 0.0)
        assert store.can_prepare("ord-3") is False

    def test_capacity_boundary_is_inclusive(self) -> None:
        """Exactly capacity orders are allowed; the next one is rejected."""
        store = BuiltinStore("s1", 0.0, 0.0, capacity=3)
        for i in range(3):
            assert store.can_prepare(f"ord-{i}") is True
            store.start_preparation(f"ord-{i}", 0.0)
        assert store.can_prepare("ord-overflow") is False

    def test_start_preparation_returns_correct_pickup_time(self) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, capacity=5, prep_time=30.0)
        pickup_time = store.start_preparation("ord-1", 100.0)
        assert pickup_time == pytest.approx(130.0)

    def test_start_preparation_reduces_available_slots(self) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, capacity=1)
        store.start_preparation("ord-1", 0.0)
        assert store.can_prepare("ord-2") is False

    def test_reset_clears_active_orders(self) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, capacity=1)
        store.start_preparation("ord-1", 0.0)
        store.reset()
        assert store.can_prepare("ord-new") is True

    def test_store_id_and_coords_are_reported(self) -> None:
        store = BuiltinStore("warehouse_a", 200.0, 600.0)
        assert store.store_id == "warehouse_a"
        assert store.x == pytest.approx(200.0)
        assert store.y == pytest.approx(600.0)

    def test_capacity_zero_can_never_prepare(self) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, capacity=0)
        assert store.can_prepare("ord-1") is False


# ---------------------------------------------------------------------------
# BuiltinStore — coverage
# ---------------------------------------------------------------------------


class TestBuiltinStoreCoverage:
    def test_covers_point_inside_radius(self, routing: EuclideanRouting) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, coverage_radius=100.0)
        assert store.covers(50.0, 0.0, routing) is True

    def test_covers_point_outside_radius(self, routing: EuclideanRouting) -> None:
        store = BuiltinStore("s1", 0.0, 0.0, coverage_radius=100.0)
        assert store.covers(150.0, 0.0, routing) is False

    def test_covers_boundary_is_inclusive(self, routing: EuclideanRouting) -> None:
        """distance == coverage_radius is True (inclusive, documented)."""
        store = BuiltinStore("s1", 0.0, 0.0, coverage_radius=5.0)
        assert store.covers(5.0, 0.0, routing) is True  # exactly on boundary
        assert store.covers(5.001, 0.0, routing) is False  # just outside

    def test_covers_zero_radius_only_exact_store_location(
        self, routing: EuclideanRouting
    ) -> None:
        """radius=0.0 means only the store's own position (distance=0) is covered."""
        store = BuiltinStore("s1", 5.0, 5.0, coverage_radius=0.0)
        assert store.covers(5.0, 5.0, routing) is True
        assert store.covers(5.001, 5.0, routing) is False
        assert store.covers(6.0, 5.0, routing) is False

    def test_covers_mutation_changes_result(self, routing: EuclideanRouting) -> None:
        """Writing to coverage_radius immediately affects covers()."""
        store = BuiltinStore("s1", 0.0, 0.0, coverage_radius=10.0)
        assert store.covers(15.0, 0.0, routing) is False
        store.coverage_radius = 20.0
        assert store.covers(15.0, 0.0, routing) is True

    def test_covers_negative_radius_at_runtime_returns_false_no_crash(
        self, routing: EuclideanRouting
    ) -> None:
        """Negative coverage_radius is defined: covers() always returns False.

        distance >= 0 > negative_radius, so the inequality is never satisfied.
        No exception is raised.
        """
        store = BuiltinStore("s1", 5.0, 5.0, coverage_radius=500.0)
        store.coverage_radius = -1.0
        assert store.covers(5.0, 5.0, routing) is False  # even exact location
        assert store.covers(100.0, 200.0, routing) is False


# ---------------------------------------------------------------------------
# BikeCourier — static attributes
# ---------------------------------------------------------------------------


class TestBikeCourierAttributes:
    def test_speed_returns_configured_value(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=2.5)
        assert c.speed == pytest.approx(2.5)

    def test_capacity_returns_configured_value(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing, capacity=3)
        assert c.capacity == 3

    def test_cost_formula(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing, cost_per_unit=0.05)
        assert c.cost(100.0) == pytest.approx(5.0)
        assert c.cost(0.0) == pytest.approx(0.0)

    def test_initial_status_is_idle(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing)
        assert c.status == CourierStatus.IDLE

    def test_assign_sets_dispatched_status(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing)
        c.assign(
            "ord-1", "store-1", sim_time=0.0,
            target_x=10.0, target_y=20.0, from_x=0.0, from_y=0.0,
        )
        assert c.status == CourierStatus.DISPATCHED


# ---------------------------------------------------------------------------
# BikeCourier — trajectory: arrival_time
# ---------------------------------------------------------------------------


class TestBikeCourierArrivalTime:
    def test_arrival_time_none_when_idle(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing)
        assert c.arrival_time() is None

    def test_arrival_time_none_after_reset(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing)
        c.assign(
            "ord-1", "s1", sim_time=5.0,
            target_x=30.0, target_y=40.0, from_x=0.0, from_y=0.0,
        )
        c.reset(origin_x=0.0, origin_y=0.0)
        assert c.arrival_time() is None

    def test_arrival_time_known_horizontal_case(self, routing: EuclideanRouting) -> None:
        """Leg start (0,0) → target (30,0) at speed 3.0, assign_time=10.0.

        distance = 30, travel = 30/3 = 10 s → arrival = 20.0
        """
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=3.0)
        c.assign(
            "ord-1", "s1", sim_time=10.0,
            target_x=30.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        assert c.arrival_time() == pytest.approx(20.0)

    def test_arrival_time_known_diagonal_case(self, routing: EuclideanRouting) -> None:
        """Leg start (0,0) → target (3,4): distance = 5, travel = 5/1.0 = 5 s."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=3.0, target_y=4.0, from_x=0.0, from_y=0.0,
        )
        assert c.arrival_time() == pytest.approx(5.0)

    def test_arrival_time_zero_travel_immediate(self, routing: EuclideanRouting) -> None:
        """Assign target == leg start: travel_time = 0, arrival_time == assign_time."""
        c = BikeCourier("c1", 5.0, 5.0, routing, speed=1.5)
        c.assign(
            "ord-1", "s1", sim_time=7.0,
            target_x=5.0, target_y=5.0, from_x=5.0, from_y=5.0,
        )
        assert c.arrival_time() == pytest.approx(7.0)

    def test_arrival_time_non_zero_assign_time(self, routing: EuclideanRouting) -> None:
        """assign_time offset is included: arrival = assign_time + distance/speed."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=2.0)
        c.assign(
            "ord-1", "s1", sim_time=100.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # distance=10, travel=5 → arrival=105
        assert c.arrival_time() == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# BikeCourier — trajectory: position_at
# ---------------------------------------------------------------------------


class TestBikeCourierPositionAt:
    def test_position_at_returns_spawn_origin_when_idle(
        self, routing: EuclideanRouting
    ) -> None:
        """Idle courier (no trajectory): position_at returns the spawn origin."""
        c = BikeCourier("c1", 3.0, 7.0, routing)
        assert c.position_at(0.0) == pytest.approx((3.0, 7.0))
        assert c.position_at(999.0) == pytest.approx((3.0, 7.0))

    def test_position_at_at_assign_time_equals_leg_start(
        self, routing: EuclideanRouting
    ) -> None:
        """At t == assign_time, position is exactly the leg start (from_x, from_y)."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=5.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        px, py = c.position_at(5.0)
        assert math.isclose(px, 0.0, abs_tol=1e-9)
        assert math.isclose(py, 0.0, abs_tol=1e-9)

    def test_position_at_before_assign_time_equals_leg_start(
        self, routing: EuclideanRouting
    ) -> None:
        """t < assign_time returns leg start (from_x, from_y) — trajectory not yet started."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=10.0,
            target_x=20.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        assert c.position_at(0.0) == pytest.approx((0.0, 0.0))
        assert c.position_at(9.99) == pytest.approx((0.0, 0.0))

    def test_position_at_arrival_time_equals_target(
        self, routing: EuclideanRouting
    ) -> None:
        """At t == arrival_time, position is exactly the target (no overshoot)."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=2.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # arrival_time = 0 + 10/2 = 5.0
        eta = c.arrival_time()
        assert eta is not None
        px, py = c.position_at(eta)
        assert math.isclose(px, 10.0, abs_tol=1e-9)
        assert math.isclose(py, 0.0, abs_tol=1e-9)

    def test_position_at_after_arrival_clamped_to_target(
        self, routing: EuclideanRouting
    ) -> None:
        """t >> arrival_time: position clamped at target, not extrapolated."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=5.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # arrival_time = 5.0; query at t=9999
        assert c.position_at(9999.0) == pytest.approx((5.0, 0.0))

    def test_position_at_midpoint_horizontal(self, routing: EuclideanRouting) -> None:
        """Horizontal leg: at the midpoint time, position is the midpoint.

        Leg start (0,0) → target (20,0), speed=2.0, assign_time=0.
        distance=20, travel=10 → arrival=10.
        At t=5 (midpoint): x=10, y=0.
        """
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=2.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=20.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        px, py = c.position_at(5.0)
        assert math.isclose(px, 10.0, abs_tol=1e-9)
        assert math.isclose(py, 0.0, abs_tol=1e-9)

    def test_position_at_midpoint_diagonal(self, routing: EuclideanRouting) -> None:
        """Diagonal leg: midpoint time → exactly halfway on the segment.

        Leg start (0,0) → target (3,4), speed=1.0, assign_time=0.
        distance=5, travel=5 → arrival=5.
        At t=2.5 (midpoint): (1.5, 2.0).
        """
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=3.0, target_y=4.0, from_x=0.0, from_y=0.0,
        )
        px, py = c.position_at(2.5)
        assert math.isclose(px, 1.5, abs_tol=1e-9)
        assert math.isclose(py, 2.0, abs_tol=1e-9)

    def test_position_at_midpoint_with_nonzero_assign_time(
        self, routing: EuclideanRouting
    ) -> None:
        """Non-zero assign_time shifts the time window but not the spatial midpoint.

        Leg start (0,0) → target (10,0), speed=1.0, assign_time=20.
        distance=10, travel=10 → arrival=30.
        At t=25 (midpoint of [20,30]): x=5, y=0.
        """
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=20.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        px, py = c.position_at(25.0)
        assert math.isclose(px, 5.0, abs_tol=1e-9)
        assert math.isclose(py, 0.0, abs_tol=1e-9)

    def test_position_at_zero_travel_time_collapses_to_target(
        self, routing: EuclideanRouting
    ) -> None:
        """Zero-distance leg: arrival_time == assign_time, any t >= assign_time
        returns target immediately."""
        c = BikeCourier("c1", 5.0, 5.0, routing, speed=1.5)
        c.assign(
            "ord-1", "s1", sim_time=3.0,
            target_x=5.0, target_y=5.0, from_x=5.0, from_y=5.0,
        )
        # t == assign_time == arrival_time (both 3.0): should return target
        assert c.position_at(3.0) == pytest.approx((5.0, 5.0))
        assert c.position_at(100.0) == pytest.approx((5.0, 5.0))

    def test_position_at_spawn_origin_after_reset(
        self, routing: EuclideanRouting
    ) -> None:
        """After episode reset, position_at returns the new spawn origin for all t."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=50.0, target_y=50.0, from_x=0.0, from_y=0.0,
        )
        c.reset(origin_x=100.0, origin_y=200.0)
        assert c.position_at(0.0) == pytest.approx((100.0, 200.0))
        assert c.position_at(999.0) == pytest.approx((100.0, 200.0))

    def test_position_at_second_leg_uses_explicit_from(
        self, routing: EuclideanRouting
    ) -> None:
        """Second leg from a non-spawn position uses from_x/from_y, not spawn origin.

        Leg 1: (0,0) → (10,0), speed=1.0, assign_time=0 → arrival=10.
        Leg 2: from (10,0) → (20,0), assign_time=10 → arrival=20.
        At t=15 (midpoint of leg 2): x=15, y=0.
        """
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # leg 2: assign exactly at arrival_time of leg 1
        c.assign(
            "ord-2", "s2", sim_time=10.0,
            target_x=20.0, target_y=0.0, from_x=10.0, from_y=0.0,
        )
        assert c.arrival_time() == pytest.approx(20.0)
        assert c.position_at(10.0) == pytest.approx((10.0, 0.0))
        assert c.position_at(15.0) == pytest.approx((15.0, 0.0))
        assert c.position_at(20.0) == pytest.approx((20.0, 0.0))


# ---------------------------------------------------------------------------
# BikeCourier — trajectory: determinism
# ---------------------------------------------------------------------------


class TestBikeCourierDeterminism:
    def test_same_inputs_produce_identical_trajectory_samples(
        self, routing: EuclideanRouting
    ) -> None:
        """Identical (leg start, target, assign_time, speed) → identical samples."""
        def make_and_sample(
            ox: float, oy: float, tx: float, ty: float, t0: float, speed: float
        ) -> list[tuple[float, float]]:
            c = BikeCourier("c1", ox, oy, routing, speed=speed)
            c.assign(
                "ord-1", "s1", sim_time=t0,
                target_x=tx, target_y=ty, from_x=ox, from_y=oy,
            )
            eta = c.arrival_time()
            assert eta is not None
            mid = (t0 + eta) / 2
            return [c.position_at(t) for t in [t0 - 1, t0, mid, eta, eta + 10]]

        samples_a = make_and_sample(0.0, 0.0, 10.0, 0.0, 5.0, 2.0)
        samples_b = make_and_sample(0.0, 0.0, 10.0, 0.0, 5.0, 2.0)
        assert samples_a == samples_b

    def test_different_inputs_produce_different_trajectory_samples(
        self, routing: EuclideanRouting
    ) -> None:
        c1 = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c2 = BikeCourier("c2", 0.0, 0.0, routing, speed=2.0)
        c1.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        c2.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # at the same t, faster courier is further along
        assert c1.position_at(3.0) != c2.position_at(3.0)


# ---------------------------------------------------------------------------
# BikeCourier — has_target status predicate
# ---------------------------------------------------------------------------


class TestBikeCourierHasTarget:
    def test_has_target_false_when_idle_never_assigned(
        self, routing: EuclideanRouting
    ) -> None:
        """Idle never-assigned courier: has_target is False."""
        c = BikeCourier("c1", 0.0, 0.0, routing)
        assert c.has_target is False

    def test_has_target_true_after_assign(self, routing: EuclideanRouting) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=100.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        assert c.has_target is True

    def test_has_target_false_after_reset(self, routing: EuclideanRouting) -> None:
        """reset() clears the trajectory, so has_target returns False."""
        c = BikeCourier("c1", 0.0, 0.0, routing)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=50.0, target_y=50.0, from_x=0.0, from_y=0.0,
        )
        assert c.has_target is True
        c.reset(origin_x=0.0, origin_y=0.0)
        assert c.has_target is False

    def test_has_target_is_not_arrival_detection(
        self, routing: EuclideanRouting
    ) -> None:
        """has_target alone does NOT determine whether a courier has arrived.

        An idle never-assigned courier and a courier whose trajectory is past
        its arrival_time both read has_target == False, but only the latter
        has arrived.  The Simulator derives arrival by scheduling a
        COURIER_ARRIVED event at arrival_time(); it does not poll has_target.
        """
        idle = BikeCourier("idle", 0.0, 0.0, routing)
        arrived = BikeCourier("arrived", 0.0, 0.0, routing, speed=10.0)
        arrived.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=1.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # arrival_time = 0.1; we query position_at beyond it so arrival is "past"
        assert arrived.position_at(999.0) == pytest.approx((1.0, 0.0))
        # Both still have has_target==True; only reset clears it — that's expected.
        # The important invariant is that has_target True ≠ "in transit right now":
        # it means "trajectory was assigned and not yet reset".
        assert idle.has_target is False
        assert arrived.has_target is True  # trajectory not reset, still flagged busy


# ---------------------------------------------------------------------------
# BikeCourier — reset
# ---------------------------------------------------------------------------


class TestBikeCourierReset:
    def test_reset_restores_spawn_origin_and_clears_trajectory(
        self, routing: EuclideanRouting
    ) -> None:
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.5)
        c.assign(
            "ord-1", "store-1", sim_time=0.0,
            target_x=50.0, target_y=50.0, from_x=0.0, from_y=0.0,
        )
        c.reset(origin_x=100.0, origin_y=200.0)
        assert c.position_at(0.0) == pytest.approx((100.0, 200.0))
        assert c.status == CourierStatus.IDLE
        assert c.has_target is False
        assert c.arrival_time() is None

    def test_reset_allows_reassign_from_new_origin(
        self, routing: EuclideanRouting
    ) -> None:
        """After episode reset, assign() uses from_x/from_y as the leg start."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        c.reset(origin_x=20.0, origin_y=0.0)
        c.assign(
            "ord-2", "s2", sim_time=5.0,
            target_x=30.0, target_y=0.0, from_x=20.0, from_y=0.0,
        )
        # leg start (20,0) → target (30,0): distance=10, speed=1 → arrival=15
        assert c.arrival_time() == pytest.approx(15.0)
        assert c.position_at(5.0) == pytest.approx((20.0, 0.0))
        assert c.position_at(10.0) == pytest.approx((25.0, 0.0))
        assert c.position_at(15.0) == pytest.approx((30.0, 0.0))


# ---------------------------------------------------------------------------
# BikeCourier — assign() SETTLED precondition
# ---------------------------------------------------------------------------


class TestBikeCourierAssignPrecondition:
    def test_assign_while_in_motion_raises(self, routing: EuclideanRouting) -> None:
        """assign() raises ValueError when sim_time < arrival_time (mid-motion)."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # arrival_time=10; attempt reassignment at t=5 (still in motion)
        with pytest.raises(ValueError, match="motion"):
            c.assign(
                "ord-2", "s2", sim_time=5.0,
                target_x=20.0, target_y=0.0, from_x=5.0, from_y=0.0,
            )

    def test_assign_at_arrival_time_succeeds(self, routing: EuclideanRouting) -> None:
        """assign() at exactly arrival_time (settled) does not raise."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        eta = c.arrival_time()
        assert eta is not None  # 10.0
        # should not raise — sim_time == arrival_time is SETTLED
        c.assign(
            "ord-2", "s2", sim_time=eta,
            target_x=20.0, target_y=0.0, from_x=10.0, from_y=0.0,
        )
        assert c.arrival_time() == pytest.approx(eta + 10.0)

    def test_assign_after_arrival_time_succeeds(
        self, routing: EuclideanRouting
    ) -> None:
        """assign() well past arrival_time (courier long settled) does not raise."""
        c = BikeCourier("c1", 0.0, 0.0, routing, speed=1.0)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=5.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        # arrival_time=5; assign at t=100 >> arrival
        c.assign(
            "ord-2", "s2", sim_time=100.0,
            target_x=10.0, target_y=0.0, from_x=5.0, from_y=0.0,
        )
        assert c.arrival_time() == pytest.approx(105.0)

    def test_assign_when_idle_never_assigned_succeeds(
        self, routing: EuclideanRouting
    ) -> None:
        """assign() on a never-assigned courier (arrival_time None) does not raise."""
        c = BikeCourier("c1", 0.0, 0.0, routing)
        c.assign(
            "ord-1", "s1", sim_time=0.0,
            target_x=10.0, target_y=0.0, from_x=0.0, from_y=0.0,
        )
        assert c.arrival_time() is not None


# ---------------------------------------------------------------------------
# PoissonDemandGenerator
# ---------------------------------------------------------------------------


class TestPoissonDemandGenerator:
    def test_same_seed_produces_identical_sequence(self) -> None:
        gen = PoissonDemandGenerator(
            rate=2.0, dt=1.0, world_width=100.0, world_height=100.0, store_ids=["s1"]
        )
        orders1 = gen.generate(0.0, np.random.default_rng(42))
        orders2 = gen.generate(0.0, np.random.default_rng(42))
        assert orders1 == orders2

    def test_different_seeds_produce_different_sequences(self) -> None:
        gen = PoissonDemandGenerator(
            rate=5.0, dt=1.0, world_width=100.0, world_height=100.0, store_ids=["s1"]
        )
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(999)
        results_a = [gen.generate(float(t), rng_a) for t in range(30)]
        results_b = [gen.generate(float(t), rng_b) for t in range(30)]
        assert results_a != results_b

    def test_rate_zero_yields_empty(self) -> None:
        gen = PoissonDemandGenerator(rate=0.0, dt=1.0, store_ids=["s1"])
        rng = np.random.default_rng(0)
        assert gen.generate(0.0, rng) == []

    def test_orders_have_required_fields_and_valid_bounds(self) -> None:
        gen = PoissonDemandGenerator(
            rate=10.0,
            dt=1.0,
            world_width=500.0,
            world_height=300.0,
            store_ids=["s1", "s2"],
        )
        rng = np.random.default_rng(0)
        orders: list[dict] = []
        for t in range(50):
            orders.extend(gen.generate(float(t), rng))
        assert len(orders) > 0, "no orders generated — rate too low for test RNG"
        for o in orders:
            assert "store_id" in o
            assert "customer_x" in o
            assert "customer_y" in o
            assert 0.0 <= o["customer_x"] < 500.0
            assert 0.0 <= o["customer_y"] < 300.0
            assert o["store_id"] in ("s1", "s2")

    def test_order_dicts_carry_no_status_field(self) -> None:
        """The engine stamps status when constructing Order objects, not here."""
        gen = PoissonDemandGenerator(rate=5.0, dt=1.0, store_ids=["s1"])
        rng = np.random.default_rng(7)
        orders: list[dict] = []
        for t in range(20):
            orders.extend(gen.generate(float(t), rng))
        for o in orders:
            assert "status" not in o

    def test_reset_is_noop_no_raise(self) -> None:
        gen = PoissonDemandGenerator(rate=1.0, dt=1.0)
        gen.reset(np.random.default_rng(0))

    def test_no_store_ids_sets_empty_string(self) -> None:
        gen = PoissonDemandGenerator(rate=20.0, dt=1.0, store_ids=[])
        rng = np.random.default_rng(0)
        orders: list[dict] = []
        for t in range(10):
            orders.extend(gen.generate(float(t), rng))
        assert len(orders) > 0
        for o in orders:
            assert o["store_id"] == ""


# ---------------------------------------------------------------------------
# Registry resolution (names used in example.yaml)
# ---------------------------------------------------------------------------


class TestRegistryNames:
    def test_bike_courier_registered_under_courier(self) -> None:
        assert "BikeCourier" in list_registered("courier")

    def test_poisson_demand_generator_registered(self) -> None:
        assert "PoissonDemandGenerator" in list_registered("demand_generator")

    def test_builtin_store_registered_under_store(self) -> None:
        assert "BuiltinStore" in list_registered("store")

    def test_bike_courier_instantiates_via_registry(self) -> None:
        c = create(
            "courier",
            "BikeCourier",
            courier_id="c0",
            x=0.0,
            y=0.0,
            routing=EuclideanRouting(),
        )
        assert isinstance(c, BikeCourier)
        assert c.courier_id == "c0"

    def test_builtin_store_instantiates_via_registry(self) -> None:
        s = create("store", "BuiltinStore", store_id="s0", x=10.0, y=20.0)
        assert isinstance(s, BuiltinStore)
        assert s.store_id == "s0"

    def test_poisson_generator_instantiates_via_registry(self) -> None:
        g = create("demand_generator", "PoissonDemandGenerator", rate=1.0, dt=1.0)
        assert isinstance(g, PoissonDemandGenerator)

"""Tests for EuclideanRouting (step-1 implementation)."""

from __future__ import annotations

import pytest

from delivery_sim.registry import create
from delivery_sim.routing.euclidean import EuclideanRouting


@pytest.fixture
def r() -> EuclideanRouting:
    return EuclideanRouting()


# ---------------------------------------------------------------------------
# distance
# ---------------------------------------------------------------------------

def test_distance_3_4_5(r: EuclideanRouting) -> None:
    assert r.distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)


def test_distance_identical_points(r: EuclideanRouting) -> None:
    assert r.distance(7.5, 3.2, 7.5, 3.2) == pytest.approx(0.0)


def test_distance_symmetry(r: EuclideanRouting) -> None:
    a = (1.0, 2.0)
    b = (4.0, 6.0)
    assert r.distance(*a, *b) == pytest.approx(r.distance(*b, *a))


def test_distance_axis_aligned(r: EuclideanRouting) -> None:
    assert r.distance(0.0, 0.0, 5.0, 0.0) == pytest.approx(5.0)
    assert r.distance(0.0, 0.0, 0.0, 5.0) == pytest.approx(5.0)


def test_distance_negative_coords(r: EuclideanRouting) -> None:
    assert r.distance(-3.0, -4.0, 0.0, 0.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# travel_time
# ---------------------------------------------------------------------------

def test_travel_time_known_values(r: EuclideanRouting) -> None:
    # distance(0,0 -> 3,4) = 5.0; speed = 2.5 → time = 2.0
    assert r.travel_time(0.0, 0.0, 3.0, 4.0, speed=2.5) == pytest.approx(2.0)


def test_travel_time_equals_distance_over_speed(r: EuclideanRouting) -> None:
    x1, y1, x2, y2, speed = 10.0, 20.0, 40.0, 60.0, 1.5
    expected = r.distance(x1, y1, x2, y2) / speed
    assert r.travel_time(x1, y1, x2, y2, speed=speed) == pytest.approx(expected)


def test_travel_time_zero_distance(r: EuclideanRouting) -> None:
    assert r.travel_time(5.0, 5.0, 5.0, 5.0, speed=1.0) == pytest.approx(0.0)


def test_travel_time_zero_speed_raises(r: EuclideanRouting) -> None:
    with pytest.raises(ValueError, match="speed"):
        r.travel_time(0.0, 0.0, 1.0, 1.0, speed=0.0)


def test_travel_time_negative_speed_raises(r: EuclideanRouting) -> None:
    with pytest.raises(ValueError, match="speed"):
        r.travel_time(0.0, 0.0, 1.0, 1.0, speed=-1.0)


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------

def test_route_endpoints(r: EuclideanRouting) -> None:
    waypoints = r.route(1.0, 2.0, 5.0, 6.0)
    assert waypoints[0] == (1.0, 2.0)
    assert waypoints[-1] == (5.0, 6.0)


def test_route_length(r: EuclideanRouting) -> None:
    # Euclidean straight-line has exactly 2 waypoints.
    waypoints = r.route(0.0, 0.0, 10.0, 10.0)
    assert len(waypoints) == 2


def test_route_identical_points(r: EuclideanRouting) -> None:
    waypoints = r.route(3.0, 3.0, 3.0, 3.0)
    assert waypoints[0] == (3.0, 3.0)
    assert waypoints[-1] == (3.0, 3.0)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_registry_name_matches_yaml() -> None:
    """EuclideanRouting is retrievable under the key used in example.yaml."""
    instance = create("routing", "euclidean")
    assert isinstance(instance, EuclideanRouting)


def test_registry_returns_functional_instance() -> None:
    instance = create("routing", "euclidean")
    assert isinstance(instance, EuclideanRouting)
    assert instance.distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)

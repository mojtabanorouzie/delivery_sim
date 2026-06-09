"""
Tests for RewardFunction ABC and SparseDeliveryReward placeholder (Step 4).

Test inventory
--------------
1. SparseDeliveryReward.compute returns correct value for known inputs.
2. Reward for zero completed_orders is 0.0.
3. Reward for mixed delivered + failed matches formula.
4. reset() is a no-op (stateless; same result before and after).
5. Registry: retrievable by name "SparseDeliveryReward".
6. Registered class satisfies RewardFunction ABC.
7. compute signature matches ABC (world, completed_orders, dt).
"""

from __future__ import annotations

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
from delivery_sim.engine.simulator import Simulator
from delivery_sim.engine.world_state import WorldState
from delivery_sim.entities.order import Order, OrderStatus
from delivery_sim.registry import create
from delivery_sim.rewards.base import RewardFunction
from delivery_sim.rewards.placeholder import SparseDeliveryReward

# ---------------------------------------------------------------------------
# Helpers — build minimal Order objects for unit tests
# ---------------------------------------------------------------------------


def _delivered_order(order_id: str = "o1") -> Order:
    o = Order(
        order_id=order_id,
        store_id="s1",
        customer_x=100.0,
        customer_y=100.0,
        created_at=0.0,
    )
    o.transition(OrderStatus.ASSIGNED, 1.0)
    o.transition(OrderStatus.PREPARING, 2.0)
    o.transition(OrderStatus.PICKED_UP, 3.0)
    o.transition(OrderStatus.IN_TRANSIT, 3.0)
    o.transition(OrderStatus.DELIVERED, 5.0)
    return o


def _failed_order(order_id: str = "o2") -> Order:
    o = Order(
        order_id=order_id,
        store_id="",
        customer_x=0.0,
        customer_y=0.0,
        created_at=0.0,
    )
    o.transition(OrderStatus.FAILED, 0.0)
    return o


def _make_world() -> WorldState:
    return WorldState(width=1000.0, height=1000.0)


# ---------------------------------------------------------------------------
# Test 1 — compute: correct value for known inputs
# ---------------------------------------------------------------------------


def test_compute_single_delivery() -> None:
    """One delivered order → reward = DELIVERY_REWARD."""
    reward_fn = SparseDeliveryReward()
    r = reward_fn.compute(
        world=_make_world(),
        completed_orders=[_delivered_order()],
        dt=1.0,
    )
    assert r == SparseDeliveryReward.DELIVERY_REWARD


def test_compute_single_failure() -> None:
    """One failed order → reward = -FAILURE_PENALTY."""
    reward_fn = SparseDeliveryReward()
    r = reward_fn.compute(
        world=_make_world(),
        completed_orders=[_failed_order()],
        dt=1.0,
    )
    assert r == -SparseDeliveryReward.FAILURE_PENALTY


def test_compute_empty_completed_orders() -> None:
    """Empty completed_orders → reward = 0.0."""
    reward_fn = SparseDeliveryReward()
    r = reward_fn.compute(
        world=_make_world(),
        completed_orders=[],
        dt=1.0,
    )
    assert r == 0.0


# ---------------------------------------------------------------------------
# Test 2 — compute: mixed delivered + failed
# ---------------------------------------------------------------------------


def test_compute_mixed_delivers_and_failures() -> None:
    """2 delivered + 3 failed → 2×1.0 − 3×0.5 = 0.5."""
    reward_fn = SparseDeliveryReward()
    completed = [
        _delivered_order("d1"),
        _delivered_order("d2"),
        _failed_order("f1"),
        _failed_order("f2"),
        _failed_order("f3"),
    ]
    r = reward_fn.compute(world=_make_world(), completed_orders=completed, dt=1.0)
    expected = 2 * SparseDeliveryReward.DELIVERY_REWARD - 3 * SparseDeliveryReward.FAILURE_PENALTY
    assert r == expected


def test_compute_all_delivered() -> None:
    """N delivered, 0 failed → reward = N × DELIVERY_REWARD."""
    reward_fn = SparseDeliveryReward()
    n = 5
    completed = [_delivered_order(f"d{i}") for i in range(n)]
    r = reward_fn.compute(world=_make_world(), completed_orders=completed, dt=1.0)
    assert r == n * SparseDeliveryReward.DELIVERY_REWARD


def test_compute_all_failed() -> None:
    """0 delivered, N failed → reward = -N × FAILURE_PENALTY."""
    reward_fn = SparseDeliveryReward()
    n = 4
    completed = [_failed_order(f"f{i}") for i in range(n)]
    r = reward_fn.compute(world=_make_world(), completed_orders=completed, dt=1.0)
    assert r == -n * SparseDeliveryReward.FAILURE_PENALTY


# ---------------------------------------------------------------------------
# Test 3 — reset() is a no-op (stateless)
# ---------------------------------------------------------------------------


def test_reset_does_not_change_behaviour() -> None:
    """reset() on a stateless reward function must not raise and must leave
    compute() producing the same result before and after."""
    reward_fn = SparseDeliveryReward()
    completed = [_delivered_order()]

    r_before = reward_fn.compute(
        world=_make_world(), completed_orders=completed, dt=1.0
    )
    reward_fn.reset()
    r_after = reward_fn.compute(
        world=_make_world(), completed_orders=completed, dt=1.0
    )
    assert r_before == r_after


# ---------------------------------------------------------------------------
# Test 4 — Registry retrieval
# ---------------------------------------------------------------------------


def test_reward_registered_by_name() -> None:
    """SparseDeliveryReward must be retrievable from the registry."""
    instance = create("reward", "SparseDeliveryReward")
    assert isinstance(instance, SparseDeliveryReward)


def test_reward_satisfies_abc() -> None:
    """SparseDeliveryReward must satisfy the RewardFunction ABC."""
    instance = create("reward", "SparseDeliveryReward")
    assert isinstance(instance, RewardFunction)


# ---------------------------------------------------------------------------
# Test 5 — dt does not change reward value
# ---------------------------------------------------------------------------


def test_compute_reward_is_dt_independent() -> None:
    """Reward value must not change when dt changes (baseline is count-based)."""
    reward_fn = SparseDeliveryReward()
    completed = [_delivered_order("d1"), _failed_order("f1")]

    r_dt1 = reward_fn.compute(world=_make_world(), completed_orders=completed, dt=1.0)
    r_dt10 = reward_fn.compute(world=_make_world(), completed_orders=completed, dt=10.0)
    assert r_dt1 == r_dt10


# ---------------------------------------------------------------------------
# Test 6 — Integration: reward on a real episode
# ---------------------------------------------------------------------------


def test_reward_positive_on_successful_episode() -> None:
    """A run with mostly successful deliveries should produce positive total reward."""
    config = ScenarioConfig(
        name="reward_test",
        seed=42,
        dt=1.0,
        max_steps=200,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[StoreConfig(
            name="s1", x=0.0, y=0.0,
            capacity=20, coverage_radius=1500.0,
        )],
        couriers=[CourierConfig(
            courier_type="BikeCourier", count=3, speed=15.0,
        )],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=1.5),
        routing=RoutingConfig(model_type="euclidean"),
    )
    sim = Simulator(config)
    sim.run()
    assert sim.world is not None

    reward_fn = SparseDeliveryReward()
    delivered = [
        o for o in sim.world.active_orders.values()
        if o.status == OrderStatus.DELIVERED
    ]
    failed = [
        o for o in sim.world.active_orders.values()
        if o.status == OrderStatus.FAILED
    ]
    total = reward_fn.compute(
        world=sim.world,
        completed_orders=delivered + failed,
        dt=config.dt,
    )
    assert total > 0.0, "expected mostly-positive reward with successful deliveries"

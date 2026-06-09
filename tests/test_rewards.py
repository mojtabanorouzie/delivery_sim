"""
Tests for RewardFunction ABC and all built-in reward implementations.

Test inventory (original)
--------------------------
1. SparseDeliveryReward.compute returns correct value for known inputs.
2. Reward for zero completed_orders is 0.0.
3. Reward for mixed delivered + failed matches formula.
4. reset() is a no-op (stateless; same result before and after).
5. Registry: retrievable by name "SparseDeliveryReward".
6. Registered class satisfies RewardFunction ABC.
7. compute signature matches ABC (world, completed_orders, dt).

Test inventory (LatencyAwareReward + OptimizedDeliveryReward)
--------------------------------------------------------------
8.  Proxy-gone: reward == 0.0 with no completions, regardless of coverage_radius.
9.  Proxy-gone: same completed orders → same reward regardless of world.stores radii.
10. Proxy-gone: end-to-end env with rate=0 yields 0.0 every step.
11. Real-signal: higher order.delivery_cost → strictly lower reward.
12. Real-signal: faster delivery time → higher latency-value → higher reward.
13. Real-signal: delivery past grace period earns zero latency value.
14. Real-signal: failure penalty fires correctly.
15. Exact known-input values for both rewards.
16. Σ(order.delivery_cost) in reward matches KPI total_delivery_cost end-to-end.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

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
from delivery_sim.engine.simulator import Simulator
from delivery_sim.engine.world_state import WorldState
from delivery_sim.entities.order import Order, OrderStatus
from delivery_sim.envs.single_agent import DeliveryEnv
from delivery_sim.registry import create
from delivery_sim.registry import register as _reg
from delivery_sim.rewards.base import RewardFunction
from delivery_sim.rewards.latency_reward import LatencyAwareReward
from delivery_sim.rewards.optimized_reward import OptimizedDeliveryReward
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


# ---------------------------------------------------------------------------
# Helpers for new reward tests
# ---------------------------------------------------------------------------

def _make_delivered(delivery_time: float, delivery_cost: float) -> Order:
    """Craft a DELIVERED Order with known elapsed time and real cost."""
    order = Order(
        order_id="o", store_id="s",
        customer_x=0.0, customer_y=0.0,
        created_at=0.0,
    )
    order.timestamps[OrderStatus.CREATED] = 0.0
    order.timestamps[OrderStatus.DELIVERED] = delivery_time
    order.status = OrderStatus.DELIVERED
    order.delivery_cost = delivery_cost
    return order


def _make_failed() -> Order:
    order = Order(
        order_id="f", store_id="s",
        customer_x=0.0, customer_y=0.0,
        created_at=0.0,
    )
    order.status = OrderStatus.FAILED
    order.delivery_cost = 0.0
    return order


def _env_config(
    *,
    seed: int = 1,
    rate: float = 2.0,
    coverage_radius: float = 800.0,
    max_steps: int = 300,
    reward_type: str = "SparseDeliveryReward",
) -> ScenarioConfig:
    return ScenarioConfig(
        name="t",
        seed=seed,
        dt=1.0,
        max_steps=max_steps,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[StoreConfig(name="s1", x=500.0, y=500.0, capacity=50,
                            coverage_radius=coverage_radius)],
        couriers=[CourierConfig(courier_type="BikeCourier", count=4, speed=10.0)],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=rate),
        routing=RoutingConfig(model_type="euclidean"),
        reward=RewardConfig(function_type=reward_type),
        decision_interval=100.0,
        max_coverage_radius=2000.0,
    )


def _run_env(config: ScenarioConfig) -> tuple[list[float], dict[str, Any]]:
    env = DeliveryEnv(config)
    env.reset(seed=config.seed)
    act = np.array([sc.coverage_radius for sc in config.stores], dtype=np.float32)
    rewards: list[float] = []
    final_info: dict[str, Any] = {}
    for _ in range(10_000):
        _, rew, _, trunc, info = env.step(act)
        rewards.append(float(rew))
        if trunc:
            final_info = info
            break
    return rewards, final_info


# ---------------------------------------------------------------------------
# Test 8-10 — Proxy gone (parametrized over both fixed rewards)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("RewardCls", [LatencyAwareReward, OptimizedDeliveryReward])
class TestProxyGone:
    """The action-feeds-reward antipattern is absent from both fixed rewards."""

    def test_empty_step_zero_reward_any_radius(self, RewardCls: type) -> None:
        """compute(world, [], dt) == 0.0 for any coverage_radius.

        Old code returned negative coverage_cost×Σradius every step even with
        no orders.  Fixed code must return 0.0 when no orders complete.
        """
        fn = RewardCls()

        class _Store:
            def __init__(self, r: float) -> None:
                self.coverage_radius = r

        class _World:
            def __init__(self, radii: list[float]) -> None:
                self.stores = [_Store(r) for r in radii]

        for radii in ([0.0], [500.0], [2000.0], [100.0, 900.0]):
            result = fn.compute(_World(radii), [], 1.0)  # type: ignore[arg-type]
            assert result == pytest.approx(0.0), (
                f"{RewardCls.__name__}: got {result!r} with no orders, "
                f"radii={radii!r} — proxy not eliminated"
            )

    def test_same_orders_same_reward_different_radii(self, RewardCls: type) -> None:
        """Same completed orders → same reward regardless of world.stores radii.

        If the proxy were present, the large-radius world would incur a bigger
        penalty than the small-radius world despite identical order outcomes.
        """
        fn = RewardCls()
        orders = [_make_delivered(300.0, 5.0)]

        class _World:
            def __init__(self, r: float) -> None:
                self.stores = [type("S", (), {"coverage_radius": r})()]

        r_small = fn.compute(_World(10.0), orders, 1.0)    # type: ignore[arg-type]
        r_large = fn.compute(_World(2000.0), orders, 1.0)  # type: ignore[arg-type]
        assert r_small == pytest.approx(r_large), (
            f"{RewardCls.__name__}: reward changed with coverage_radius "
            f"(small={r_small!r}, large={r_large!r}) — proxy not eliminated"
        )

    def test_zero_demand_env_every_step_reward_zero(self, RewardCls: type) -> None:
        """End-to-end: rate=0 (no orders) → every step reward must be 0.0.

        Old code would return −coverage_cost×radius each step even without
        any demand.  Fixed code returns 0.0 because no orders complete.
        """
        config = _env_config(rate=0.0, max_steps=300, reward_type=RewardCls.__name__)
        rewards, _ = _run_env(config)
        for i, r in enumerate(rewards):
            assert r == pytest.approx(0.0), (
                f"{RewardCls.__name__}: step {i} reward={r!r} with zero demand; "
                "must be 0.0 — no completions means no cost term fires"
            )


# ---------------------------------------------------------------------------
# Test 11-16 — Real signal present (parametrized over both fixed rewards)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("RewardCls", [LatencyAwareReward, OptimizedDeliveryReward])
class TestRealSignal:

    def test_higher_delivery_cost_lower_reward(self, RewardCls: type) -> None:
        """Increasing order.delivery_cost must strictly decrease the reward."""
        fn = RewardCls(w_cost=0.01)
        for low_c, high_c in [(1.0, 10.0), (5.0, 50.0), (0.0, 100.0)]:
            r_low = fn.compute(None, [_make_delivered(300.0, low_c)], 1.0)   # type: ignore[arg-type]
            r_high = fn.compute(None, [_make_delivered(300.0, high_c)], 1.0) # type: ignore[arg-type]
            assert r_low > r_high, (
                f"{RewardCls.__name__}: cost {low_c}→{r_low!r}; "
                f"cost {high_c}→{r_high!r}; higher cost must give lower reward"
            )

    def test_faster_delivery_higher_reward(self, RewardCls: type) -> None:
        """Faster delivery → higher latency value → higher reward."""
        fn = RewardCls()
        fast = _make_delivered(fn.target_time * 0.5, delivery_cost=0.0)
        slow = _make_delivered(fn.target_time + fn.grace_period * 0.5, delivery_cost=0.0)
        r_fast = fn.compute(None, [fast], 1.0)  # type: ignore[arg-type]
        r_slow = fn.compute(None, [slow], 1.0)  # type: ignore[arg-type]
        assert r_fast > r_slow, (
            f"{RewardCls.__name__}: fast ({r_fast!r}) must beat slow ({r_slow!r})"
        )

    def test_past_grace_period_earns_zero_latency_value(self, RewardCls: type) -> None:
        """Delivery beyond target_time + grace_period → latency value clamped to 0."""
        fn = RewardCls(w_cost=0.0)
        very_late = _make_delivered(
            fn.target_time + fn.grace_period + 1.0, delivery_cost=0.0
        )
        assert fn.compute(None, [very_late], 1.0) == pytest.approx(0.0)  # type: ignore[arg-type]

    def test_failure_penalty_fires(self, RewardCls: type) -> None:
        fn = RewardCls(failure_penalty=1.5)
        assert fn.compute(None, [_make_failed()], 1.0) == pytest.approx(-1.5)  # type: ignore[arg-type]

    def test_zero_cost_no_cost_penalty(self, RewardCls: type) -> None:
        """delivery_cost==0 → no cost deduction, only latency value."""
        fn = RewardCls(w_cost=0.01)
        order = _make_delivered(100.0, 0.0)
        result = fn.compute(None, [order], 1.0)  # type: ignore[arg-type]
        overshoot = max(0.0, 100.0 - fn.target_time)
        expected = max(0.0, 1.0 - overshoot / fn.grace_period)
        assert result == pytest.approx(expected)

    def test_reward_cost_matches_kpi_total_delivery_cost(
        self, RewardCls: type
    ) -> None:
        """Σ(order.delivery_cost) fed into reward must equal KPI total_delivery_cost."""
        class_name = RewardCls.__name__
        tracked: list[float] = []

        orig_cls = RewardCls

        class _Tracker(orig_cls):  # type: ignore[valid-type]
            def compute(self, world, completed_orders, dt):  # type: ignore[override]
                tracked.append(sum(
                    o.delivery_cost for o in completed_orders
                    if o.status == OrderStatus.DELIVERED
                ))
                return super().compute(world, completed_orders, dt)

        tracker_name = f"_tracker_{class_name}_v2"
        _reg("reward", name=tracker_name)(_Tracker)

        config = _env_config(rate=2.0, max_steps=300, coverage_radius=800.0,
                             reward_type=tracker_name)
        _, final_info = _run_env(config)
        kpi_cost = float(final_info["kpi"]["total_delivery_cost"])
        reward_cost = sum(tracked)
        assert reward_cost == pytest.approx(kpi_cost, rel=1e-6), (
            f"{class_name}: reward Σcost={reward_cost!r} ≠ "
            f"KPI total_delivery_cost={kpi_cost!r}"
        )


# ---------------------------------------------------------------------------
# Exact known-input tests per reward class
# ---------------------------------------------------------------------------

class TestLatencyAwareRewardKnownInputs:

    def test_all_within_target_with_cost(self) -> None:
        """2 fast deliveries + 1 fail; expected = 1+1 − 0.5 − 0.01×30 = 1.2."""
        fn = LatencyAwareReward(target_time=600.0, grace_period=600.0,
                                failure_penalty=0.5, w_cost=0.01)
        orders = [
            _make_delivered(300.0, 10.0),
            _make_delivered(400.0, 20.0),
            _make_failed(),
        ]
        assert fn.compute(None, orders, 1.0) == pytest.approx(1.2)  # type: ignore[arg-type]

    def test_past_grace_only_cost_penalty(self) -> None:
        """Delivery 100s past grace → value=0; only cost penalty −0.5."""
        fn = LatencyAwareReward(target_time=600.0, grace_period=600.0, w_cost=0.01)
        order = _make_delivered(1300.0, 50.0)
        assert fn.compute(None, [order], 1.0) == pytest.approx(-0.5)  # type: ignore[arg-type]

    def test_empty_zero(self) -> None:
        assert LatencyAwareReward().compute(None, [], 1.0) == pytest.approx(0.0)  # type: ignore[arg-type]


class TestOptimizedDeliveryRewardKnownInputs:

    def test_fast_delivery_full_value(self) -> None:
        """Delivery well within target → value 1.0, minus cost 0.01×8=0.08."""
        fn = OptimizedDeliveryReward(target_time=500.0, grace_period=700.0,
                                     failure_penalty=1.0, w_cost=0.01)
        assert fn.compute(None, [_make_delivered(200.0, 8.0)], 1.0) == pytest.approx(0.92)  # type: ignore[arg-type]

    def test_half_grace_half_value(self) -> None:
        """Delivery at target + 0.5×grace → value 0.5, no cost penalty."""
        fn = OptimizedDeliveryReward(target_time=500.0, grace_period=700.0,
                                     failure_penalty=1.0, w_cost=0.0)
        assert fn.compute(None, [_make_delivered(850.0, 0.0)], 1.0) == pytest.approx(0.5)  # type: ignore[arg-type]

    def test_mixed_exact(self) -> None:
        """2 delivered + 1 failed: 1+1 − 1 − 0.01×20 = 0.8."""
        fn = OptimizedDeliveryReward(target_time=500.0, grace_period=700.0,
                                     failure_penalty=1.0, w_cost=0.01)
        orders = [
            _make_delivered(200.0, 5.0),
            _make_delivered(200.0, 15.0),
            _make_failed(),
        ]
        assert fn.compute(None, orders, 1.0) == pytest.approx(0.8)  # type: ignore[arg-type]

    def test_empty_zero(self) -> None:
        assert OptimizedDeliveryReward().compute(None, [], 1.0) == pytest.approx(0.0)  # type: ignore[arg-type]

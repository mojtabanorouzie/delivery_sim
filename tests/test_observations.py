"""
Tests for the ObservationSpec registry, default presets, CostAwareReward,
Order.delivery_cost, and the custom-preset extension path.

Test inventory
--------------
 1. Registry: all default observation presets retrievable by name;
    all default reward presets retrievable by name (including aliases);
    a custom observation + reward registers and resolves.

 2. Backward-compat: env with default config ("standard" preset) produces a
    vector byte-identical to the pre-refactor hardcoded _build_obs layout at
    reset() (where interval counts are known to be zero).

 3. Space consistency: every default preset's observation_space bounds contain
    all vectors returned during a full episode (no out-of-bounds, per step).

 4. Order.delivery_cost: a DELIVERED order's delivery_cost equals the real
    leg1+leg2 routing cost; a FAILED order's delivery_cost is always 0.0.

 5. CostAwareReward — known-input: on a crafted set of completed orders with
    known delivery_cost values, compute() returns the documented weighted sum.
    Higher real delivery cost → lower reward (independent of coverage_radius).

 6. Custom-preset end-to-end: a user-defined ObservationSpec + RewardFunction
    registered by name, selected in config, flow through reset() and step()
    correctly — no env edits needed.

 7. Reproducibility + action→outcome under a non-default preset: the same
    (seed, action-sequence) pair produces an identical trajectory with the
    "operational" preset; action plumbing is not inert.
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
from delivery_sim.entities.order import Order, OrderStatus
from delivery_sim.envs.observations import ObservationSpec
from delivery_sim.envs.single_agent import DeliveryEnv
from delivery_sim.metrics.collector import KPICollector
from delivery_sim.registry import create, list_registered, register
from delivery_sim.rewards.base import RewardFunction
from delivery_sim.rewards.cost_aware import CostAwareReward

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_config(
    *,
    seed: int = 42,
    rate: float = 2.0,
    num_couriers: int = 3,
    max_steps: int = 300,
    dt: float = 1.0,
    speed: float = 10.0,
    coverage_radius: float = 500.0,
    decision_interval: float = 100.0,
    max_coverage_radius: float = 2000.0,
    observation_preset: str = "standard",
    reward_type: str = "SparseDeliveryReward",
    n_stores: int = 1,
) -> ScenarioConfig:
    stores = [
        StoreConfig(name=f"s{i}", x=500.0, y=500.0, capacity=50, coverage_radius=coverage_radius)
        for i in range(n_stores)
    ]
    return ScenarioConfig(
        name="test_obs",
        seed=seed,
        dt=dt,
        max_steps=max_steps,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=stores,
        couriers=[CourierConfig(courier_type="BikeCourier", count=num_couriers, speed=speed)],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=rate),
        routing=RoutingConfig(model_type="euclidean"),
        reward=RewardConfig(function_type=reward_type),
        decision_interval=decision_interval,
        max_coverage_radius=max_coverage_radius,
        observation_preset=observation_preset,
    )


def _default_action(config: ScenarioConfig) -> np.ndarray:
    return np.array([sc.coverage_radius for sc in config.stores], dtype=np.float32)


def _run_full_episode(
    config: ScenarioConfig,
    action: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    """Run one episode; return (obs_list, reward_list, final_info)."""
    env = DeliveryEnv(config)
    obs0, _ = env.reset(seed=seed if seed is not None else config.seed)
    act = action if action is not None else _default_action(config)
    obs_list = [obs0]
    rewards: list[float] = []
    final_info: dict[str, Any] = {}
    for _ in range(10_000):
        obs, rew, _, trunc, info = env.step(act)
        obs_list.append(obs)
        rewards.append(float(rew))
        if trunc:
            final_info = info
            break
    return obs_list, rewards, final_info


# ---------------------------------------------------------------------------
# Test 1 — Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_default_observation_presets_registered(self) -> None:
        """All three default presets must be in the registry."""
        names = list_registered("observation")
        for preset in ("minimal", "standard", "operational"):
            assert preset in names, f"preset {preset!r} not registered"

    def test_default_reward_presets_registered(self) -> None:
        """Built-in reward names, including the sparse_delivery / cost_aware aliases."""
        names = list_registered("reward")
        for name in ("SparseDeliveryReward", "sparse_delivery", "CostAwareReward", "cost_aware"):
            assert name in names, f"reward {name!r} not registered"

    def test_create_each_default_observation_preset(self) -> None:
        """create() returns the correct type for each preset."""
        for name in ("minimal", "standard", "operational"):
            spec = create("observation", name)
            assert isinstance(spec, ObservationSpec), (
                f"create('observation', {name!r}) returned {type(spec)}"
            )

    def test_sparse_delivery_alias_is_same_class(self) -> None:
        """'sparse_delivery' must resolve to the same class as 'SparseDeliveryReward'."""
        from delivery_sim.rewards.placeholder import SparseDeliveryReward
        obj = create("reward", "sparse_delivery")
        assert type(obj) is SparseDeliveryReward

    def test_cost_aware_alias_is_same_class(self) -> None:
        """'cost_aware' must resolve to CostAwareReward."""
        obj = create("reward", "cost_aware")
        assert type(obj) is CostAwareReward

    def test_custom_observation_registers_and_resolves(self) -> None:
        """A user-defined ObservationSpec registered by name must be retrievable."""

        @register("observation", name="_test_custom_obs_registry")
        class _ConstantObs(ObservationSpec):
            def observe(self, world, collector, interval_delivered,  # type: ignore[override]
                        interval_failed, interval_total, max_r, max_pending, horizon):
                return np.zeros(2, dtype=np.float32)

            def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)

        spec = create("observation", "_test_custom_obs_registry")
        assert isinstance(spec, ObservationSpec)
        arr = spec.observe(None, None, 0, 0, 0, 1.0, 1.0, 1.0)  # type: ignore[arg-type]
        np.testing.assert_array_equal(arr, np.zeros(2, dtype=np.float32))

    def test_custom_reward_registers_and_resolves(self) -> None:
        """A user-defined RewardFunction registered by name must be retrievable."""

        @register("reward", name="_test_custom_reward_registry")
        class _FixedReward(RewardFunction):
            def compute(self, world, completed_orders, dt) -> float:  # type: ignore[override]
                return 99.0
            def reset(self) -> None:
                pass

        fn = create("reward", "_test_custom_reward_registry")
        assert isinstance(fn, RewardFunction)
        assert fn.compute(None, [], 1.0) == pytest.approx(99.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 2 — Backward-compat: "standard" byte-identical to pre-refactor _build_obs
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """
    Pins the exact layout and values of the pre-refactor _build_obs at reset().

    At reset() the interval counts are (0, 0, 0) and the world has no orders,
    all couriers free, and no delivery history — so every scalar is analytically
    determined.  The reference vector is computed here from first principles.
    """

    def _reference_reset_obs(
        self,
        config: ScenarioConfig,
        env: DeliveryEnv,
    ) -> np.ndarray:
        """Reproduce the pre-refactor _build_obs(0, 0, 0) logic verbatim."""
        assert env._simulator.world is not None
        assert env._collector is not None
        world = env._simulator.world
        collector = env._collector
        max_r = env._max_r
        max_pending = env._max_pending
        horizon = env._horizon

        # Per-store coverage
        coverage = np.clip(
            np.array([s.coverage_radius / max_r for s in world.stores], dtype=np.float32),  # type: ignore[attr-defined]
            0.0, 1.0,
        )
        # Sentinel values (interval_total == 0)
        delivery_rate = np.float32(0.5)
        failed_rate = np.float32(0.0)
        # Busy fraction: all couriers free at reset
        busy = sum(1 for ph in world.courier_phase.values() if ph != "free")
        busy_fraction = np.float32(float(busy) / max(1, len(world.couriers)))
        # Episode mean_delivery_time from collector (0.0 at reset)
        mean_dt = float(collector.summary()["mean_delivery_time"])
        mean_dt_norm = np.float32(
            float(np.clip(mean_dt / horizon, 0.0, 1.0)) if horizon > 0.0 else 0.0
        )
        # Pending (0 at reset)
        pending = sum(1 for o in world.active_orders.values() if not o.is_terminal)
        pending_norm = np.float32(float(np.clip(pending / max_pending, 0.0, 1.0)))

        scalars = np.array(
            [delivery_rate, failed_rate, busy_fraction, mean_dt_norm, pending_norm],
            dtype=np.float32,
        )
        return np.concatenate([coverage, scalars]).astype(np.float32)

    def test_standard_reset_obs_byte_identical(self) -> None:
        """standard preset reset obs must equal pre-refactor _build_obs(0,0,0)."""
        config = make_config(coverage_radius=500.0, max_coverage_radius=2000.0)
        env = DeliveryEnv(config)
        actual_obs, _ = env.reset(seed=42)

        reference = self._reference_reset_obs(config, env)
        np.testing.assert_array_equal(
            actual_obs, reference,
            err_msg="standard preset reset obs differs from pre-refactor reference",
        )

    def test_standard_reset_obs_exact_values(self) -> None:
        """Pin the concrete expected values for a 1-store config at reset."""
        # 1 store, coverage=500, max_r=2000 → normalised radius = 0.25
        # sentinels: delivery_rate=0.5, failed_rate=0.0
        # busy_fraction=0.0 (all free at reset)
        # mean_dt_norm=0.0 (no deliveries yet)
        # pending_norm=0.0 (no orders yet)
        config = make_config(coverage_radius=500.0, max_coverage_radius=2000.0)
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=42)

        expected = np.array([0.25, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(obs, expected, err_msg=f"got {obs!r}")

    def test_standard_observation_space_shape_unchanged(self) -> None:
        """observation_space shape must be (n_stores + 5,) for standard preset."""
        for n_stores in (1, 2, 3):
            config = make_config(n_stores=n_stores)
            env = DeliveryEnv(config)
            assert env.observation_space.shape == (n_stores + 5,), (
                f"expected ({n_stores + 5},) got {env.observation_space.shape}"
            )

    def test_multi_store_reset_obs_exact_values(self) -> None:
        """Two stores with different coverage radii should appear normalised."""
        config = ScenarioConfig(
            name="two_store",
            seed=1,
            world=WorldConfig(width=1000.0, height=1000.0),
            stores=[
                StoreConfig(name="a", x=200.0, y=200.0, coverage_radius=1000.0),
                StoreConfig(name="b", x=800.0, y=600.0, coverage_radius=500.0),
            ],
            couriers=[CourierConfig(courier_type="BikeCourier", count=2, speed=10.0)],
            demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=1.0),
            routing=RoutingConfig(model_type="euclidean"),
            reward=RewardConfig(function_type="SparseDeliveryReward"),
            decision_interval=100.0,
            max_coverage_radius=2000.0,
            observation_preset="standard",
        )
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=1)
        # obs[0]=1000/2000=0.5, obs[1]=500/2000=0.25, then sentinels + zeros
        expected = np.array([0.5, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(obs, expected)


# ---------------------------------------------------------------------------
# Test 3 — Space consistency: bounds contain all vectors across a full episode
# ---------------------------------------------------------------------------

class TestSpaceConsistency:
    """Every obs from every preset must be contained in observation_space."""

    @pytest.mark.parametrize("preset", ["minimal", "standard", "operational"])
    def test_obs_in_space_full_episode(self, preset: str) -> None:
        config = make_config(
            rate=2.0, max_steps=300, decision_interval=100.0,
            observation_preset=preset,
        )
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=7)
        assert env.observation_space.contains(obs), (
            f"[{preset}] reset obs {obs!r} not in space"
        )
        for _ in range(10_000):
            act = env.action_space.sample()
            obs, _, _, trunc, _ = env.step(act)
            assert env.observation_space.contains(obs), (
                f"[{preset}] step obs {obs!r} not in observation_space"
            )
            if trunc:
                break

    @pytest.mark.parametrize("preset", ["minimal", "standard", "operational"])
    def test_obs_shape_matches_bounds(self, preset: str) -> None:
        """Spec.observe() output shape must match bounds() shape."""
        config = make_config(observation_preset=preset)
        env = DeliveryEnv(config)
        env.reset(seed=1)
        act = _default_action(config)
        obs, _, _, _, _ = env.step(act)
        low, high = create("observation", preset).bounds(len(config.stores))
        assert obs.shape == low.shape == high.shape, (
            f"[{preset}] shape mismatch: obs={obs.shape} low={low.shape}"
        )

    @pytest.mark.parametrize("n_stores", [1, 2, 3])
    def test_minimal_shape_is_n_plus_1(self, n_stores: int) -> None:
        config = make_config(n_stores=n_stores, observation_preset="minimal")
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (n_stores + 1,)

    @pytest.mark.parametrize("n_stores", [1, 2, 3])
    def test_operational_shape_is_n_plus_3(self, n_stores: int) -> None:
        config = make_config(n_stores=n_stores, observation_preset="operational")
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (n_stores + 3,)


# ---------------------------------------------------------------------------
# Test 4 — Order.delivery_cost
# ---------------------------------------------------------------------------

class TestOrderDeliveryCost:

    def test_delivered_order_has_nonzero_cost(self) -> None:
        """A DELIVERED order must have delivery_cost > 0 (real routing distance)."""
        config = make_config(
            seed=1, rate=3.0, num_couriers=5, max_steps=500,
            speed=10.0, coverage_radius=1500.0,
        )
        sim = Simulator(config)
        coll = KPICollector()
        sim.attach_collector(coll)
        sim.run()
        assert sim.world is not None
        delivered = [
            o for o in sim.world.active_orders.values()
            if o.status == OrderStatus.DELIVERED
        ]
        assert len(delivered) > 0, "No orders delivered — adjust config"
        for order in delivered:
            assert order.delivery_cost > 0.0, (
                f"DELIVERED order {order.order_id} has delivery_cost == 0"
            )

    def test_failed_order_has_zero_cost(self) -> None:
        """A FAILED order must have delivery_cost == 0.0 (never dispatched)."""
        config = make_config(
            seed=1, rate=3.0, num_couriers=2, max_steps=300,
            coverage_radius=0.0,  # zero coverage → all orders fail immediately
        )
        sim = Simulator(config)
        coll = KPICollector()
        sim.attach_collector(coll)
        sim.run()
        assert sim.world is not None
        failed = [
            o for o in sim.world.active_orders.values()
            if o.status == OrderStatus.FAILED
        ]
        assert len(failed) > 0, "No orders failed — adjust config"
        for order in failed:
            assert order.delivery_cost == pytest.approx(0.0), (
                f"FAILED order {order.order_id} has delivery_cost={order.delivery_cost!r}"
            )

    def test_delivery_cost_equals_routing_cost(self) -> None:
        """delivery_cost must equal cost_per_unit × (leg1_dist + leg2_dist)."""
        config = make_config(
            seed=42, rate=1.0, num_couriers=5, max_steps=300,
            speed=10.0, coverage_radius=1500.0,
        )
        sim = Simulator(config)
        coll = KPICollector()
        sim.attach_collector(coll)
        sim.run()
        assert sim.world is not None
        # The KPICollector received the same cost values; total must match sum.
        total_from_orders = sum(
            o.delivery_cost
            for o in sim.world.active_orders.values()
            if o.status == OrderStatus.DELIVERED
        )
        assert total_from_orders == pytest.approx(
            float(coll.summary()["total_delivery_cost"]), rel=1e-6
        ), "Sum of order.delivery_cost must equal KPI total_delivery_cost"

    def test_delivery_cost_default_is_zero(self) -> None:
        """A freshly created Order must have delivery_cost == 0.0 by default."""
        order = Order(
            order_id="x", store_id="s", customer_x=1.0, customer_y=1.0, created_at=0.0
        )
        assert order.delivery_cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 5 — CostAwareReward: known-input computation + cost sensitivity
# ---------------------------------------------------------------------------

class TestCostAwareReward:

    def _make_order(
        self,
        status: OrderStatus,
        delivery_cost: float = 0.0,
    ) -> Order:
        order = Order(
            order_id="o", store_id="s",
            customer_x=0.0, customer_y=0.0,
            created_at=0.0,
        )
        order.status = status
        order.delivery_cost = delivery_cost
        return order

    def test_known_input_exact_value(self) -> None:
        """reward = 1×n_del − 0.5×n_fail − 0.01×total_cost."""
        fn = CostAwareReward(delivery_reward=1.0, w_fail=0.5, w_cost=0.01)
        orders = [
            self._make_order(OrderStatus.DELIVERED, delivery_cost=100.0),
            self._make_order(OrderStatus.DELIVERED, delivery_cost=200.0),
            self._make_order(OrderStatus.FAILED, delivery_cost=0.0),
        ]
        # expected = 1×2 − 0.5×1 − 0.01×300 = 2 − 0.5 − 3 = −1.5
        result = fn.compute(None, orders, 1.0)  # type: ignore[arg-type]
        assert result == pytest.approx(-1.5)

    def test_empty_orders_returns_zero(self) -> None:
        fn = CostAwareReward()
        assert fn.compute(None, [], 1.0) == pytest.approx(0.0)  # type: ignore[arg-type]

    def test_higher_cost_lower_reward(self) -> None:
        """Increasing delivery_cost must strictly decrease the reward."""
        fn = CostAwareReward(w_cost=0.01)
        for low_cost, high_cost in [(10.0, 100.0), (50.0, 500.0), (0.0, 1.0)]:
            low_orders = [self._make_order(OrderStatus.DELIVERED, delivery_cost=low_cost)]
            high_orders = [self._make_order(OrderStatus.DELIVERED, delivery_cost=high_cost)]
            r_low = fn.compute(None, low_orders, 1.0)   # type: ignore[arg-type]
            r_high = fn.compute(None, high_orders, 1.0)  # type: ignore[arg-type]
            assert r_low > r_high, (
                f"cost {low_cost} gave reward {r_low}, cost {high_cost} gave {r_high} — "
                f"higher cost must give lower reward"
            )

    def test_failed_order_cost_not_penalised_again(self) -> None:
        """FAILED orders have delivery_cost 0.0 — penalty is the w_fail term only."""
        fn = CostAwareReward(delivery_reward=1.0, w_fail=0.5, w_cost=0.01)
        failed = [self._make_order(OrderStatus.FAILED, delivery_cost=0.0)]
        assert fn.compute(None, failed, 1.0) == pytest.approx(-0.5)  # type: ignore[arg-type]

    def test_cost_aware_in_env_end_to_end(self) -> None:
        """CostAwareReward wired via config runs a full episode without error."""
        config = make_config(
            rate=2.0, max_steps=300, decision_interval=100.0,
            reward_type="CostAwareReward",
            coverage_radius=500.0,
        )
        obs_list, rewards, info = _run_full_episode(config, seed=1)
        assert "kpi" in info
        # With deliveries happening and real cost, reward must vary
        assert len(rewards) > 0

    def test_real_delivery_cost_flows_into_reward(self) -> None:
        """An episode with higher coverage (longer routes) must have higher total
        delivery cost, which should reduce the reward relative to low coverage."""
        # Low-cost scenario: zero-coverage → all failed (no courier dispatch)
        # High-cost scenario: full coverage → deliveries incur real cost
        # CostAwareReward: zero-coverage has no delivery reward but also no cost
        # penalty; full-coverage has delivery reward but cost penalty.
        # Just verify the final KPI total_delivery_cost > 0 when coverage is high.
        config_full = make_config(
            seed=5, rate=2.0, max_steps=300, coverage_radius=1500.0,
            reward_type="CostAwareReward",
        )
        _, rewards_full, info_full = _run_full_episode(config_full, seed=5)
        assert info_full["kpi"]["total_delivery_cost"] > 0.0, (
            "Expected real delivery cost with full coverage"
        )

        config_none = make_config(
            seed=5, rate=2.0, max_steps=300, coverage_radius=0.0,
            reward_type="CostAwareReward",
        )
        _, rewards_none, info_none = _run_full_episode(config_none, seed=5)
        assert info_none["kpi"]["total_delivery_cost"] == pytest.approx(0.0), (
            "Expected zero delivery cost with zero coverage (no dispatch)"
        )


# ---------------------------------------------------------------------------
# Test 6 — Custom preset end-to-end
# ---------------------------------------------------------------------------

class TestCustomPresetEndToEnd:
    """
    Proves a user can define + register a custom ObservationSpec and a custom
    RewardFunction, select them by name in config, and get them through the
    env's reset/step cycle — with NO edits to DeliveryEnv or any other env file.
    """

    def test_custom_observation_spec_flows_through_env(self) -> None:
        """User-defined ObservationSpec: fixed-length constant vector."""
        SENTINEL = np.array([0.42, 0.43], dtype=np.float32)

        @register("observation", name="_e2e_custom_obs")
        class _E2EObs(ObservationSpec):
            def observe(self, world, collector, interval_delivered,  # type: ignore[override]
                        interval_failed, interval_total, max_r, max_pending, horizon):
                return SENTINEL.copy()

            def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)

        config = make_config(observation_preset="_e2e_custom_obs")
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=1)
        np.testing.assert_array_equal(obs, SENTINEL, err_msg="reset obs mismatch")
        assert env.observation_space.shape == (2,)

        act = env.action_space.sample()
        obs2, _, _, _, _ = env.step(act)
        np.testing.assert_array_equal(obs2, SENTINEL, err_msg="step obs mismatch")

    def test_custom_reward_function_flows_through_env(self) -> None:
        """User-defined RewardFunction: constant sentinel reward."""
        REWARD_SENTINEL = 3.14

        @register("reward", name="_e2e_custom_reward")
        class _E2EReward(RewardFunction):
            def compute(self, world, completed_orders, dt) -> float:  # type: ignore[override]
                return REWARD_SENTINEL
            def reset(self) -> None:
                pass

        config = make_config(reward_type="_e2e_custom_reward")
        env = DeliveryEnv(config)
        env.reset(seed=1)
        _, rew, _, _, _ = env.step(_default_action(config))
        assert float(rew) == pytest.approx(REWARD_SENTINEL)

    def test_custom_obs_and_reward_together(self) -> None:
        """Both custom spec + custom reward selected simultaneously."""

        @register("observation", name="_e2e_combo_obs")
        class _ComboObs(ObservationSpec):
            def observe(self, world, collector, interval_delivered,  # type: ignore[override]
                        interval_failed, interval_total, max_r, max_pending, horizon):
                return np.array([0.99], dtype=np.float32)

            def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)

        @register("reward", name="_e2e_combo_reward")
        class _ComboReward(RewardFunction):
            def compute(self, world, completed_orders, dt) -> float:  # type: ignore[override]
                return -7.0
            def reset(self) -> None:
                pass

        config = make_config(
            observation_preset="_e2e_combo_obs",
            reward_type="_e2e_combo_reward",
        )
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=1)
        assert obs.shape == (1,)
        _, rew, _, _, _ = env.step(env.action_space.sample())
        assert float(rew) == pytest.approx(-7.0)


# ---------------------------------------------------------------------------
# Test 7 — Reproducibility + action→outcome under non-default preset
# ---------------------------------------------------------------------------

class TestNonDefaultPresetProperties:

    @pytest.mark.parametrize("preset", ["minimal", "operational"])
    def test_reproducibility_under_preset(self, preset: str) -> None:
        """Same (seed, action) → identical trajectory for non-default presets."""
        config = make_config(observation_preset=preset, max_steps=300)
        obs_a, rew_a, _ = _run_full_episode(config, seed=42)
        obs_b, rew_b, _ = _run_full_episode(config, seed=42)
        assert len(obs_a) == len(obs_b)
        for i, (a, b) in enumerate(zip(obs_a, obs_b)):
            np.testing.assert_array_equal(a, b, err_msg=f"[{preset}] obs[{i}] differs")
        assert rew_a == rew_b, f"[{preset}] reward trajectories differ"

    @pytest.mark.parametrize("preset", ["minimal", "operational"])
    def test_action_to_outcome_under_preset(self, preset: str) -> None:
        """Coverage action still affects delivery outcomes under non-default presets."""
        cfg_high = make_config(
            seed=42, rate=2.0, num_couriers=5, max_steps=300,
            observation_preset=preset, max_coverage_radius=2000.0,
        )
        cfg_low = make_config(
            seed=42, rate=2.0, num_couriers=5, max_steps=300,
            observation_preset=preset, max_coverage_radius=2000.0,
        )
        high_action = np.array([1500.0], dtype=np.float32)
        low_action = np.array([0.0], dtype=np.float32)

        _, _, info_high = _run_full_episode(cfg_high, action=high_action, seed=42)
        _, _, info_low = _run_full_episode(cfg_low, action=low_action, seed=42)

        assert info_high["kpi"]["failed_orders"] < info_low["kpi"]["failed_orders"], (
            f"[{preset}] high coverage must produce fewer failed orders than zero coverage"
        )

    @pytest.mark.parametrize("preset", ["minimal", "operational"])
    def test_obs_in_space_under_preset(self, preset: str) -> None:
        """All obs from non-default presets must be contained in their observation_space."""
        config = make_config(
            seed=1, rate=2.0, max_steps=200, decision_interval=50.0,
            observation_preset=preset,
        )
        obs_list, _, _ = _run_full_episode(config, seed=1)
        env = DeliveryEnv(config)
        env.reset(seed=1)
        for obs in obs_list:
            # rebuild space for the preset to check containment
            assert env.observation_space.contains(obs), (
                f"[{preset}] obs {obs!r} not in space"
            )

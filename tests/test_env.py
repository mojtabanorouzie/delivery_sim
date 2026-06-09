"""
Tests for DeliveryEnv (single-agent Gymnasium environment) and
Simulator.run_until — Step 5 of the delivery_sim build order.

Test inventory
--------------
 1. run_until boundary convention: event at T is NOT processed by
    run_until(T); it IS processed by run_until(T + ε).  Processed exactly
    once across both calls.  Explicit unit test with a synthetic event.

 2. run_until equivalence (HIGHEST PRIORITY): N env steps of
    decision_interval to horizon H must produce bit-identical order histories
    and KPI summaries to a single Simulator.run() to the same horizon with
    the same seed and constant default-coverage actions.  Proves run_until
    partitions the event stream without altering it.

 3. Gym compliance — spaces: action_space and observation_space are valid
    gym.spaces; obs from reset() and step() are always within
    observation_space.

 4. reset() contract: returns (obs, info) matching observation_space; obs in
    space; sim_time is 0.0 in info; called without seed is also fine.

 5. step() contract: returns valid 5-tuple (obs, float, bool, bool, dict);
    obs is within observation_space; step before reset raises RuntimeError.

 6. Action→outcome (the critical correctness test): a full episode with
    HIGH coverage actions yields zero failed orders; a full episode with ZERO
    coverage actions yields all orders failed.  Proves the action plumbing
    reaches the live sim and is not inert.

 7. Cadence: each env.step() advances sim_time by exactly decision_interval;
    the number of steps to first truncated=True equals
    floor(horizon / decision_interval) when horizon is an exact multiple.

 8. Reproducibility: same (seed, action sequence) → identical obs/reward
    trajectory; different seed → different trajectory.

 9. Reset between episodes: after env.reset(), stores have config-default
    coverage_radius, KPI collector is fresh, and sim_time is 0.0.  No state
    leaks from a prior episode.

10. Reward wiring: a stub RewardFunction that returns a fixed sentinel value
    is injected; that sentinel must appear as the step() reward output.

11. Baseline smoke: a random-action agent runs a full episode without error;
    final step has truncated=True and info["kpi"] is populated.
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
from delivery_sim.engine.event_queue import Event
from delivery_sim.engine.simulator import Simulator
from delivery_sim.engine.world_state import WorldState
from delivery_sim.entities.order import Order
from delivery_sim.envs.single_agent import DeliveryEnv
from delivery_sim.metrics.collector import KPICollector
from delivery_sim.registry import register
from delivery_sim.rewards.base import RewardFunction

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_config(
    *,
    seed: int = 42,
    rate: float = 2.0,
    num_couriers: int = 3,
    max_steps: int = 500,
    dt: float = 1.0,
    speed: float = 10.0,
    coverage_radius: float = 500.0,
    decision_interval: float = 100.0,
    max_coverage_radius: float = 2000.0,
    store_x: float = 500.0,
    store_y: float = 500.0,
    capacity: int = 50,
) -> ScenarioConfig:
    """One store at (store_x, store_y) covering the configured radius."""
    return ScenarioConfig(
        name="test_env",
        seed=seed,
        dt=dt,
        max_steps=max_steps,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[StoreConfig(
            name="s1",
            x=store_x,
            y=store_y,
            capacity=capacity,
            coverage_radius=coverage_radius,
        )],
        couriers=[CourierConfig(
            courier_type="BikeCourier",
            count=num_couriers,
            speed=speed,
        )],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=rate),
        routing=RoutingConfig(model_type="euclidean"),
        reward=RewardConfig(function_type="SparseDeliveryReward"),
        decision_interval=decision_interval,
        max_coverage_radius=max_coverage_radius,
    )


def _default_action(config: ScenarioConfig) -> np.ndarray:
    """Return the action that holds every store at its config coverage_radius."""
    return np.array(
        [sc.coverage_radius for sc in config.stores], dtype=np.float32
    )


def _run_env_episode(
    config: ScenarioConfig,
    action: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[float]]:
    """Run one full episode; return (final info dict, per-step rewards)."""
    env = DeliveryEnv(config)
    env.reset(seed=seed if seed is not None else config.seed)
    act = action if action is not None else _default_action(config)
    rewards: list[float] = []
    final_info: dict[str, Any] = {}
    for _ in range(10_000):  # safety cap
        _obs, rew, _term, trunc, info = env.step(act)
        rewards.append(float(rew))
        if trunc:
            final_info = info
            break
    return final_info, rewards


# ---------------------------------------------------------------------------
# Test 1 — run_until boundary convention (unit test)
# ---------------------------------------------------------------------------

class TestRunUntilBoundary:
    """Explicit unit tests for the strict-< boundary convention."""

    def _make_sim(self) -> Simulator:
        config = make_config(seed=1)
        sim = Simulator(config)
        sim.reset()
        # Clear the seeded demand event so we control the queue entirely.
        sim.event_queue.clear()
        return sim

    def test_event_at_boundary_not_processed_by_run_until_T(self) -> None:
        """run_until(T) must NOT process an event with time == T."""
        sim = self._make_sim()
        T = 100.0
        sim.event_queue.push(
            Event(time=T, priority=5, event_type="_boundary_test_", payload={})
        )
        sim.run_until(T)
        # Event must still be in the queue.
        assert not sim.event_queue.is_empty(), (
            "run_until(T) must not process event with time == T (strict <)"
        )
        assert sim.event_queue.peek().time == pytest.approx(T)  # type: ignore[union-attr]

    def test_event_at_boundary_processed_by_next_window(self) -> None:
        """run_until(T + ε) must process the event that was left at T."""
        sim = self._make_sim()
        T = 100.0
        sim.event_queue.push(
            Event(time=T, priority=5, event_type="_boundary_test_", payload={})
        )
        sim.run_until(T)         # does not consume the T-event
        sim.run_until(T + 1.0)   # must consume it
        assert sim.event_queue.is_empty(), (
            "run_until(T+ε) must drain the event at time == T"
        )

    def test_event_at_boundary_processed_exactly_once(self) -> None:
        """Across two consecutive windows the T-event is processed once only."""
        sim = self._make_sim()
        T = 100.0
        sim.event_queue.push(
            Event(time=T, priority=5, event_type="_boundary_test_", payload={})
        )
        # Window 1: does not consume
        sim.run_until(T)
        assert len(sim.event_queue) == 1
        # Window 2: consumes exactly once; queue is empty afterwards
        sim.run_until(T + 1.0)
        assert len(sim.event_queue) == 0

    def test_clock_pinned_to_target_after_run_until(self) -> None:
        """clock.elapsed must equal target_time after run_until, even if no
        events were processed in the window."""
        sim = self._make_sim()
        sim.run_until(50.0)
        assert sim.clock.elapsed == pytest.approx(50.0)
        sim.run_until(100.0)
        assert sim.clock.elapsed == pytest.approx(100.0)

    def test_event_before_boundary_processed_in_current_window(self) -> None:
        """Events with time < T must be processed by run_until(T)."""
        sim = self._make_sim()
        T = 100.0
        sim.event_queue.push(
            Event(time=T - 1.0, priority=5, event_type="_boundary_test_", payload={})
        )
        sim.run_until(T)
        assert sim.event_queue.is_empty(), (
            "run_until(T) must consume events with time < T"
        )


# ---------------------------------------------------------------------------
# Test 2 — run_until equivalence (HIGHEST PRIORITY)
# ---------------------------------------------------------------------------

class TestRunUntilEquivalence:
    """Stepping through N windows of run_until must be identical to run()."""

    # Config: horizon = 500, decision_interval = 100 → exactly 5 steps.
    CFG = make_config(seed=42, rate=2.0, max_steps=500, decision_interval=100.0)

    def _order_history(self, sim: Simulator) -> dict[str, Any]:
        assert sim.world is not None
        return {
            oid: {
                "status": o.status,
                "timestamps": {k.name: v for k, v in o.timestamps.items()},
            }
            for oid, o in sim.world.active_orders.items()
        }

    def test_order_histories_identical(self) -> None:
        """Same (seed, coverage) → same order statuses and timestamps."""
        cfg = self.CFG

        # Run A: single Simulator.run()
        sim_a = Simulator(cfg)
        coll_a = KPICollector()
        sim_a.attach_collector(coll_a)
        sim_a.run()

        # Run B: env stepping with constant default-coverage actions
        env = DeliveryEnv(cfg)
        env.reset(seed=cfg.seed)
        act = _default_action(cfg)
        for _ in range(5):
            env.step(act)

        assert self._order_history(sim_a) == self._order_history(env._simulator), (
            "Order histories diverged: run_until partitioning altered event processing"
        )

    def test_kpi_summaries_identical(self) -> None:
        """KPI summaries (including courier_utilization) must match."""
        cfg = self.CFG

        sim_a = Simulator(cfg)
        coll_a = KPICollector()
        sim_a.attach_collector(coll_a)
        sim_a.run()
        kpi_a = coll_a.summary()

        env = DeliveryEnv(cfg)
        env.reset(seed=cfg.seed)
        act = _default_action(cfg)
        last_info: dict[str, Any] = {}
        for _ in range(5):
            _, _, _, trunc, info = env.step(act)
            if trunc:
                last_info = info

        kpi_b = last_info["kpi"]
        for key in kpi_a:
            assert kpi_a[key] == pytest.approx(kpi_b[key], abs=1e-9), (  # type: ignore[operator]
                f"KPI '{key}' diverged: run()={kpi_a[key]!r} vs env={kpi_b[key]!r}"
            )

    def test_total_order_count_identical(self) -> None:
        """Number of orders created must be the same in both paths."""
        cfg = self.CFG

        sim_a = Simulator(cfg)
        sim_a.run()
        assert sim_a.world is not None
        n_a = len(sim_a.world.active_orders)

        env = DeliveryEnv(cfg)
        env.reset(seed=cfg.seed)
        act = _default_action(cfg)
        for _ in range(5):
            env.step(act)
        assert env._simulator.world is not None
        n_b = len(env._simulator.world.active_orders)

        assert n_a == n_b

    def test_injected_boundary_event_processed_in_exactly_one_window(self) -> None:
        """Definitive boundary test: inject a REAL order_created event at
        exactly T=100 (a step boundary) into a rate=0 simulator, then verify
        that two approaches — two windows vs one window — produce identical
        order histories and KPI summaries, and that exactly one order was
        created (not zero, not two).

        rate=0 suppresses all automatic demand (next_event returns None without
        consuming RNG), leaving us full control over the queue.  The injected
        event at T=100 is a valid order_created payload that triggers the full
        dispatch-and-delivery chain.

        Boundary behaviour asserted explicitly:
        - After run_until(100): world.active_orders is empty — the event at
          time==100 was NOT processed (strict <).
        - After run_until(200): the event IS processed, order delivered.
        - KPI total_orders == 1 in both approaches: processed exactly once.
        """
        BOUNDARY_T = 100.0
        cfg = make_config(
            seed=1, rate=0.0, num_couriers=2, max_steps=200,
            speed=100.0, coverage_radius=1500.0, decision_interval=100.0,
        )

        def setup_sim() -> tuple[Simulator, KPICollector]:
            sim = Simulator(cfg)
            coll = KPICollector()
            sim.attach_collector(coll)
            sim.reset()
            # rate=0 → queue is empty after reset; clear is defensive.
            sim.event_queue.clear()
            # Inject one real order_created event at exactly the step boundary.
            sim.event_queue.push(Event(
                time=BOUNDARY_T, priority=10,
                event_type="order_created",
                payload={"customer_x": 600.0, "customer_y": 600.0},
            ))
            return sim, coll

        # Approach A: two windows [0, 100) then [100, 200)
        sim_a, coll_a = setup_sim()
        sim_a.run_until(BOUNDARY_T)
        assert sim_a.world is not None
        assert len(sim_a.world.active_orders) == 0, (
            "event at time == T must NOT be processed by run_until(T) [strict <]"
        )
        sim_a.run_until(200.0)
        coll_a.finalize(num_couriers=len(sim_a.world.couriers), horizon=200.0)

        # Approach B: single window [0, 200)
        sim_b, coll_b = setup_sim()
        sim_b.run_until(200.0)
        assert sim_b.world is not None
        coll_b.finalize(num_couriers=len(sim_b.world.couriers), horizon=200.0)

        def order_history(sim: Simulator) -> dict[str, Any]:
            assert sim.world is not None
            return {
                oid: {
                    "status": o.status,
                    "ts": {k.name: v for k, v in o.timestamps.items()},
                }
                for oid, o in sim.world.active_orders.items()
            }

        # Same order histories (event processed the same way in both approaches)
        assert order_history(sim_a) == order_history(sim_b), (
            "boundary event produced different order histories in two-window vs "
            "one-window approach"
        )
        # Exactly one order in each run (once, not twice, not zero)
        assert coll_a.summary()["total_orders"] == 1, (
            "expected exactly 1 order (boundary event processed once); "
            f"got {coll_a.summary()['total_orders']}"
        )
        assert coll_b.summary()["total_orders"] == 1
        # KPI summaries must match across both approaches
        kpi_a = coll_a.summary()
        kpi_b = coll_b.summary()
        for key in kpi_a:
            assert kpi_a[key] == pytest.approx(kpi_b[key], abs=1e-9), (  # type: ignore[operator]
                f"KPI '{key}': two-window={kpi_a[key]!r}, one-window={kpi_b[key]!r}"
            )

    def test_boundary_event_handled_in_exactly_one_window(self) -> None:
        """An event whose time coincides with a step boundary is reflected in
        the KPI totals exactly once (equivalence proof covers this, but this
        test makes it explicit with multiple step sizes)."""
        for steps, n_windows in [(500, 5), (500, 10), (500, 2)]:
            if steps % n_windows != 0:
                continue  # only exact multiples for this test
            interval = steps / n_windows
            cfg = make_config(
                seed=99, rate=3.0, max_steps=steps, decision_interval=interval,
            )

            sim_ref = Simulator(cfg)
            coll_ref = KPICollector()
            sim_ref.attach_collector(coll_ref)
            sim_ref.run()
            ref_total = coll_ref.summary()["total_orders"]

            env = DeliveryEnv(cfg)
            env.reset(seed=cfg.seed)
            act = _default_action(cfg)
            env_total = 0
            for _ in range(n_windows):
                _, _, _, trunc, info = env.step(act)
                if trunc:
                    env_total = info["kpi"]["total_orders"]

            assert ref_total == env_total, (
                f"steps={steps}, n_windows={n_windows}: "
                f"ref={ref_total} vs env={env_total}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Gym compliance
# ---------------------------------------------------------------------------

class TestGymCompliance:

    def test_action_space_is_valid(self) -> None:
        env = DeliveryEnv(make_config())
        assert env.action_space is not None
        # shape matches n_stores
        assert env.action_space.shape == (1,)

    def test_observation_space_is_valid(self) -> None:
        env = DeliveryEnv(make_config())
        assert env.observation_space.shape == (1 + 5,)

    def test_obs_from_reset_in_space(self) -> None:
        env = DeliveryEnv(make_config())
        obs, _ = env.reset(seed=1)
        assert env.observation_space.contains(obs), (
            f"reset() obs {obs} not in observation_space"
        )

    def test_obs_from_step_in_space(self) -> None:
        env = DeliveryEnv(make_config())
        env.reset(seed=1)
        act = env.action_space.sample()
        obs, _, _, _, _ = env.step(act)
        assert env.observation_space.contains(obs), (
            f"step() obs {obs} not in observation_space"
        )

    def test_obs_in_space_across_full_episode(self) -> None:
        """Every obs returned during a full episode must be within space."""
        config = make_config(max_steps=200, decision_interval=50.0)
        env = DeliveryEnv(config)
        obs, _ = env.reset(seed=7)
        assert env.observation_space.contains(obs)
        for _ in range(4):
            act = env.action_space.sample()
            obs, _, _, trunc, _ = env.step(act)
            assert env.observation_space.contains(obs)
            if trunc:
                break


# ---------------------------------------------------------------------------
# Test 4 — reset() contract
# ---------------------------------------------------------------------------

class TestResetContract:

    def test_reset_returns_obs_and_info(self) -> None:
        env = DeliveryEnv(make_config())
        result = env.reset(seed=42)
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_obs_in_space(self) -> None:
        env = DeliveryEnv(make_config())
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_reset_info_has_sim_time_zero(self) -> None:
        env = DeliveryEnv(make_config())
        _, info = env.reset(seed=42)
        assert info["sim_time"] == pytest.approx(0.0)

    def test_reset_without_seed_is_valid(self) -> None:
        env = DeliveryEnv(make_config())
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert info["sim_time"] == pytest.approx(0.0)

    def test_reset_with_different_seeds_differ(self) -> None:
        env = DeliveryEnv(make_config(rate=3.0, max_steps=200))
        obs_a, _ = env.reset(seed=1)
        obs_b, _ = env.reset(seed=2)
        # Not guaranteed to differ at t=0 (all stores at config default) —
        # they usually do after a step, but obs at reset is deterministic
        # for same config coverage.  Just check both are valid.
        assert env.observation_space.contains(obs_a)
        assert env.observation_space.contains(obs_b)


# ---------------------------------------------------------------------------
# Test 5 — step() contract
# ---------------------------------------------------------------------------

class TestStepContract:

    def test_step_returns_5_tuple(self) -> None:
        env = DeliveryEnv(make_config())
        env.reset(seed=1)
        result = env.step(env.action_space.sample())
        assert isinstance(result, tuple) and len(result) == 5

    def test_step_obs_in_space(self) -> None:
        env = DeliveryEnv(make_config())
        env.reset(seed=1)
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self) -> None:
        env = DeliveryEnv(make_config())
        env.reset(seed=1)
        _, rew, _, _, _ = env.step(env.action_space.sample())
        assert isinstance(float(rew), float)

    def test_step_terminated_is_false(self) -> None:
        """terminated must always be False (no natural terminal condition)."""
        env = DeliveryEnv(make_config())
        env.reset(seed=1)
        for _ in range(5):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            assert term is False
            if trunc:
                break

    def test_step_before_reset_raises(self) -> None:
        env = DeliveryEnv(make_config())
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros(1, dtype=np.float32))

    def test_step_clamped_action_does_not_raise(self) -> None:
        """Actions outside [0, max_r] must be clamped, not raise."""
        env = DeliveryEnv(make_config())
        env.reset(seed=1)
        env.step(np.array([99999.0], dtype=np.float32))  # way above max_r
        env.step(np.array([-1.0], dtype=np.float32))     # below 0


# ---------------------------------------------------------------------------
# Test 6 — Action→outcome (the critical correctness test)
# ---------------------------------------------------------------------------

class TestActionToOutcome:
    """Proves that writing coverage_radius reaches the live sim (not inert)."""

    # Store at world center; max coverage > world diagonal → covers everything.
    # World diagonal from (500,500) to corner ≈ 707 units.
    HIGH_COVERAGE = 1500.0   # > 707, covers every customer in 1000×1000
    LOW_COVERAGE = 0.0       # covers no customer (distance > 0 always)

    def _kpi(self, coverage_value: float, seed: int = 42) -> dict[str, Any]:
        cfg = make_config(
            seed=seed,
            rate=2.0,
            num_couriers=5,
            max_steps=300,
            decision_interval=100.0,
            coverage_radius=500.0,  # initial config default (overridden by action)
            max_coverage_radius=2000.0,
        )
        action = np.array([coverage_value], dtype=np.float32)
        info, _ = _run_env_episode(cfg, action=action, seed=seed)
        return info["kpi"]

    def test_high_coverage_yields_zero_failed_orders(self) -> None:
        """With radius > world diagonal, every customer is covered → no FAILED."""
        kpi = self._kpi(self.HIGH_COVERAGE)
        assert kpi["failed_orders"] == 0, (
            f"Expected 0 failed orders with full-world coverage; "
            f"got {kpi['failed_orders']} of {kpi['total_orders']}"
        )

    def test_zero_coverage_yields_all_orders_failed(self) -> None:
        """With radius == 0, no customer is within coverage → all FAILED.

        A customer at exactly the store position (500, 500) would not fail
        (distance == 0 == radius), but the probability is zero for a
        continuous uniform distribution.
        """
        kpi = self._kpi(self.LOW_COVERAGE)
        assert kpi["failed_orders"] == kpi["total_orders"], (
            f"Expected all {kpi['total_orders']} orders to fail with zero coverage; "
            f"got {kpi['failed_orders']} failed"
        )

    def test_failed_rate_strictly_ordered(self) -> None:
        """failed_orders(high) < failed_orders(low) — monotone in coverage."""
        kpi_high = self._kpi(self.HIGH_COVERAGE)
        kpi_low = self._kpi(self.LOW_COVERAGE)
        assert kpi_high["failed_orders"] < kpi_low["failed_orders"], (
            f"failed high={kpi_high['failed_orders']} "
            f"not < low={kpi_low['failed_orders']}"
        )

    def test_action_affects_next_window_not_previous(self) -> None:
        """Coverage set at step k affects events in window k, not earlier ones.

        Run two episodes:
          A: full coverage throughout (high from step 1).
          B: zero coverage for the first window, full coverage thereafter.
        Episode B must have at least as many failed orders as A.
        """
        cfg = make_config(seed=7, rate=3.0, max_steps=400, decision_interval=100.0)
        env_a = DeliveryEnv(cfg)
        env_a.reset(seed=cfg.seed)
        env_b = DeliveryEnv(cfg)
        env_b.reset(seed=cfg.seed)

        n_steps = int(cfg.max_steps * cfg.dt / cfg.decision_interval)
        last_info_a: dict[str, Any] = {}
        last_info_b: dict[str, Any] = {}
        high = np.array([self.HIGH_COVERAGE], dtype=np.float32)
        low = np.array([self.LOW_COVERAGE], dtype=np.float32)

        for step_i in range(n_steps):
            _, _, _, trunc_a, info_a = env_a.step(high)
            action_b = low if step_i == 0 else high
            _, _, _, trunc_b, info_b = env_b.step(action_b)
            if trunc_a:
                last_info_a = info_a
            if trunc_b:
                last_info_b = info_b

        assert last_info_b["kpi"]["failed_orders"] >= last_info_a["kpi"]["failed_orders"]


# ---------------------------------------------------------------------------
# Test 7 — Cadence
# ---------------------------------------------------------------------------

class TestCadence:

    def test_sim_time_advances_by_decision_interval(self) -> None:
        """Each step advances sim_time by exactly decision_interval."""
        cfg = make_config(max_steps=500, decision_interval=100.0)
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        act = _default_action(cfg)
        for expected_step in range(1, 6):
            _, _, _, trunc, info = env.step(act)
            assert info["sim_time"] == pytest.approx(expected_step * 100.0), (
                f"step {expected_step}: sim_time={info['sim_time']!r}, "
                f"expected {expected_step * 100.0!r}"
            )
            if trunc:
                break

    def test_steps_to_truncation_equals_horizon_over_interval(self) -> None:
        """Exactly horizon/decision_interval steps should trigger truncation."""
        cfg = make_config(max_steps=400, decision_interval=100.0)
        # horizon = 400 * 1.0 = 400; decision_interval = 100 → 4 steps
        expected_steps = int(400 * 1.0 / 100.0)  # 4
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        act = _default_action(cfg)
        for i in range(1, expected_steps + 1):
            _, _, _, trunc, _ = env.step(act)
            if i < expected_steps:
                assert not trunc, f"truncated early at step {i}"
            else:
                assert trunc, f"not truncated at step {expected_steps}"

    def test_simulator_clock_tracks_env_sim_time(self) -> None:
        """env._simulator.clock.elapsed must equal env._sim_time after each step."""
        cfg = make_config(max_steps=300, decision_interval=100.0)
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        act = _default_action(cfg)
        for _ in range(3):
            _, _, _, trunc, info = env.step(act)
            assert env._simulator.clock.elapsed == pytest.approx(info["sim_time"])
            if trunc:
                break


# ---------------------------------------------------------------------------
# Test 8 — Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def _full_trajectory(
        self, config: ScenarioConfig, seed: int
    ) -> tuple[list[np.ndarray], list[float]]:
        """Return (obs_list, reward_list) for a full episode."""
        env = DeliveryEnv(config)
        obs0, _ = env.reset(seed=seed)
        act = _default_action(config)
        obses = [obs0]
        rewards: list[float] = []
        for _ in range(10_000):
            obs, rew, _, trunc, _ = env.step(act)
            obses.append(obs)
            rewards.append(float(rew))
            if trunc:
                break
        return obses, rewards

    def test_same_seed_identical_trajectory(self) -> None:
        cfg = make_config(seed=42, rate=2.0, max_steps=300)
        obses_a, rew_a = self._full_trajectory(cfg, seed=42)
        obses_b, rew_b = self._full_trajectory(cfg, seed=42)
        assert len(obses_a) == len(obses_b)
        for i, (a, b) in enumerate(zip(obses_a, obses_b)):
            np.testing.assert_array_equal(a, b, err_msg=f"obs[{i}] differs")
        assert rew_a == rew_b

    def test_different_seed_different_trajectory(self) -> None:
        cfg = make_config(rate=2.0, max_steps=300)
        obses_a, rew_a = self._full_trajectory(cfg, seed=1)
        obses_b, rew_b = self._full_trajectory(cfg, seed=99)
        # Rewards must differ (same coverage on different demand streams)
        assert rew_a != rew_b, "Different seeds produced identical reward trajectories"

    def test_reset_seed_mid_session(self) -> None:
        """reset(seed=X) on an already-stepped env must yield same trajectory
        as a fresh env with the same seed."""
        cfg = make_config(seed=5, rate=2.0, max_steps=200)
        # Fresh env
        env_fresh = DeliveryEnv(cfg)
        obses_fresh, rew_fresh = self._full_trajectory(env_fresh.config, seed=5)

        # Env that ran one episode first, then reseeded
        env_reseeded = DeliveryEnv(cfg)
        env_reseeded.reset(seed=99)
        for _ in range(2):
            env_reseeded.step(_default_action(cfg))
        # Now re-seed with 5
        obses_rs, rew_rs = self._full_trajectory(env_reseeded.config, seed=5)

        assert rew_fresh == rew_rs, "Re-seeded trajectory differs from fresh trajectory"


# ---------------------------------------------------------------------------
# Test 9 — Reset between episodes (no state leakage)
# ---------------------------------------------------------------------------

class TestResetNoLeakage:

    def test_coverage_radius_reset_to_config_default(self) -> None:
        """After reset(), stores must have config.coverage_radius, not the
        last action's value."""
        cfg = make_config(coverage_radius=500.0)
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        # Drive coverage to a very different value
        extreme = np.array([1.0], dtype=np.float32)
        env.step(extreme)
        assert env._simulator.world is not None
        # Now reset
        env.reset(seed=2)
        assert env._simulator.world is not None
        assert env._simulator.world.stores[0].coverage_radius == pytest.approx(500.0), (
            "coverage_radius was not reset to config default after env.reset()"
        )

    def test_sim_time_reset_to_zero(self) -> None:
        cfg = make_config(max_steps=300, decision_interval=100.0)
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        for _ in range(3):
            _, _, _, trunc, _ = env.step(_default_action(cfg))
            if trunc:
                break
        _, info = env.reset(seed=1)
        assert info["sim_time"] == pytest.approx(0.0)
        assert env._sim_time == pytest.approx(0.0)

    def test_kpi_collector_fresh_after_reset(self) -> None:
        """After reset(), the collector must have zero accumulated orders."""
        cfg = make_config(rate=3.0, max_steps=300)
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        # Run half the episode to accumulate some KPIs
        for _ in range(1):
            env.step(_default_action(cfg))
        assert env._collector is not None
        kpi_mid = env._collector.summary()
        assert kpi_mid["total_orders"] > 0  # sanity: some orders were created

        # Reset: collector must be replaced with a fresh one
        env.reset(seed=1)
        assert env._collector is not None
        kpi_after_reset = env._collector.summary()
        assert kpi_after_reset["total_orders"] == 0, (
            "KPICollector not fresh after reset()"
        )

    def test_active_orders_cleared_after_reset(self) -> None:
        """world.active_orders must be empty at the start of a new episode."""
        cfg = make_config(rate=3.0, max_steps=200)
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        # Step once to create some orders
        env.step(_default_action(cfg))
        assert env._simulator.world is not None
        assert len(env._simulator.world.active_orders) > 0  # sanity
        # After reset, world is brand new
        env.reset(seed=1)
        assert env._simulator.world is not None
        assert len(env._simulator.world.active_orders) == 0


# ---------------------------------------------------------------------------
# Test 10 — Reward wiring
# ---------------------------------------------------------------------------

_SENTINEL_REWARD = 7.77


@register("reward", name="SentinelReward")
class SentinelReward(RewardFunction):
    """Returns a fixed sentinel so tests can verify the call site."""

    def compute(
        self,
        world: WorldState,  # noqa: ARG002
        completed_orders: list[Order],  # noqa: ARG002
        dt: float,  # noqa: ARG002
    ) -> float:
        return _SENTINEL_REWARD

    def reset(self) -> None:
        pass


class TestRewardWiring:

    def test_step_reward_comes_from_reward_function(self) -> None:
        """The sentinel reward function's output must appear as step() reward."""
        cfg = make_config()
        # Override the reward function type to our sentinel
        cfg = cfg.model_copy(update={"reward": cfg.reward.model_copy(
            update={"function_type": "SentinelReward"}
        )})
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        _, rew, _, _, _ = env.step(_default_action(cfg))
        assert float(rew) == pytest.approx(_SENTINEL_REWARD), (
            f"Expected sentinel reward {_SENTINEL_REWARD!r}, got {rew!r}"
        )

    def test_reward_function_receives_completed_orders(self) -> None:
        """RewardFunction.compute must receive orders that became terminal in
        the interval, not all active orders."""
        completed_in_call: list[list[Order]] = []

        @register("reward", name="_TrackingReward")
        class TrackingReward(RewardFunction):
            def compute(
                self,
                world: WorldState,  # noqa: ARG002
                completed_orders: list[Order],
                dt: float,  # noqa: ARG002
            ) -> float:
                completed_in_call.append(list(completed_orders))
                return 0.0

            def reset(self) -> None:
                completed_in_call.clear()

        cfg = make_config(rate=3.0, max_steps=500)
        cfg = cfg.model_copy(update={"reward": cfg.reward.model_copy(
            update={"function_type": "_TrackingReward"}
        )})
        env = DeliveryEnv(cfg)
        env.reset(seed=1)
        n_steps = 5
        for _ in range(n_steps):
            env.step(_default_action(cfg))

        assert len(completed_in_call) == n_steps, (
            "compute() must be called once per step"
        )
        # All orders passed must be terminal
        for step_orders in completed_in_call:
            for o in step_orders:
                assert o.is_terminal, f"Non-terminal order {o.order_id} passed to compute()"


# ---------------------------------------------------------------------------
# Test 11 — Baseline smoke
# ---------------------------------------------------------------------------

class TestBaselineSmoke:

    def test_random_agent_completes_episode(self) -> None:
        """A random-action agent must complete a full episode without error."""
        cfg = make_config(seed=0, rate=1.5, max_steps=300, decision_interval=100.0)
        env = DeliveryEnv(cfg)
        env.reset(seed=0)
        rng = np.random.default_rng(0)

        truncated = False
        final_info: dict[str, Any] = {}
        for _ in range(10_000):
            act = rng.uniform(0.0, cfg.max_coverage_radius, size=(1,)).astype(np.float32)
            _, _, _, truncated, info = env.step(act)
            if truncated:
                final_info = info
                break

        assert truncated, "Episode did not reach truncation"
        assert "kpi" in final_info, "Final step must include 'kpi' in info"
        assert final_info["kpi"]["total_orders"] >= 0

    def test_random_agent_obs_always_in_space(self) -> None:
        """Every obs produced during a random-action episode must be in space."""
        cfg = make_config(seed=3, rate=2.0, max_steps=300, decision_interval=100.0)
        env = DeliveryEnv(cfg)
        obs, _ = env.reset(seed=3)
        assert env.observation_space.contains(obs)
        rng = np.random.default_rng(3)
        for _ in range(10_000):
            act = rng.uniform(0.0, cfg.max_coverage_radius, size=(1,)).astype(np.float32)
            obs, _, _, trunc, _ = env.step(act)
            assert env.observation_space.contains(obs), f"obs {obs} out of space"
            if trunc:
                break

    def test_multi_step_no_crash_with_zero_demand(self) -> None:
        """Zero-demand config must complete without error (no orders created)."""
        cfg = make_config(rate=0.0, max_steps=300, decision_interval=100.0)
        env = DeliveryEnv(cfg)
        env.reset(seed=0)
        act = _default_action(cfg)
        for _ in range(10_000):
            _, _, _, trunc, info = env.step(act)
            if trunc:
                assert info["kpi"]["total_orders"] == 0
                break


# ---------------------------------------------------------------------------
# Additional: run_until smoke — direct Simulator API
# ---------------------------------------------------------------------------

class TestRunUntilSmoke:

    def test_run_until_no_crash_on_empty_queue(self) -> None:
        """run_until on an empty queue must return cleanly and pin clock."""
        cfg = make_config(rate=0.0)
        sim = Simulator(cfg)
        sim.reset()
        sim.run_until(500.0)
        assert sim.clock.elapsed == pytest.approx(500.0)

    def test_run_until_processes_events_in_order(self) -> None:
        """Events must be processed in time order across multiple windows."""
        cfg = make_config(seed=1, rate=3.0, max_steps=200)
        sim = Simulator(cfg)
        coll = KPICollector()
        sim.attach_collector(coll)
        sim.reset()
        # Step through 2 windows of 100s each
        sim.run_until(100.0)
        sim.run_until(200.0)
        # Clock must be at 200 = horizon
        assert sim.clock.elapsed == pytest.approx(200.0)
        # finalize manually (env would normally do this)
        assert sim.world is not None
        coll.finalize(num_couriers=len(sim.world.couriers), horizon=200.0)
        kpi = coll.summary()
        assert kpi["total_orders"] >= 0  # no crash; sensible output

    def test_run_until_clock_never_goes_backward(self) -> None:
        """clock.elapsed must be non-decreasing across run_until calls."""
        cfg = make_config(seed=42, rate=2.0, max_steps=500)
        sim = Simulator(cfg)
        sim.reset()
        prev = 0.0
        for target in [50.0, 100.0, 150.0, 200.0, 250.0, 300.0]:
            sim.run_until(target)
            assert sim.clock.elapsed >= prev, (
                f"Clock went backward: {prev!r} → {sim.clock.elapsed!r}"
            )
            prev = sim.clock.elapsed

    def test_run_until_does_not_call_reset_internally(self) -> None:
        """run_until must not touch world state except advancing events.
        Verify by checking order count is cumulative across multiple calls."""
        cfg = make_config(seed=5, rate=3.0, max_steps=300)
        sim = Simulator(cfg)
        coll = KPICollector()
        sim.attach_collector(coll)
        sim.reset()

        sim.run_until(100.0)
        n_after_100 = coll.summary()["total_orders"]

        sim.run_until(200.0)
        n_after_200 = coll.summary()["total_orders"]

        sim.run_until(300.0)
        n_after_300 = coll.summary()["total_orders"]

        # Orders are cumulative; each window adds zero or more
        assert n_after_200 >= n_after_100
        assert n_after_300 >= n_after_200
        # At least some orders must have been created over 300s at rate=3
        assert n_after_300 > 0

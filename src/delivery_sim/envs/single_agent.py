"""
DeliveryEnv — single-agent Gymnasium environment for coverage-radius control.

Layer: Control / RL Interface.

The agent sets per-store ``coverage_radius`` values on a slow (decision_interval)
timescale.  At each env step the chosen radii are written to the live Store
objects; the simulator then advances by ``decision_interval`` sim-seconds via
``Simulator.run_until``, processing every queued event in that window.  Reward
comes from a swappable ``RewardFunction``; coverage-vs-cost tradeoffs live there.

Observation
-----------
The shape and content of the observation vector are determined by the
``ObservationSpec`` selected via ``config.observation_preset``.  The default
preset is ``"standard"``, which reproduces the original hardcoded vector:

  obs[0 .. n-1]   coverage_radius[i] / max_coverage_radius
  obs[n]          interval delivery rate = delivered / (delivered+failed) this
                  step.  Sentinel 0.5 when no orders completed.
  obs[n+1]        interval failed rate   = failed / (delivered+failed) this
                  step.  Sentinel 0.0 when no orders completed.
  obs[n+2]        instantaneous busy-courier fraction = non-free couriers /
                  total couriers at the end of the step window.
  obs[n+3]        episode mean_delivery_time / horizon, clipped [0, 1]
  obs[n+4]        pending order count / max_pending, clipped [0, 1]

The ``observation_space`` is derived from the selected spec's ``bounds()``
method, ensuring shape and bounds always match what ``observe()`` returns.

Action space:  Box(low=0, high=max_coverage_radius, shape=(n_stores,), float32)
  Each element is the desired coverage_radius for the corresponding store
  (matched by config order).  The action is clamped to [0, max_coverage_radius]
  before being written to the live Store object, so coverage immediately affects
  the next interval's demand-coverage decisions.

Termination semantics:
  terminated = False  — no natural terminal condition in the delivery domain
  truncated  = True   — when sim_time reaches horizon (max_steps * dt)

Cadence:
  steps_to_truncation = floor(horizon / decision_interval).
  If horizon is not an integer multiple of decision_interval the last
  run_until window extends past horizon, but only events with time < horizon
  are ever scheduled, so no extra events are processed.  Truncation fires
  as soon as self._sim_time >= horizon.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from delivery_sim.config.schema import ScenarioConfig
from delivery_sim.engine.simulator import Simulator
from delivery_sim.entities.order import OrderStatus
from delivery_sim.envs.observations import (
    ObservationSpec,  # noqa: F401 — triggers preset registration
)
from delivery_sim.metrics.collector import KPICollector
from delivery_sim.registry import create
from delivery_sim.render.headless import HeadlessRenderer
from delivery_sim.rewards.base import RewardFunction


class DeliveryEnv(gym.Env[Any, Any]):
    """Single-agent Gymnasium environment: per-store coverage-radius control.

    One agent sets ``coverage_radius`` for every store at each decision step.
    The simulator advances by ``config.decision_interval`` sim-seconds between
    decisions, processing all events in that window.

    The observation vector shape and bounds are determined by the
    ``ObservationSpec`` named in ``config.observation_preset``.  Default is
    ``"standard"``, which preserves the original hardcoded vector exactly.
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "headless"]}

    def __init__(self, config: ScenarioConfig, render_mode: str = "headless") -> None:
        """Build action/observation spaces and attach simulator components.

        Args:
            config:      Validated scenario configuration.  Determines n_stores,
                         max_coverage_radius, decision_interval, and which
                         ObservationSpec / RewardFunction are instantiated.
            render_mode: ``"headless"`` (default, training) or ``"human"``
                         (reserved for a future pygame renderer).
        """
        super().__init__()
        self.config = config
        self.render_mode = render_mode

        n_stores: int = len(config.stores)
        max_r: float = float(config.max_coverage_radius)

        self.action_space: spaces.Space[Any] = spaces.Box(
            low=np.float32(0.0),
            high=np.float32(max_r),
            shape=(n_stores,),
            dtype=np.float32,
        )

        self._obs_spec: ObservationSpec = create("observation", config.observation_preset)
        _low, _high = self._obs_spec.bounds(n_stores)
        self.observation_space: spaces.Space[Any] = spaces.Box(
            low=_low, high=_high, dtype=np.float32
        )

        self._simulator = Simulator(config)
        self._simulator.attach_renderer(HeadlessRenderer())
        self._reward_fn: RewardFunction = create("reward", config.reward.function_type)

        self._horizon: float = config.max_steps * config.dt
        self._n_stores: int = n_stores
        self._max_r: float = max_r
        self._decision_interval: float = float(config.decision_interval)
        # Normaliser for pending-order count in the observation.
        self._max_pending: float = max(
            1.0, config.demand.rate * config.decision_interval * 2.0
        )

        # Initialised in reset(); guarded in step().
        self._collector: KPICollector | None = None
        self._sim_time: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment; return (observation, info).

        If *seed* is provided it replaces ``config.seed`` for this episode,
        ensuring the (seed, action-sequence) pair fully determines the
        trajectory.  Subsequent calls with ``seed=None`` keep the last seed.

        The ObservationSpec is rebuilt from the registry so that any
        config change (including a seed-driven config swap) takes effect.
        """
        super().reset(seed=seed)

        if seed is not None:
            new_cfg = self.config.model_copy(update={"seed": seed})
            self.config = new_cfg
            self._simulator.config = new_cfg

        # Rebuild spec and space (cheap; guarantees config and space stay in sync).
        self._obs_spec = create("observation", self.config.observation_preset)
        _low, _high = self._obs_spec.bounds(self._n_stores)
        self.observation_space = spaces.Box(low=_low, high=_high, dtype=np.float32)

        self._collector = KPICollector()
        self._simulator.attach_collector(self._collector)
        self._simulator.reset()
        self._reward_fn.reset()
        self._sim_time = 0.0

        obs = self._build_obs(interval_delivered=0, interval_failed=0, interval_total=0)
        return obs, {"sim_time": 0.0}

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply *action*, advance by decision_interval, return 5-tuple.

        Action plumbing (non-negotiable for correct RL training):
          1. Clamp action to [0, max_coverage_radius].
          2. Write clamped[i] to ``world.stores[i].coverage_radius`` (live
             Store object, same order as config.stores).
          3. Call ``simulator.run_until(sim_time + decision_interval)``.
          4. All events in that window see the new coverage_radius when
             ``Store.covers()`` is called — the action is not inert.
        """
        if self._collector is None or self._simulator.world is None:
            raise RuntimeError("call reset() before step()")

        world = self._simulator.world

        # 1+2. Apply action to live store objects.
        clamped = np.clip(
            np.asarray(action, dtype=np.float32), 0.0, self._max_r
        )
        for i, store in enumerate(world.stores):
            store.coverage_radius = float(clamped[i])

        # Snapshot which orders are already terminal before this window.
        before_terminal: frozenset[str] = frozenset(
            oid for oid, o in world.active_orders.items() if o.is_terminal
        )

        # 3. Advance the simulator.
        target = self._sim_time + self._decision_interval
        self._simulator.run_until(target)
        self._sim_time = target

        # Collect orders that became terminal during this interval.
        completed_orders = [
            world.active_orders[oid]
            for oid, o in world.active_orders.items()
            if o.is_terminal and oid not in before_terminal
        ]

        # 4. Compute reward via the injected RewardFunction.
        reward = float(
            self._reward_fn.compute(world, completed_orders, self._decision_interval)
        )

        # Termination: truncated at horizon; no natural terminal condition.
        terminated = False
        truncated = self._sim_time >= self._horizon

        if truncated:
            # Finalise courier-utilisation accounting; only called once.
            self._collector.finalize(
                num_couriers=len(world.couriers),
                horizon=self._horizon,
            )

        n_delivered = sum(
            1 for o in completed_orders if o.status == OrderStatus.DELIVERED
        )
        n_failed = sum(
            1 for o in completed_orders if o.status == OrderStatus.FAILED
        )
        obs = self._build_obs(n_delivered, n_failed, n_delivered + n_failed)

        info: dict[str, Any] = {
            "sim_time": self._sim_time,
            "interval_completed": len(completed_orders),
        }
        if truncated:
            info["kpi"] = self._collector.summary()

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        """Render the current state.

        # TODO(step-6): swap HeadlessRenderer for PygameRenderer when
        # render_mode == "human".
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release renderer resources."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        interval_delivered: int,
        interval_failed: int,
        interval_total: int,
    ) -> np.ndarray:
        """Delegate observation construction to the selected ObservationSpec."""
        assert self._simulator.world is not None
        assert self._collector is not None
        return self._obs_spec.observe(
            self._simulator.world,
            self._collector,
            interval_delivered,
            interval_failed,
            interval_total,
            self._max_r,
            self._max_pending,
            self._horizon,
        )

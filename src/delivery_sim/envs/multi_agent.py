"""
DeliveryParallelEnv — multi-agent PettingZoo environment.

Layer: Control / RL Interface.
"""

from __future__ import annotations

from typing import Any

from gymnasium import spaces
from pettingzoo import ParallelEnv

from delivery_sim.config.schema import ScenarioConfig
from delivery_sim.engine.simulator import Simulator
from delivery_sim.render.headless import HeadlessRenderer


class DeliveryParallelEnv(ParallelEnv):  # type: ignore[misc]
    """Multi-agent PettingZoo parallel environment for the delivery simulator.

    Each courier is an independent agent.  All agents act simultaneously;
    their actions are dispatch decisions (which order to pick up next).

    # TODO(step-5): derive agent IDs from config.couriers, define per-agent
    # observation and action spaces, implement reset and step.
    """

    metadata: dict[str, Any] = {"name": "delivery_parallel_v0"}

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self._simulator = Simulator(config)
        self._simulator.attach_renderer(HeadlessRenderer())

        # TODO(step-5): populate from config.couriers.
        self.possible_agents: list[str] = []
        self.agents: list[str] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment; return (observations, infos).

        # TODO(step-5): call simulator.reset(), build per-agent obs dicts.
        """
        raise NotImplementedError

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Apply per-agent *actions*, advance world, return 5-tuple.

        # TODO(step-5): translate per-courier actions, run simulator tick(s),
        # collect per-agent rewards, build obs, check termination.
        """
        raise NotImplementedError

    def observation_space(self, agent: str) -> spaces.Space[Any]:
        """Return the observation space for *agent*.

        # TODO(step-5): build from world size + order count dimensions.
        """
        raise NotImplementedError

    def action_space(self, agent: str) -> spaces.Space[Any]:
        """Return the action space for *agent*.

        # TODO(step-5): Discrete(n_pending_orders + 1) for assign-or-idle.
        """
        raise NotImplementedError

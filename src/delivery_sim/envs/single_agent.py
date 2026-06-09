"""
DeliveryEnv — single-agent Gymnasium environment.

Layer: Control / RL Interface.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium import spaces

from delivery_sim.config.schema import ScenarioConfig
from delivery_sim.engine.simulator import Simulator
from delivery_sim.render.headless import HeadlessRenderer


class DeliveryEnv(gym.Env[Any, Any]):
    """Single-agent Gymnasium environment wrapping the delivery simulator.

    One agent controls the entire dispatch policy: at each decision step it
    receives a flattened observation of world state and outputs an action
    that assigns pending orders to idle couriers.

    # TODO(step-5): define observation_space and action_space from config,
    # implement reset (call simulator.reset(), build obs), implement step
    # (translate action → dispatch commands, call simulator.step() N times,
    # build next obs, compute reward, check termination).
    """

    metadata: dict[str, Any] = {"render_modes": ["human", "headless"]}

    def __init__(self, config: ScenarioConfig, render_mode: str = "headless") -> None:
        super().__init__()
        self.config = config
        self.render_mode = render_mode
        self._simulator = Simulator(config)
        self._simulator.attach_renderer(HeadlessRenderer())

        # TODO(step-5): derive meaningful spaces from config dimensions.
        self.observation_space: spaces.Space[Any] = spaces.Dict({})
        self.action_space: spaces.Space[Any] = spaces.Discrete(1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and return (observation, info).

        # TODO(step-5): call super().reset(seed=seed), simulator.reset(),
        # build initial observation dict.
        """
        super().reset(seed=seed)
        raise NotImplementedError

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply *action*, advance the world, return (obs, reward, term, trunc, info).

        # TODO(step-5): translate action → dispatch commands, run N simulator
        # ticks, collect reward from RewardFunction, build obs, check truncation.
        """
        raise NotImplementedError

    def render(self) -> Any:
        """Render the current state to the configured mode.

        # TODO(step-6): swap HeadlessRenderer for PygameRenderer when
        # render_mode == "human".
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release renderer resources.

        # TODO(step-6): call self._renderer.close().
        """

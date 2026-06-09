"""
RewardFunction ABC.

Layer: Control / RL Interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delivery_sim.engine.world_state import WorldState
    from delivery_sim.entities.order import Order


class RewardFunction(ABC):
    """Computes a scalar reward signal for the RL agent at each tick.

    Concrete subclasses are registered via ``@register("reward")`` so the
    scenario YAML can name the reward function without importing it.
    """

    @abstractmethod
    def compute(
        self,
        world: WorldState,
        completed_orders: list[Order],
        dt: float,
    ) -> float:
        """Return a scalar reward for the current tick.

        *completed_orders* contains orders that reached a terminal state
        (DELIVERED or FAILED) during this tick.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset any accumulated state at the start of a new episode."""
        raise NotImplementedError

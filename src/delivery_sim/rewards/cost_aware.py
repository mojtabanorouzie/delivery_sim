"""
CostAwareReward — delivery reward penalising real per-interval delivery cost.

Layer: Control / RL Interface.

Reward formula per step
-----------------------
  reward = delivery_reward × n_delivered
         − w_fail          × n_failed
         − w_cost          × Σ order.delivery_cost  (over orders terminal this step)

``order.delivery_cost`` is the real courier-distance cost (leg1 + leg2, both
computed as distance × cost_per_unit) set by the Simulator at the exact moment
the order reaches a terminal state.  FAILED orders always have delivery_cost 0.0
(an uncovered order is never dispatched, so no courier cost is incurred).

Coverage-vs-cost tradeoff
-------------------------
  Large radius → more orders accepted → more deliveries → higher delivery reward.
  Large radius → longer courier routes → higher per-delivery cost → lower net reward.
  The agent must find the radius that maximises the net signal.

Default weights
---------------
  delivery_reward = 1.0   full credit per delivered order
  w_fail          = 0.5   penalty per failed (coverage-miss) order
  w_cost          = 0.01  cost coefficient; scale so Σcost is same order as
                          n_delivered over a typical step (≈ 1-5 orders, route
                          distances ≈ 200-700 units, cost_per_unit ≈ 0.01 →
                          typical step cost ≈ 0.5-3.5, well below reward signal)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from delivery_sim.entities.order import OrderStatus
from delivery_sim.registry import register
from delivery_sim.rewards.base import RewardFunction

if TYPE_CHECKING:
    from delivery_sim.engine.world_state import WorldState
    from delivery_sim.entities.order import Order


@register("reward")
@register("reward", name="cost_aware")
class CostAwareReward(RewardFunction):
    """Per-step reward using real courier-distance delivery cost.

    Parameters
    ----------
    delivery_reward : Credit per successfully delivered order this step.
    w_fail          : Penalty per order that reached FAILED this step.
    w_cost          : Cost coefficient applied to Σ(order.delivery_cost) over
                      orders that became terminal this step.
    """

    def __init__(
        self,
        delivery_reward: float = 1.0,
        w_fail: float = 0.5,
        w_cost: float = 0.01,
    ) -> None:
        """See class docstring for parameter descriptions."""
        self.delivery_reward = delivery_reward
        self.w_fail = w_fail
        self.w_cost = w_cost

    def compute(
        self,
        world: WorldState,  # noqa: ARG002
        completed_orders: list[Order],
        dt: float,  # noqa: ARG002
    ) -> float:
        """Return the per-step scalar reward.

        Args:
            world:            Current world state (unused by this reward).
            completed_orders: Orders that reached DELIVERED or FAILED this step.
            dt:               Step duration in sim-seconds (unused here).

        Returns:
            ``delivery_reward × n_delivered − w_fail × n_failed
            − w_cost × Σ(order.delivery_cost)``.
            Returns 0.0 when *completed_orders* is empty.
        """
        n_delivered = 0
        n_failed = 0
        total_cost = 0.0
        for order in completed_orders:
            if order.status == OrderStatus.DELIVERED:
                n_delivered += 1
                total_cost += order.delivery_cost
            elif order.status == OrderStatus.FAILED:
                n_failed += 1
                # delivery_cost is 0.0 for failed orders (never dispatched)

        return (
            self.delivery_reward * n_delivered
            - self.w_fail * n_failed
            - self.w_cost * total_cost
        )

    def reset(self) -> None:
        """No-op: this reward function carries no episode state."""

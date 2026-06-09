"""
LatencyAwareReward — delivery reward with time-decay and real delivery cost.

Layer: Control / RL Interface.

Reward formula per step
-----------------------
  delivery_value(t) = max(0, 1 - max(0, t - target_time) / grace_period)
      Full +1.0 when delivered within target_time.
      Decays linearly to 0 at target_time + grace_period ("customer cancels").

  step_reward =
      sum(delivery_value(t) for each DELIVERED order this step)
    - failure_penalty  * n_failed
    - w_cost           * Σ(order.delivery_cost) over DELIVERED orders this step

``order.delivery_cost`` is the real leg1+leg2 courier-distance cost set by the
Simulator at delivery time (distance × cost_per_unit).  It is never a function
of coverage_radius.

Trade-off surface
-----------------
  Radius too large → far customers accepted → long travel
      → latency decay reduces delivery_value (time signal)
      → longer routes raise order.delivery_cost (cost signal)
  Radius too small → near customers only → coverage misses → failure penalty
  Agent must find the radius that maximises net reward.
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
class LatencyAwareReward(RewardFunction):
    """Per-step reward with time-decayed delivery value and real delivery cost.

    Parameters
    ----------
    target_time     : Delivery time (created→delivered) that earns full value.
    grace_period    : Extra seconds after target_time before value reaches 0.
    failure_penalty : Subtracted for each FAILED order this step.
    w_cost          : Applied to Σ(order.delivery_cost) over DELIVERED orders
                      this step.  order.delivery_cost is the real courier-
                      distance cost (leg1 + leg2); it is never a function of
                      coverage_radius.
    """

    def __init__(
        self,
        target_time: float = 600.0,
        grace_period: float = 600.0,
        failure_penalty: float = 0.5,
        w_cost: float = 0.01,
    ) -> None:
        self.target_time = target_time
        self.grace_period = grace_period
        self.failure_penalty = failure_penalty
        self.w_cost = w_cost

    def compute(
        self,
        world: WorldState,  # noqa: ARG002
        completed_orders: list[Order],
        dt: float,  # noqa: ARG002
    ) -> float:
        """Return the per-step scalar reward.

        Args:
            world:            Current world state (unused — no proxy terms).
            completed_orders: Orders that reached DELIVERED or FAILED this step.
            dt:               Step duration in sim-seconds (unused here).
        """
        reward = 0.0
        total_cost = 0.0

        for order in completed_orders:
            if order.status == OrderStatus.DELIVERED:
                created_t = order.timestamps.get(OrderStatus.CREATED, 0.0)
                delivered_t = order.timestamps.get(OrderStatus.DELIVERED, 0.0)
                elapsed = delivered_t - created_t
                overshoot = max(0.0, elapsed - self.target_time)
                reward += max(0.0, 1.0 - overshoot / self.grace_period)
                total_cost += order.delivery_cost
            elif order.status == OrderStatus.FAILED:
                reward -= self.failure_penalty

        reward -= self.w_cost * total_cost
        return reward

    def reset(self) -> None:
        """No-op: this reward function carries no episode state."""

"""
OptimizedDeliveryReward — tuned reward for coverage-radius RL agents.

Layer: Control / RL Interface.

Design rationale vs SparseDeliveryReward
-----------------------------------------
1. Failure penalty raised to 1.0 (from 0.5).
   A coverage miss that immediately fails an order is worse than a slow
   delivery.  Stronger signal drives the agent away from dangerously small
   radii.

2. Real delivery cost term added (w_cost × Σ order.delivery_cost this step).
   order.delivery_cost is the real leg1+leg2 courier-distance cost.  Large
   radius → far customers accepted → longer routes → higher real cost →
   lower reward.  The tradeoff is grounded in the actual cost incurred, not
   in coverage_radius (the action itself).  Default w_cost=0.01: at
   cost_per_unit=0.01, typical per-delivery cost ≈ 2–7 → penalty ≈ 0.02–0.07
   against delivery_value ≈ 0.5–1.0 → a mild nudge.

3. Time-decay target recalibrated to 500s / grace 700s.
   Typical delivery time with 3 couriers, speed 1.5, world 1000×1000 is
   680–700 s.  target=600 penalises almost every delivery equally.
   target=500 creates a real gradient: fast deliveries (short routes from
   small-radius coverage) earn near-full +1; slow ones (long routes from
   large-radius coverage) earn 0.5–0.8.  This amplifies the cost-and-latency
   signal the agent must trade off against coverage-miss penalties.
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
class OptimizedDeliveryReward(RewardFunction):
    """Coverage-aware reward with calibrated latency decay and real delivery cost.

    step_reward =
        sum over DELIVERED: max(0, 1 - max(0, elapsed - target_time) / grace_period)
      - failure_penalty  * n_failed
      - w_cost           * Σ(order.delivery_cost) over DELIVERED orders this step

    Parameters
    ----------
    target_time     : Delivery time earning full latency value (seconds).
    grace_period    : Seconds beyond target_time before value reaches 0.
    failure_penalty : Penalty per FAILED order this step.
    w_cost          : Applied to Σ(order.delivery_cost) over DELIVERED orders
                      this step.  order.delivery_cost is the real courier-
                      distance cost (leg1 + leg2); it is never a function of
                      coverage_radius.
    """

    def __init__(
        self,
        target_time: float = 500.0,
        grace_period: float = 700.0,
        failure_penalty: float = 1.0,
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

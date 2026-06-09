"""
SparseDeliveryReward — placeholder reward function.

Layer: Control / RL Interface.

Granularity: per-step (tick).
``compute()`` is called once per tick by the RL env (step 5) with the list of
orders that reached a terminal state during that tick.  The Simulator does NOT
call this; the env wrapper drives it.

Reward formula (baseline):
    reward = +1.0 × n_delivered  −  0.5 × n_failed

This is a researcher's placeholder.  Cost-weighted and latency-penalised
variants should subclass ``RewardFunction`` and inject a ``KPICollector``
reference to read accumulated cost/latency KPIs without recomputing them.
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
class SparseDeliveryReward(RewardFunction):
    """Per-step reward: +1 per delivery, −0.5 per failed order this tick.

    Stateless — ``reset()`` is a no-op.  Registered under the name
    ``"SparseDeliveryReward"`` so scenario YAMLs can select it via
    ``reward.function_type``.

    Reward shaping (cost penalties, latency weights, SLA bonuses) is the
    researcher's variable.  Read ``KPICollector.summary()`` for the full
    episode KPI surface rather than recomputing metrics here.
    """

    #: Reward credited per successfully delivered order this tick.
    DELIVERY_REWARD: float = 1.0
    #: Penalty applied per order that reached FAILED this tick.
    FAILURE_PENALTY: float = 0.5

    def compute(
        self,
        world: WorldState,  # noqa: ARG002
        completed_orders: list[Order],
        dt: float,  # noqa: ARG002
    ) -> float:
        """Return the per-tick scalar reward.

        Args:
            world:            Current world state (unused by this baseline).
            completed_orders: Orders that transitioned to DELIVERED or FAILED
                              during this tick.
            dt:               Tick duration in seconds (unused by this baseline).

        Returns:
            ``DELIVERY_REWARD × n_delivered − FAILURE_PENALTY × n_failed``.
            Returns 0.0 when *completed_orders* is empty.
        """
        n_delivered = sum(
            1 for o in completed_orders if o.status == OrderStatus.DELIVERED
        )
        n_failed = sum(
            1 for o in completed_orders if o.status == OrderStatus.FAILED
        )
        return self.DELIVERY_REWARD * n_delivered - self.FAILURE_PENALTY * n_failed

    def reset(self) -> None:
        """No-op: this reward function carries no episode state."""

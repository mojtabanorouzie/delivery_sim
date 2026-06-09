"""Rewards module: RewardFunction ABC and built-in implementations."""

from __future__ import annotations

from delivery_sim.rewards.base import RewardFunction
from delivery_sim.rewards.cost_aware import CostAwareReward
from delivery_sim.rewards.latency_reward import LatencyAwareReward
from delivery_sim.rewards.optimized_reward import OptimizedDeliveryReward
from delivery_sim.rewards.placeholder import SparseDeliveryReward

__all__ = [
    "RewardFunction",
    "CostAwareReward",
    "LatencyAwareReward",
    "OptimizedDeliveryReward",
    "SparseDeliveryReward",
]

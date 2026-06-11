# Extending the RL Interface: Custom Observations and Rewards

> **Superseded.** This file has been replaced by [extending.md](extending.md),
> which covers all six extension points: ObservationSpec, RewardFunction, Store,
> Courier, DemandGenerator, and RoutingModel.

Adding a new observation layout or reward signal is a **one-file change**.
You register a class, name it in config, and the env picks it up automatically —
no edits to `DeliveryEnv` or any other env file.

---

## How the registry works

Both `ObservationSpec` and `RewardFunction` are registered under named categories
in the shared registry (`delivery_sim.registry`).  The env calls
`create("observation", config.observation_preset)` and
`create("reward", config.reward.function_type)` at reset time.  Any class
registered under those names is a valid plug-in.

---

## 1. Custom ObservationSpec

```python
# my_project/my_observations.py
import numpy as np
from delivery_sim.envs.observations import ObservationSpec
from delivery_sim.registry import register

@register("observation", name="my_risk_obs")
class RiskObservation(ObservationSpec):
    """
    Three-feature vector: normalised coverage per store + interval failed rate.

    Vector layout (n + 1 features):
      obs[0..n-1]  coverage_radius[i] / max_r
      obs[n]       interval failed rate (sentinel 0.0 when no orders)
    """

    def observe(
        self,
        world,
        collector,
        interval_delivered,
        interval_failed,
        interval_total,
        max_r,
        max_pending,
        horizon,
    ) -> np.ndarray:
        coverage = np.clip(
            np.array(
                [s.coverage_radius / max_r for s in world.stores],
                dtype=np.float32,
            ),
            0.0, 1.0,
        )
        failed_rate = (
            float(interval_failed) / interval_total
            if interval_total > 0
            else 0.0
        )
        return np.append(coverage, np.float32(failed_rate)).astype(np.float32)

    def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
        n = n_stores + 1
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
```

Select it in config:

```yaml
# scenarios/my_scenario.yaml
observation_preset: my_risk_obs
```

Or in Python:

```python
from delivery_sim.config.schema import ScenarioConfig
import my_project.my_observations  # import triggers @register

config = ScenarioConfig(
    ...,
    observation_preset="my_risk_obs",
)
env = DeliveryEnv(config)
```

---

## 2. Custom RewardFunction

```python
# my_project/my_rewards.py
from delivery_sim.rewards.base import RewardFunction
from delivery_sim.entities.order import OrderStatus
from delivery_sim.registry import register

@register("reward", name="my_sla_reward")
class SLAReward(RewardFunction):
    """
    +1 per delivery within SLA, 0 otherwise, -1 per failure.
    Uses order.delivery_cost to add a small cost penalty.
    """

    def __init__(self, sla_seconds: float = 900.0, w_cost: float = 0.005) -> None:
        self.sla_seconds = sla_seconds
        self.w_cost = w_cost

    def compute(self, world, completed_orders, dt) -> float:
        reward = 0.0
        for order in completed_orders:
            if order.status == OrderStatus.DELIVERED:
                created = order.timestamps.get(OrderStatus.CREATED, 0.0)
                delivered = order.timestamps.get(OrderStatus.DELIVERED, 0.0)
                within_sla = (delivered - created) <= self.sla_seconds
                reward += 1.0 if within_sla else 0.0
                reward -= self.w_cost * order.delivery_cost
            elif order.status == OrderStatus.FAILED:
                reward -= 1.0
        return reward

    def reset(self) -> None:
        pass  # no episode state
```

Select it:

```yaml
reward:
  function_type: my_sla_reward
```

---

## 3. Using both together

```python
import my_project.my_observations  # triggers registration
import my_project.my_rewards        # triggers registration

from delivery_sim.config.schema import ScenarioConfig, RewardConfig
from delivery_sim.envs.single_agent import DeliveryEnv

config = ScenarioConfig(
    name="custom_agent",
    ...,
    observation_preset="my_risk_obs",
    reward=RewardConfig(function_type="my_sla_reward"),
)
env = DeliveryEnv(config)
obs, _ = env.reset(seed=42)
# obs.shape == (n_stores + 1,) — from RiskObservation
```

No edits to `DeliveryEnv`, `single_agent.py`, or any other env file.

---

## 4. Built-in presets for reference

| Preset name | Category | Features | Description |
| --- | --- | --- | --- |
| `"minimal"` | observation | n + 1 | coverage + delivery rate |
| `"standard"` | observation | n + 5 | coverage + rates + busy + mean_dt + pending (default) |
| `"operational"` | observation | n + 3 | coverage + failed rate + busy + pending |
| `"SparseDeliveryReward"` / `"sparse_delivery"` | reward | — | +1 delivered − 0.5 failed (default) |
| `"CostAwareReward"` / `"cost_aware"` | reward | — | delivered − w_fail·failed − w_cost·real_cost |
| `"LatencyAwareReward"` | reward | — | time-decayed delivery value + failure penalty |
| `"OptimizedDeliveryReward"` | reward | — | calibrated latency + failure signal |

All presets compose existing `KPICollector` metrics; no new simulation
measurements are added.

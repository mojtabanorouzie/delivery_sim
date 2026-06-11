# Extending delivery_sim

Every pluggable component — observations, rewards, stores, couriers, demand
generators, and routing models — uses the same `@register` / `create` pattern.
You write one class, decorate it, import it once, and the env picks it up
automatically. No edits to `DeliveryEnv` or any engine file.

See [ARCHITECTURE.md](../ARCHITECTURE.md#registry--extensibility-pattern) for the
registry design.

---

## How the registry works

```python
from delivery_sim.registry import register, create

@register("observation", name="my_preset")
class MyObservation(ObservationSpec): ...

# The env calls this at reset time — or you can call it directly:
spec = create("observation", "my_preset")
```

If you omit `name=`, the class name is used as the key
(`@register("courier")` on `class MyCourier` → key `"MyCourier"`).

The import that triggers `@register` must happen before the env is constructed.
In a training script, a top-level `import my_project.my_components` is enough.

---

## 1. Custom ObservationSpec

```python
# my_project/observations.py
import numpy as np
from delivery_sim.envs.observations import ObservationSpec
from delivery_sim.registry import register

@register("observation", name="risk_obs")
class RiskObservation(ObservationSpec):
    """n + 1 vector: normalised coverage per store + interval failed rate."""

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
            np.array([s.coverage_radius / max_r for s in world.stores], dtype=np.float32),
            0.0, 1.0,
        )
        failed_rate = float(interval_failed) / interval_total if interval_total > 0 else 0.0
        return np.append(coverage, np.float32(failed_rate)).astype(np.float32)

    def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
        n = n_stores + 1
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
```

The `observe` signature must match `ObservationSpec.observe` exactly. The returned
array length must equal `len(bounds(n_stores)[0])`.

Select in config:

```yaml
# scenarios/my_scenario.yaml
observation_preset: risk_obs
```

Or in Python:

```python
import my_project.observations  # triggers @register
from delivery_sim.config.schema import ScenarioConfig
from delivery_sim.envs.single_agent import DeliveryEnv

config = ScenarioConfig(..., observation_preset="risk_obs")
env = DeliveryEnv(config)
```

Built-in presets for reference:

| Name | Features | Description |
| --- | --- | --- |
| `"minimal"` | n + 1 | Coverage per store + interval delivery rate |
| `"standard"` | n + 5 | Coverage + rates + busy fraction + mean delivery time + pending |
| `"operational"` | n + 3 | Coverage + failed rate + busy fraction + pending |

---

## 2. Custom RewardFunction

```python
# my_project/rewards.py
from delivery_sim.rewards.base import RewardFunction
from delivery_sim.entities.order import OrderStatus
from delivery_sim.registry import register

@register("reward", name="sla_reward")
class SLAReward(RewardFunction):
    """+1 per delivery within SLA, 0 otherwise, −1 per failure."""

    def __init__(self, sla_seconds: float = 900.0, w_cost: float = 0.005) -> None:
        self.sla_seconds = sla_seconds
        self.w_cost = w_cost

    def compute(self, world, completed_orders, dt) -> float:
        reward = 0.0
        for order in completed_orders:
            if order.status == OrderStatus.DELIVERED:
                elapsed = (
                    order.timestamps.get(OrderStatus.DELIVERED, 0.0)
                    - order.timestamps.get(OrderStatus.CREATED, 0.0)
                )
                reward += 1.0 if elapsed <= self.sla_seconds else 0.0
                reward -= self.w_cost * order.delivery_cost
            elif order.status == OrderStatus.FAILED:
                reward -= 1.0
        return reward

    def reset(self) -> None:
        pass  # no episode state
```

`completed_orders` contains every `Order` that reached `DELIVERED` or `FAILED`
during this decision interval. `order.delivery_cost` is the real courier-distance
cost (leg1 + leg2, set by the Simulator at delivery time).

Select in config:

```yaml
reward:
  function_type: sla_reward
```

Built-in rewards for reference:

| Name | Description |
| --- | --- |
| `"SparseDeliveryReward"` / `"sparse_delivery"` | +1 delivered − 0.5 failed (default) |
| `"CostAwareReward"` / `"cost_aware"` | delivered − w_fail·failed − w_cost·real_cost |

> **Note:** `LatencyAwareReward` and `OptimizedDeliveryReward` exist in the codebase
> but are not listed as recommended here. They use `order.delivery_cost` (a function
> of actual travel distance, not `coverage_radius`) as their cost signal, which
> produces an indirect gradient from action to reward. They may work in specific
> scenarios but have not been validated against the balanced benchmark. Use at your
> own risk and verify via `scripts/diagnose_scenario.py`.

---

## 3. Custom Store

```python
# my_project/stores.py
from delivery_sim.entities.store import Store
from delivery_sim.routing.base import RoutingModel
from delivery_sim.registry import register

@register("store", name="fast_store")
class FastStore(Store):
    """Store with configurable coverage radius and instant preparation."""

    def __init__(
        self,
        store_id: str,
        x: float,
        y: float,
        coverage_radius: float = 500.0,
    ) -> None:
        self._store_id = store_id
        self._x = x
        self._y = y
        self.coverage_radius = coverage_radius

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    def covers(self, point_x: float, point_y: float, routing: RoutingModel) -> bool:
        return routing.distance(self._x, self._y, point_x, point_y) <= self.coverage_radius

    def can_prepare(self, order_id: str) -> bool:
        return True  # unlimited capacity

    def start_preparation(self, order_id: str, sim_time: float) -> float:
        return sim_time  # ready instantly

    def reset(self) -> None:
        pass  # no preparation state
```

`covers()` receives the active `RoutingModel` — use it to compute distances so your
store is routing-model-agnostic (no hardcoded Euclidean math). The boundary is
inclusive: a customer at exactly `coverage_radius` distance is covered.

`start_preparation()` returns the absolute sim time when the order will be ready for
pickup. The Simulator schedules an `ORDER_READY` event at that time.

Select in config:

```yaml
stores:
  - name: my_fast_store
    store_type: fast_store
    x: 400.0
    y: 400.0
    coverage_radius: 600.0
```

---

## 4. Custom DemandGenerator

```python
# my_project/demand.py
from __future__ import annotations
from typing import Any
import numpy as np
from delivery_sim.entities.demand_generator import DemandGenerator
from delivery_sim.registry import register

@register("demand_generator", name="hotspot_demand")
class HotspotDemandGenerator(DemandGenerator):
    """Poisson arrivals concentrated in a circular hotspot."""

    def __init__(
        self,
        rate: float,
        dt: float,
        hotspot_x: float,
        hotspot_y: float,
        hotspot_radius: float,
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        store_ids: list[str] | None = None,
    ) -> None:
        self.rate = rate
        self.dt = dt
        self.hotspot_x = hotspot_x
        self.hotspot_y = hotspot_y
        self.hotspot_radius = hotspot_radius
        self.world_width = world_width
        self.world_height = world_height
        self.store_ids: list[str] = store_ids or []
        self._next_arrival: float | None = None

    def generate(self, sim_time: float, rng: np.random.Generator) -> list[dict[str, Any]]:
        n = int(rng.poisson(self.rate * self.dt))
        orders = []
        for _ in range(n):
            angle = rng.uniform(0.0, 2 * np.pi)
            r = rng.uniform(0.0, self.hotspot_radius)
            orders.append({
                "customer_x": float(np.clip(self.hotspot_x + r * np.cos(angle), 0, self.world_width)),
                "customer_y": float(np.clip(self.hotspot_y + r * np.sin(angle), 0, self.world_height)),
            })
        return orders

    def next_event(
        self, sim_time: float, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]] | None:
        if self.rate == 0.0:
            return None
        inter_arrival = rng.exponential(1.0 / self.rate)  # draw 1: timing
        arrival_time = sim_time + inter_arrival
        angle = rng.uniform(0.0, 2 * np.pi)              # draw 2: location
        r = rng.uniform(0.0, self.hotspot_radius)
        return arrival_time, {
            "customer_x": float(np.clip(self.hotspot_x + r * np.cos(angle), 0, self.world_width)),
            "customer_y": float(np.clip(self.hotspot_y + r * np.sin(angle), 0, self.world_height)),
        }

    def reset(self, rng: np.random.Generator) -> None:
        self._next_arrival = None
```

**Critical:** The draw order within `next_event` (timing first, location second) must
be stable across calls to preserve seed-compatibility. Never reorder RNG draws once
results have been recorded. `order_attrs` must contain `customer_x` and `customer_y`
only — no `store_id`; the Simulator assigns the store via `Store.covers()`.

Select in config:

```yaml
demand:
  generator_type: hotspot_demand
  rate: 0.03
  dt: 1.0
  hotspot_x: 500.0
  hotspot_y: 500.0
  hotspot_radius: 200.0
```

---

## 5. Custom RoutingModel

```python
# my_project/routing.py
from delivery_sim.routing.base import RoutingModel
from delivery_sim.registry import register

@register("routing", name="manhattan")
class ManhattanRouting(RoutingModel):
    """L1 (taxicab) distance routing."""

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return abs(x2 - x1) + abs(y2 - y1)

    def travel_time(
        self, x1: float, y1: float, x2: float, y2: float, speed: float
    ) -> float:
        return self.distance(x1, y1, x2, y2) / speed

    def route(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> list[tuple[float, float]]:
        # L-shaped path: move along x first, then y.
        return [(x1, y1), (x2, y1), (x2, y2)]
```

`route()` must return a list of `(x, y)` waypoints where the first element is
`(x1, y1)` and the last is `(x2, y2)`. The engine uses `travel_time()` for
scheduling; `route()` is used by renderers only.

Select in config:

```yaml
routing:
  model_type: manhattan
```

---

## 6. Custom Courier type

The simplest and most common customisation is to subclass `BikeCourier` and override
speed or cost, while inheriting the trajectory logic:

```python
# my_project/couriers.py
from delivery_sim.entities.courier import BikeCourier
from delivery_sim.registry import register

@register("courier", name="express_bike")
class ExpressBikeCourier(BikeCourier):
    """Faster, pricier courier: 3.0 world-units/s, 0.03/unit cost."""

    def __init__(self, courier_id: str, x: float, y: float, routing, **kwargs):
        super().__init__(
            courier_id, x, y, routing,
            speed=3.0, cost_per_unit=0.03,
            **kwargs,
        )
```

If you need a fully custom trajectory model (e.g. a drone that can hover), implement
`Courier` from scratch. The key invariants to preserve:

- `position_at(t)` must be a **pure read-only query** — no mutations.
- `assign(order_id, store_id, sim_time, target_x, target_y, from_x, from_y)` must
  raise `ValueError` if called while the courier is still in motion
  (`sim_time < arrival_time()`).
- `reset(origin_x, origin_y)` is called at episode start only; clear all trajectory
  state and return to `CourierStatus.IDLE`.

Select in config:

```yaml
couriers:
  - courier_type: express_bike
    count: 5
    speed: 3.0           # overridden in __init__; field ignored when subclass fixes it
    cost_per_unit: 0.03
```

---

## 7. Using multiple custom components together

```python
import my_project.observations   # triggers all @register calls
import my_project.rewards
import my_project.routing

from delivery_sim import load_scenario, DeliveryEnv
from delivery_sim.config.schema import ScenarioConfig, RewardConfig, RoutingConfig

config = ScenarioConfig(
    name="my_experiment",
    observation_preset="risk_obs",
    reward=RewardConfig(function_type="sla_reward"),
    routing=RoutingConfig(model_type="manhattan"),
    # ... stores, couriers, demand ...
)
env = DeliveryEnv(config)
obs, _ = env.reset(seed=0)
```

No edits to `DeliveryEnv`, `Simulator`, or any other env file.

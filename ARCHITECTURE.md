# Architecture

## Four-layer design

```text
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Researcher experiment                    │
│  User's agent / policy + their RL library.          │
│  NOT our code.                                      │
└──────────────────────┬──────────────────────────────┘
                       │  gymnasium API
┌──────────────────────▼──────────────────────────────┐
│  Layer 2 — Control / RL Interface                   │
│  DeliveryEnv (gymnasium)                            │
│  RewardFunction, KPICollector                       │
│  Modules: envs/, rewards/, metrics/                 │
└──────────────────────┬──────────────────────────────┘
                       │  domain objects
┌──────────────────────▼──────────────────────────────┐
│  Layer 3 — Domain Entities (pluggable)              │
│  Store, Courier, Order, DemandGenerator,            │
│  RoutingModel — all ABCs, registered by name.       │
│  Modules: entities/, routing/                       │
└──────────────────────┬──────────────────────────────┘
                       │  tick / events
┌──────────────────────▼──────────────────────────────┐
│  Layer 4 — Simulation Engine                        │
│  SimClock, EventQueue, WorldState, Simulator        │
│  Modules: engine/                                   │
└─────────────────────────────────────────────────────┘
```

**Cross-cutting:**

- `config/` — Pydantic v2 schemas + `load_scenario()`; the reproducibility unit is
  one `(ScenarioConfig, seed)` pair.
- `render/` — pure consumer of engine snapshots; `HeadlessRenderer` is the default
  (training never touches a display).
- `registry.py` — `@register` / `create` factory for all pluggable types.

---

## Event-driven engine

The core design decision: the simulation clock advances to **exact event timestamps**
rather than marching in fixed dt increments. See [ADR-001](docs/adr/ADR-001.md) for
the full rationale. In summary:

- `SimClock.advance_to(t)` jumps to time `t`; it never interpolates.
- `dt` remains as the *observer/snapshot cadence*, not the simulation resolution.
- The `Simulator` drains `EventQueue` until the queue is empty or
  `sim_time >= horizon`; each event is processed at its exact floating-point time.
- Episodes with sparse demand (few events) run much faster than a fixed-dt loop
  would; KPI measurements are never dt-dependent.

**Event types dispatched by `Simulator`:**

| Event | Priority | Meaning |
| --- | --- | --- |
| `ORDER_CREATED` | 10 | Demand generator fires a new order |
| `COURIER_ARRIVED_STORE` | 5 | Courier reaches the origin store |
| `ORDER_READY` | 5 | Store finishes preparing an order |
| `COURIER_ARRIVED_CUSTOMER` | 5 | Courier completes delivery |

Lower priority integer = higher urgency at the same timestamp. A `seq` counter
ensures a total order even when time and priority match (FIFO within a priority
level, so determinism holds regardless of event payload type).

---

## Courier trajectories

Couriers are **time-parameterised trajectories**, not stateful steppers. See
[ADR-002](docs/adr/ADR-002.md). Key consequences:

- `courier.assign(order_id, store_id, sim_time, from_x, from_y, target_x, target_y)`
  establishes a fully-determined leg; the Simulator supplies both endpoints explicitly.
- `courier.position_at(t)` is a pure read-only analytic query — any renderer or
  inspector can call it at any time without mutating state.
- `courier.arrival_time()` is pre-computed at `assign()` time via
  `routing.travel_time()`.
- `courier.reset()` is an episode-boundary call only; it is not a per-step mutation.

---

## Reproducibility guarantee

One `(ScenarioConfig, seed)` pair fully determines an episode:

1. A single `numpy.random.Generator` is created from `config.seed` at `reset()`.
2. Every stochastic call (demand arrivals, customer locations, store assignment)
   draws from this one generator in a fixed sequence.
3. No global state; no external randomness.

The same config + seed always produces byte-identical KPI outputs.

---

## Observer / headless split

The `SnapshotConsumer` protocol (`render/protocol.py`) is the only interface between
the engine and any renderer. The Simulator calls `consumer.on_snapshot(snapshot)`
after each `dt` boundary, passing an **immutable** `WorldSnapshot` built from
`position_at(elapsed)` — purely read-only calls that do not affect sim state.

`HeadlessRenderer` implements `SnapshotConsumer` as a no-op and is the default in
all training paths. A display-capable renderer can be swapped in by calling
`simulator.attach_renderer(my_renderer)` without touching any engine code.

---

## Order state machine

Orders progress through a strict state machine enforced by `Order.transition()`:

```text
CREATED → ASSIGNED → PREPARING → PICKED_UP → IN_TRANSIT → DELIVERED
                                                          ↗
                                            (any state) → FAILED
```

`ALLOWED_TRANSITIONS` maps each `OrderStatus` to the set of valid next states.
Any illegal transition raises `ValueError` immediately — there is no silent state
corruption. `Order.is_terminal` is `True` for `DELIVERED` and `FAILED`.

---

## Registry / extensibility pattern

All pluggable types use a shared string-keyed registry:

```python
from delivery_sim.registry import register, create

@register("reward", name="my_reward")
class MyReward(RewardFunction): ...

# later:
reward_fn = create("reward", "my_reward")
```

Selecting a plugin in a scenario YAML requires no Python imports by the user
(the import that triggers `@register` is the only requirement). The env calls
`create("observation", config.observation_preset)` and
`create("reward", config.reward.function_type)` at reset time.

Registered categories and their built-in names:

| Category | Built-in names |
| --- | --- |
| `"store"` | `"BuiltinStore"` |
| `"courier"` | `"BikeCourier"` |
| `"demand_generator"` | `"PoissonDemandGenerator"` |
| `"routing"` | `"euclidean"` |
| `"observation"` | `"minimal"`, `"standard"`, `"operational"` |
| `"reward"` | `"SparseDeliveryReward"` / `"sparse_delivery"`, `"CostAwareReward"` / `"cost_aware"` |

See [docs/extending.md](docs/extending.md) for the full guide to adding new types.

---

## KPI collection

`KPICollector` is notified at the **exact simulation timestamp** of each order/courier
event — never at snapshot boundaries — so aggregated values (delivery rate, latency
percentiles, utilisation) are dt-independent. `KPICollector.finalize()` must be
called at episode end before reading `summary()`.

The env calls `finalize()` inside `step()` on the truncation step; the summary dict
is included in the `info` return value.

---

## Scenario files

Two scenarios ship with the repo:

| File | Purpose |
| --- | --- |
| `scenarios/example.yaml` | Minimal demo (1 000 steps, 3 couriers, demand=0.5). Good for API exploration; too sparse for RL training. |
| `scenarios/balanced.yaml` | RL training scenario (4 000 steps, 20 couriers, demand=0.025). Calibrated so demand/throughput ≈ 0.92× at r=700; agent has a clear interior optimum to find. |

---

## Roadmap — not yet implemented

The following items are **planned but not yet built**. They appear in stub files or
optional dependencies but raise `NotImplementedError` or are absent entirely.

| Item | Status |
| --- | --- |
| `DeliveryParallelEnv` | File exists (`envs/multi_agent.py`); `reset()`, `step()`, `observation_space()`, `action_space()` all raise `NotImplementedError`. PettingZoo multi-agent interface not implemented. |
| `PygameRenderer` | Listed under `[render]` optional dep; no implementation file. `HeadlessRenderer` is the only renderer. |
| Additional courier types (Drone, Box) | Only `BikeCourier` implemented. |
| Road-graph routing | Only `EuclideanRouting` implemented. |
| Dispatch scheduler | No dispatch / order-assignment layer; the Simulator assigns greedily. |

# Architecture

## Four-layer design (top to bottom)

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Researcher experiment                    │
│  User's agent / policy + their RL library.          │
│  NOT our code.                                      │
└──────────────────────┬──────────────────────────────┘
                       │  gymnasium / pettingzoo API
┌──────────────────────▼──────────────────────────────┐
│  Layer 2 — Control / RL Interface                   │
│  DeliveryEnv (gymnasium), DeliveryParallelEnv       │
│  (pettingzoo), RewardFunction, KPICollector         │
│  Modules: envs/, rewards/, metrics/                 │
└──────────────────────┬──────────────────────────────┘
                       │  domain objects
┌──────────────────────▼──────────────────────────────┐
│  Layer 3 — Domain Entities (pluggable)              │
│  Store, DeliveryProvider/Courier, Order,            │
│  DemandGenerator, RoutingModel                      │
│  All ABCs; extras registered by name in config.     │
│  Modules: entities/, routing/                       │
└──────────────────────┬──────────────────────────────┘
                       │  tick / events
┌──────────────────────▼──────────────────────────────┐
│  Layer 4 — Simulation Engine                        │
│  SimClock, EventQueue, WorldState, Simulator        │
│  Fixed dt tick; agent cadence decoupled.            │
│  Modules: engine/                                   │
└─────────────────────────────────────────────────────┘
```

**Cross-cutting concerns**
- `config/` — Pydantic v2 schemas + `load_scenario()`; one (config, seed) pair
  is fully reproducible.
- `render/` — pure consumer of engine snapshots; headless by default.
- `registry.py` — string-keyed `@register` / `create` for all pluggable types.

---

## Implementation build order

Fill stubs in this order. Each step's prompt should reference only the files
listed and the TODOs marked `# TODO(step-N)`.

| Step | What to implement | Key files |
|------|------------------|-----------|
| 1 | `EuclideanRouting` — distance, travel_time, route | `routing/euclidean.py` |
| 2 | Domain entity logic — `BuiltinStore`, `BikeCourier`, `PoissonDemandGenerator` | `entities/store.py`, `entities/courier.py`, `entities/demand_generator.py` |
| 3 | Engine core — `Simulator.reset/step/run`, `WorldState.snapshot` | `engine/simulator.py`, `engine/world_state.py`, `render/protocol.py` |
| 4 | Reward + metrics — `SparseDeliveryReward`, `KPICollector` | `rewards/placeholder.py`, `metrics/collector.py` |
| 5 | RL wrappers — `DeliveryEnv.reset/step`, `DeliveryParallelEnv.reset/step` | `envs/single_agent.py`, `envs/multi_agent.py` |
| 6 | Pygame renderer — `PygameRenderer` (optional dep) | `render/pygame_renderer.py` (new file) |
| 7 | Additional courier types — Drone, Box, custom | `entities/courier.py` + registry |
| 8 | Road-graph routing — replace Euclidean with graph model | `routing/` + new dep |

---

## Key design decisions

- **Headless first**: `HeadlessRenderer` is the default; training never touches pygame.
- **Single seeded RNG**: `numpy.random.Generator` threaded everywhere from `config.seed`.
- **Registry pattern**: users register custom types by name so YAML configs are
  self-describing — no Python code needed to swap implementations.
- **Decoupled tick cadences**: the world ticks at `dt`; agents can act every N
  ticks; this is expressed in the Gymnasium wrapper, not the engine.
- **Order state machine**: explicit enum + allowed-transition map; illegal
  transitions raise `ValueError` immediately.

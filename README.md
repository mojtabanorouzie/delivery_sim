# delivery_sim

[![version](https://img.shields.io/badge/version-v0.2.0--b--realistic-0d6efd)](https://github.com/mojtabanorouzie/delivery_sim/tags)
[![python](https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white)](https://docs.python.org/3.11/)
[![license](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![tests](https://img.shields.io/badge/tests-477%20passing-22c55e)](tests/)
[![style: ruff](https://img.shields.io/badge/style-ruff-000000?logo=ruff)](https://docs.astral.sh/ruff/)
[![type-checked: mypy strict](https://img.shields.io/badge/type--checked-mypy%20strict-2a6db2)](https://mypy.readthedocs.io/)

> **Event-driven, reproducible RL environment for last-mile delivery research.**  
> A trained PPO agent discovers an interior coverage optimum (r ≈ 676) that beats every fixed-radius baseline — zero failure rate, higher return than full-coverage — verified on 10 held-out seeds.

---

## What it is

`delivery_sim` is a Gymnasium-compatible reinforcement-learning environment built on a
**pure event-driven simulation engine**: the clock jumps to exact event timestamps, so
episodes with sparse demand are fast and all KPIs are timestep-independent.

An agent controls the **coverage radius** of each store. The simulator dispatches bike
couriers to customers within range, queues orders when stores are at capacity, and
optionally models probabilistic door-refusals (returns). Every environment parameter
lives in a YAML scenario file — no code changes to swap demand patterns, reward
functions, or observation layouts.

**Who it's for:** RL researchers benchmarking coverage/dispatch policies; students
learning to build Gymnasium environments; engineers who need a reproducible,
event-driven discrete-event simulation scaffold.

---

## Headline result

PPO trained on the `balanced` scenario (1 M steps, seed 7), evaluated on **10 held-out
seeds** (1000–1009):

| Policy | Return | Delivery rate | Failure rate | Mean radius |
| --- | --- | --- | --- | --- |
| **PPO (trained)** | **69.63 ±7.84** | **79.0 % ±5.1 %** | **0.0 %** | **676 ±3.6** |
| fixed r = 700 (manual optimum) | 68.77 ±8.60 | 78.6 % ±4.0 % | 2.5 % | 700 |
| fixed r = 1000 (full coverage) | 67.29 ±7.64 | 76.7 % ±2.8 % | 0.0 % | 1000 |
| fixed r = 500 | 57.53 ±10.31 | 72.3 % ±5.4 % | 17.2 % | 500 |
| fixed r = 0 (no coverage) | −48.50 ±5.71 | 0.0 % | 100.0 % | 0 |

**The agent converges to r ≈ 676 — an interior optimum it was never told existed.** The
cost-aware reward penalises large radii, so the trained policy outperforms full-coverage
(r = 1000) by +2.34 return points while matching its zero-failure rate, and beats the
manually-diagnosed best fixed radius (r = 700) on all three headline metrics.

*Source: `scripts/evaluate.py` against checkpoint `outputs/balanced_1M/ppo_delivery.zip`,
10 held-out seeds. Training-curve DR (~0.80 on the last rollout) is a separate measurement
on training seeds and is not the headline result.*

---

## Key capabilities

- **Event-driven clock** — wall-clock jumps to exact event timestamps; KPIs are
  dt-independent and reproducible. ([ADR-001](docs/adr/ADR-001.md))
- **Analytic courier trajectories** — `position_at(t)` queries any time without
  stepping; rendering is observer-invariant. ([ADR-002](docs/adr/ADR-002.md))
- **Pluggable registry** — `Store`, `Courier`, `DemandGenerator`, `RoutingModel`,
  `RewardFunction`, `ObservationSpec` all use `@register / create`; swap any
  component by changing one YAML field.
- **Gymnasium API** — `DeliveryEnv` is a drop-in for any SB3-compatible training loop.
  Action space: `Box(0, max_r, shape=(n_stores,), dtype=float32)`.
- **Three demand generators** — stationary Poisson; piecewise-linear time-varying
  (`DailyProfileDemandGenerator`); periodic burst (`BurstDemandGenerator`).
- **Store capacity & queueing** — orders queue when a store's courier slots are full;
  queue depth is visible in observations and the HUD.
- **Probabilistic returns** — configurable `return_rate` models door-refusals; refused
  couriers travel back to store, tracked as a distinct RETURNED state.
- **Three scenario presets** — `scenarios/presets/{light,balanced,heavy}.yaml` cover
  light-load through heavily-overloaded operating points.
- **Three observation presets** — `minimal` (n+1), `standard` (n+5), `operational`
  (n+3); select by name in config, no code change.
- **Live pygame visualization** — stores, coverage circles, couriers by phase, orders
  by state, queue stress, return paths, demand intensity bar, scenario HUD.
- **477 tests** — engine, entities, routing, config, rewards, metrics, env, renderer.

---

## Why it's rigorous

> **Reproducibility guarantee:** a single `(ScenarioConfig, seed)` pair fully
> determines an episode. One `numpy.random.Generator` is threaded through every
> stochastic call; child streams are spawned by stable index so RNG streams never
> collide.
>
> **No proxy rewards:** every reward signal traces to a real measured delivery outcome
> (delivered, failed, returned). There are no shaped signals that could mask a
> policy learning to game a proxy metric.
>
> **Action → outcome integrity:** the action-proxy defect (SB3 PPO collapses to r = 0
> on raw large action spaces without `RescaleAction`) is fixed and regression-tested.
> The agent's action is always what the simulator actually receives.
>
> **Observer-invariant rendering:** `PygameRenderer` consumes only `WorldSnapshot`
> objects; it never calls back into the engine. Rendering cannot affect simulation
> state or reproducibility.

---

## Visualization

The pygame renderer draws a live window at each decision step (or each dt boundary in
fixed-coverage mode):

- **Stores** — amber square, tint shifts red under queue load; badge shows queue depth
- **Coverage circles** — yellow translucent disc at each store's current radius
- **Couriers by phase** — teal (free) · blue (en-route-store) · violet (at-store) ·
  amber+□ (waiting/queued) · orange (en-route-customer) · coral+‹ (returning)
- **Orders by state** — grey (created) · blue (assigned) · violet (preparing) ·
  orange (in-transit) · teal ○ (delivered) · red ✕ (failed) · coral ◇ (returned)
- **HUD** — sim time, delivered / failed / returned counts, return rate, pending
  orders, total queue depth, mean coverage radius, demand intensity bar, legend

### Capture a demo

```bash
# Install render extra first (adds pygame)
pip install -e ".[render]"

# Watch a heavy-load episode (fixed coverage, dt-cadence frames)
python scripts/visualize.py --scenario scenarios/presets/heavy.yaml

# Watch the trained PPO agent act (one frame per decision step)
python scripts/visualize.py \
  --model-path outputs/balanced_1M/ppo_delivery \
  --obs-preset operational \
  --fps 10
```

Place a screenshot or screen-capture here:

```markdown
![demo](docs/assets/demo.gif)
```

*(To generate: run either command above and record with your OS screen-capture tool,
or use `scripts/visualize.py --fps 5` for a slower, easier-to-record playback.)*

---

## Install

```bash
# core env only (numpy, gymnasium, pydantic, pyyaml)
pip install -e .

# with dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# with training stack (stable-baselines3, torch, matplotlib)
pip install -e ".[dev,train]"

# with pygame visualization
pip install -e ".[render]"

# everything
pip install -e ".[dev,train,render]"

# uv users
uv sync --extra dev --extra train --extra render
```

Python 3.11+ required.

---

## Quickstart

### Step the environment (random agent)

```python
from delivery_sim import load_scenario, DeliveryEnv

config = load_scenario("scenarios/balanced.yaml")
env = DeliveryEnv(config)                 # render_mode="headless" by default
obs, info = env.reset(seed=42)

terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()    # shape (n_stores,), float32
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
print(info)  # KPI summary dict
```

See [examples/random_agent.py](examples/random_agent.py) for the same loop with logging.

### Train PPO (requires `[train]`)

```bash
python scripts/train.py \
  --scenario scenarios/balanced.yaml \
  --total-timesteps 1000000 \
  --seed 7 \
  --obs-preset operational \
  --ent-coef 0.01 \
  --output-dir outputs/balanced_1M
```

Writes `ppo_delivery.zip`, `learning_curve.csv`, `learning_curve.png` to the output
directory. Wall-clock: ~8–20 min on a modern CPU (MLP policy, CPU only).

### Evaluate against baselines

```bash
python scripts/evaluate.py \
  --scenario scenarios/balanced.yaml \
  --model-path outputs/balanced_1M/ppo_delivery \
  --n-eval-seeds 10 \
  --obs-preset operational
```

Prints the comparison table and a `SUCCESS / PARTIAL / COLLAPSE` verdict.

### Try a B-realistic preset

```bash
# Heavy load: high demand, constrained fleet, store queueing
python scripts/diagnose_scenario.py --scenario scenarios/presets/heavy.yaml

# Visualize it live (requires [render])
python scripts/visualize.py --scenario scenarios/presets/heavy.yaml
```

---

## Project layout

```text
src/delivery_sim/
├── config/          — Pydantic v2 schema (ScenarioConfig) + load_scenario()
├── engine/          — SimClock, EventQueue, WorldState, Simulator
├── entities/        — Store, Courier, Order, DemandGenerator ABCs + built-ins
│                      (PoissonDemandGenerator, DailyProfileDemandGenerator,
│                       BurstDemandGenerator, BikeCourier, BuiltinStore)
├── routing/         — RoutingModel ABC + EuclideanRouting
├── envs/            — DeliveryEnv (Gymnasium), ObservationSpec presets
│                      (minimal / standard / operational)
├── rewards/         — RewardFunction ABC + SparseDelivery, CostAware, …
├── metrics/         — KPICollector (event-driven KPI aggregation)
├── render/          — SnapshotConsumer protocol, HeadlessRenderer, PygameRenderer
└── registry.py      — @register / create factory

scenarios/           — balanced.yaml (validated training scenario), example.yaml
scenarios/presets/   — light.yaml, balanced.yaml, heavy.yaml (B-realistic presets)
scripts/             — train.py, evaluate.py, diagnose_scenario.py, visualize.py
docs/adr/            — ADR-001 (event clock), ADR-002 (courier trajectories)
docs/design/         — increment-b-realistic-architecture.md
examples/            — random_agent.py, greedy_agent.py
tests/               — 477 tests (engine · entities · routing · config · rewards ·
                        metrics · env · renderer · fingerprint regression)
```

---

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — four-layer design, event model, registry pattern
- [docs/extending.md](docs/extending.md) — add custom stores, couriers, rewards, observations
- [docs/adr/ADR-001.md](docs/adr/ADR-001.md) — why the clock is event-driven (not tick-based)
- [docs/adr/ADR-002.md](docs/adr/ADR-002.md) — why couriers are time-parameterised trajectories

---

## Development

```bash
pytest                    # 477 tests (476 passing, 1 skipped)
ruff check .              # lint
mypy src                  # type-check (strict)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev workflow.

---

## Roadmap

The following are **planned but not built**:

- **Multi-agent dispatch** — PettingZoo `ParallelEnv` wrapper (stub exists at
  `DeliveryParallelEnv`; not wired to a training loop)
- **In-store / customer agents** — packing, tipping, preference modelling
- **Drone and cargo-box courier types** — capacity > 1, altitude routing
- **Road-graph routing** — replace Euclidean straight-line with an OSM road network

---

## License

Licensed under MIT — see [LICENSE](LICENSE).

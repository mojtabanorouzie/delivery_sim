# delivery_sim

Extensible, reproducible simulation of e-commerce last-mile delivery, designed as
a reinforcement-learning environment for researchers and students.

## What it is

`delivery_sim` is a Gymnasium-compatible RL environment built on an **event-driven
simulation engine**. An agent controls the coverage radius of each store; the
simulator dispatches couriers to nearby customers and scores delivery outcomes.

Every environment parameter is set in a YAML scenario file — no code changes needed
to swap courier types, reward functions, demand patterns, or observation layouts.
All pluggable components are registered by name so configs are self-describing.

## Validated result

A PPO agent trained on the `balanced` scenario (1 M steps, seed 7) was evaluated on
**10 held-out seeds** (1000–1009). Measured results:

| Policy | Return | DR/total | FR/resolved | Mean radius |
| --- | --- | --- | --- | --- |
| **ppo_trained** | **69.63 ±7.84** | **79.0 % ±5.1 %** | **0.0 %** | **676 ±3.6** |
| fixed r=700 (diagnosed optimum) | 68.77 ±8.60 | 78.6 % ±4.0 % | 2.5 % | 700 |
| fixed r=1000 (full coverage) | 67.29 ±7.64 | 76.7 % ±2.8 % | 0.0 % | 1000 |
| fixed r=500 | 57.53 ±10.31 | 72.3 % ±5.4 % | 17.2 % | 500 |
| fixed r=0 (no coverage) | −48.50 ±5.71 | 0.0 % | 100.0 % | 0 |

**Verdict: SUCCESS.** The agent converges to r ≈ 676 — an interior optimum that
outperforms the manually-diagnosed best fixed radius (r=700) on all three headline
metrics. The cost-aware reward penalises large radii, so the trained policy beats
full-coverage (r=1000) by 2.34 return points while matching its zero-failure rate.

*Numbers from `evaluate.py` against checkpoint `outputs/balanced_1M/ppo_delivery.zip`,
10 held-out seeds. Training-curve DR (~0.80 on the last rollout) is a different
measurement on training seeds and is not the headline result.*

## Install

```bash
# core env only (numpy, gymnasium, pydantic, pyyaml)
pip install -e .

# with dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# with training stack (stable-baselines3, torch, matplotlib)
pip install -e ".[dev,train]"

# uv users
uv sync --extra dev --extra train
```

Python 3.11+ required.

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

See `examples/random_agent.py` for the same loop.

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

Prints the comparison table and a SUCCESS/PARTIAL/COLLAPSE verdict.

## Key features

- **Event-driven engine** — the clock jumps to exact event timestamps, not fixed
  ticks. Episodes with sparse demand run fast; KPIs are dt-independent. ([ADR-001](docs/adr/ADR-001.md))
- **Analytic courier trajectories** — `position_at(t)` queries any time without
  stepping; reproducible across rendering modes. ([ADR-002](docs/adr/ADR-002.md))
- **Pluggable everything** — Store, Courier, DemandGenerator, RoutingModel,
  RewardFunction, ObservationSpec all use a `@register / create` registry. Swap
  any component by changing one field in the YAML scenario.
- **Gymnasium API** — `DeliveryEnv` is a drop-in for any SB3-compatible training
  loop. Action space: `Box(0, max_r, shape=(n_stores,))`.
- **Reproducibility** — one `(config, seed)` pair fully determines an episode.
  The single `numpy.random.Generator` is threaded through every stochastic call.
- **310 tests** — full coverage of engine, entities, routing, config, rewards,
  metrics, and env reset/step.

## Project layout

```text
src/delivery_sim/
├── config/          — Pydantic v2 schema + load_scenario()
├── engine/          — SimClock, EventQueue, WorldState, Simulator
├── entities/        — Store, Courier, Order, DemandGenerator ABCs + built-ins
├── routing/         — RoutingModel ABC + EuclideanRouting
├── envs/            — DeliveryEnv (Gymnasium), observations presets
├── rewards/         — RewardFunction ABC + SparseDelivery, CostAware, …
├── metrics/         — KPICollector (event-driven KPI aggregation)
├── render/          — SnapshotConsumer protocol, HeadlessRenderer
└── registry.py      — @register / create factory

scenarios/           — balanced.yaml (RL training), example.yaml (demo)
scripts/             — train.py, evaluate.py, diagnose_scenario.py, run_notes.md
docs/adr/            — ADR-001 (event clock), ADR-002 (courier trajectories)
examples/            — random_agent.py, greedy_agent.py
```

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — four-layer design, event model, registry pattern
- [docs/extending.md](docs/extending.md) — add custom stores, couriers, rewards, observations
- [docs/adr/ADR-001.md](docs/adr/ADR-001.md) — why the clock is event-driven
- [docs/adr/ADR-002.md](docs/adr/ADR-002.md) — why couriers are time-parameterised trajectories

## Development

```bash
pytest                    # 308 tests
ruff check .              # lint
mypy src                  # type-check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full dev workflow.

## License

MIT — see `license` field in `pyproject.toml`.

# Contributing

## Dev setup

```bash
# install with dev + train extras
pip install -e ".[dev,train]"

# uv users
uv sync --extra dev --extra train
```

Python 3.11+ required.

## Tests

```bash
pytest                       # all 308 tests
pytest -x                    # stop on first failure
pytest tests/test_engine.py  # single module
```

The test suite covers engine, entities, routing, config, rewards, metrics, and
env reset/step. Tests must be green before any PR. Do not add mocks for
components that can be exercised directly — integration coverage is the signal.

## Lint and type-check

```bash
ruff check .                 # lint (E, F, I, UP rules)
ruff check . --fix           # auto-fix safe issues
mypy src                     # strict type-check
```

All three must pass cleanly. `ruff` and `mypy` configs are in `pyproject.toml`.

## Training stack (`[train]` extra)

`stable-baselines3`, `torch`, and `matplotlib` are optional. Install the `[train]`
extra to use `scripts/train.py` and `scripts/evaluate.py`. Core env functionality
and all 308 tests run without it.

## Plan before implementing

For non-trivial changes (new entities, reward functions, env modifications), agree on
the design before writing code:

1. Open an issue or PR draft describing the change, the layer it lives in, and the
   ABCs or hooks it touches.
2. Read the relevant ADRs ([ADR-001](docs/adr/ADR-001.md),
   [ADR-002](docs/adr/ADR-002.md)) before touching engine or entity code.
3. Implement; run tests; update docstrings and this guide if public API changes.

## Known tech debt

- **`DeliveryParallelEnv`:** The PettingZoo multi-agent wrapper exists in
  `envs/multi_agent.py` but `reset()`, `step()`, `observation_space()`, and
  `action_space()` all raise `NotImplementedError`. It is exported from the package
  but not usable. A full multi-agent implementation requires defining per-courier
  observation and action spaces and the dispatch decision interface.

## Flagged stale comments (not yet fixed in code)

- `examples/random_agent.py:9` — says *"NOTE: env.reset() and env.step() raise
  NotImplementedError until step-5 is complete."* Step-5 is complete; the file works.
  The comment should be removed.

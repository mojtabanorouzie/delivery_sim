# delivery_sim

An extensible, reproducible simulation of e-commerce last-mile delivery,
designed as a reinforcement-learning environment for researchers and students.

## Quick start

```bash
# install (uv preferred)
uv sync --extra dev

# or with pip
pip install -e ".[dev]"

# smoke tests
pytest

# lint + type-check
ruff check .
mypy src
```

## Usage shape

```python
from delivery_sim import load_scenario, DeliveryEnv

config = load_scenario("scenarios/example.yaml")
env = DeliveryEnv(config, render_mode="headless")
obs, info = env.reset(seed=config.seed)
```

See `examples/random_agent.py` for the full loop.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the four-layer design and the
implementation build order.

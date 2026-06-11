#!/usr/bin/env python
"""
Live visualization of a delivery_sim scenario using PygameRenderer.

REQUIRES:  pip install 'delivery-sim[render]'   (adds pygame)
OPTIONAL:  pip install 'delivery-sim[train]'    (adds stable-baselines3 for --model-path)

USAGE
-----
  # Smooth fixed-coverage animation (dt cadence, one frame per sim-second):
  python scripts/visualize.py

  # Custom scenario:
  python scripts/visualize.py --scenario scenarios/balanced.yaml

  # Trained PPO policy (one frame per decision step):
  python scripts/visualize.py --model-path outputs/ppo_delivery

  # Slower playback:
  python scripts/visualize.py --fps 10

MODES
-----
  Without --model-path
      Simulator.run() feeds the renderer at every dt boundary (smooth).
      Coverage radius stays at the value from the scenario config (default 500).

  With --model-path
      DeliveryEnv + RescaleAction wrapper drives the model; one snapshot is
      emitted per decision step so you can watch coverage_radius change as
      the agent responds to demand.  The trained model must have been trained
      on the same scenario / obs-preset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import delivery_sim  # noqa: F401 — triggers all @register decorators
from delivery_sim.config.loader import load_scenario
from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    RewardConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.engine.simulator import Simulator
from delivery_sim.render.pygame_renderer import PygameRenderer

# ---------------------------------------------------------------------------
# Default scenario (matches evaluate.py build_eval_config)
# ---------------------------------------------------------------------------

def _default_config(seed: int) -> ScenarioConfig:
    return ScenarioConfig(
        name="visualize_demo",
        seed=seed,
        dt=1.0,
        max_steps=1000,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[
            StoreConfig(name="warehouse_a", x=200.0, y=200.0, capacity=20),
            StoreConfig(name="warehouse_b", x=800.0, y=600.0, capacity=15),
        ],
        couriers=[
            CourierConfig(courier_type="BikeCourier", count=3, speed=1.5, cost_per_unit=0.01),
        ],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=0.5),
        reward=RewardConfig(function_type="cost_aware"),
        decision_interval=100.0,
        max_coverage_radius=1000.0,
        observation_preset="operational",
    )


# ---------------------------------------------------------------------------
# Headless run (smooth dt-cadence rendering, fixed coverage_radius)
# ---------------------------------------------------------------------------

def run_fixed(config: ScenarioConfig, renderer: PygameRenderer) -> None:
    """Run a full episode via Simulator.run(); renderer receives one snapshot per dt."""
    sim = Simulator(config)
    sim.run(consumer=renderer)


# ---------------------------------------------------------------------------
# RL-policy run (one snapshot per decision step)
# ---------------------------------------------------------------------------

def run_with_policy(
    config: ScenarioConfig,
    renderer: PygameRenderer,
    model_path: Path,
    obs_preset: str,
    seed: int,
) -> None:
    """Load a PPO model and run one episode, emitting one snapshot per decision step."""
    try:
        from gymnasium.wrappers import RescaleAction  # noqa: PLC0415
        from stable_baselines3 import PPO  # noqa: PLC0415
    except ImportError:
        print(
            "stable-baselines3 / gymnasium not installed. "
            "Run: pip install 'delivery-sim[train]'",
            file=sys.stderr,
        )
        sys.exit(1)

    model_file = (
        Path(str(model_path) + ".zip")
        if not str(model_path).endswith(".zip")
        else model_path
    )
    if not model_file.exists():
        print(f"Model file not found: {model_file}", file=sys.stderr)
        sys.exit(1)
    model = PPO.load(str(model_path))
    print(f"Loaded model from {model_file}")

    # Patch obs preset so it matches training
    cfg = config.model_copy(update={"observation_preset": obs_preset})

    from delivery_sim.envs.single_agent import DeliveryEnv  # noqa: PLC0415

    raw_env = DeliveryEnv(cfg, render_mode="headless")
    env = RescaleAction(raw_env, min_action=np.float32(-1.0), max_action=np.float32(1.0))

    obs, _ = env.reset(seed=seed)
    done = False
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        # Emit a snapshot from the inner simulator's current world state.
        # The renderer sees only the WorldSnapshot — it never touches the sim.
        inner = raw_env  # type: ignore[attr-defined]
        if inner._simulator.world is not None:
            snap = inner._simulator.world.snapshot(step, inner._sim_time)
            renderer.consume(snap)

        step += 1
        done = terminated or truncated

    renderer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario", type=Path, default=None,
        help="YAML scenario file (default: built-in two-warehouse demo)",
    )
    p.add_argument(
        "--model-path", type=Path, default=None, dest="model_path",
        help="Trained PPO model path (without .zip); omit for fixed-coverage mode",
    )
    p.add_argument("--seed", type=int, default=42, help="Episode seed (default 42)")
    p.add_argument("--fps", type=int, default=30, help="Target frame rate (default 30)")
    p.add_argument(
        "--obs-preset", type=str, default="operational", dest="obs_preset",
        help="ObservationSpec preset used during training (default 'operational')",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.scenario is not None:
        config = load_scenario(args.scenario)
        config = config.model_copy(update={"seed": args.seed})
    else:
        config = _default_config(seed=args.seed)

    renderer = PygameRenderer(
        world_width=config.world.width,
        world_height=config.world.height,
        fps=args.fps,
    )

    if args.model_path is not None:
        run_with_policy(config, renderer, args.model_path, args.obs_preset, args.seed)
    else:
        run_fixed(config, renderer)
        renderer.close()


if __name__ == "__main__":
    main()

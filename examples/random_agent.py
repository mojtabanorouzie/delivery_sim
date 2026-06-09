"""
Random-agent example — demonstrates the intended DeliveryEnv usage shape.

NOTE: env.reset() and env.step() raise NotImplementedError until step-5 is
complete.  This file is intentionally kept as a usage skeleton so researchers
can see the API contract without running it yet.

Run after step-5:
    python examples/random_agent.py
"""

from __future__ import annotations

from pathlib import Path

from delivery_sim import DeliveryEnv, load_scenario


def main() -> None:
    config = load_scenario(Path(__file__).parent.parent / "scenarios" / "example.yaml")

    env = DeliveryEnv(config, render_mode="headless")
    obs, info = env.reset(seed=config.seed)

    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    env.close()
    print(f"Episode finished: {steps} steps, total reward = {total_reward:.2f}")


if __name__ == "__main__":
    main()

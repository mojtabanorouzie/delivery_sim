"""
Agent comparison for DeliveryEnv — four coverage-control strategies.

Strategies
----------
RandomAgent        Samples coverage_radius uniformly each step (baseline).

MaxCoverageAgent   Sets every store to max_coverage_radius every step.
                   Optimal for SparseDeliveryReward: no cost for large
                   radius, so coverage misses are the only failure path.

AdaptiveAgent      Reactive P-controller. Starts at 50% radius. When a
                   coverage failure is detected (fail_rate > 0), it drives
                   each store's radius toward max at a configurable rate.
                   When no failures occur it slowly retreats.

QLearningAgent     Tabular Q-learning over a discretised (state, action)
                   space. State = (fail_bin, busy_bin, pending_bin);
                   actions = all combinations of 4 radius levels per store.
                   Trains for N episodes with epsilon-greedy exploration,
                   then evaluates greedily.

Run:
    python examples/greedy_agent.py
"""

from __future__ import annotations

import itertools
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np

from delivery_sim import DeliveryEnv, load_scenario
from delivery_sim.config.schema import ScenarioConfig


# ---------------------------------------------------------------------------
# Agent interface
# ---------------------------------------------------------------------------

class Agent(ABC):
    """Policy that maps an observation to a coverage-radius action vector."""

    @abstractmethod
    def reset(self) -> None:
        """Called once at the start of every episode."""

    @abstractmethod
    def act(self, obs: np.ndarray, n_stores: int, max_r: float) -> np.ndarray:
        """Return a float32 action array of shape (n_stores,)."""


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------

class RandomAgent(Agent):
    """Uniform random coverage radius — the baseline."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        pass

    def act(self, _obs: np.ndarray, n_stores: int, max_r: float) -> np.ndarray:
        return self._rng.uniform(0, max_r, size=n_stores).astype(np.float32)


class MaxCoverageAgent(Agent):
    """Always open every store to maximum radius — eliminates coverage failures."""

    def reset(self) -> None:
        pass

    def act(self, _obs: np.ndarray, n_stores: int, max_r: float) -> np.ndarray:
        return np.full(n_stores, max_r, dtype=np.float32)


class AdaptiveAgent(Agent):
    """Reactive P-controller driven by the observation's fail_rate signal."""

    def __init__(
        self,
        alpha_up: float = 0.6,
        alpha_down: float = 0.05,
        init_frac: float = 0.5,
    ) -> None:
        self._alpha_up = alpha_up
        self._alpha_down = alpha_down
        self._init_frac = init_frac
        self._radii: np.ndarray | None = None

    def reset(self) -> None:
        self._radii = None

    def act(self, obs: np.ndarray, n_stores: int, max_r: float) -> np.ndarray:
        if self._radii is None:
            self._radii = np.full(n_stores, self._init_frac * max_r, dtype=np.float32)

        delivery_rate = float(obs[n_stores])
        fail_rate = float(obs[n_stores + 1])
        sentinel = abs(delivery_rate - 0.5) < 1e-6 and abs(fail_rate) < 1e-6

        if not sentinel:
            if fail_rate > 0.0:
                self._radii += self._alpha_up * (max_r - self._radii)
            else:
                self._radii *= (1.0 - self._alpha_down)

        self._radii = np.clip(self._radii, 0.0, max_r).astype(np.float32)
        return self._radii.copy()


# ---------------------------------------------------------------------------
# v1: original action levels (baseline — wide range, 4 levels)
_QL_LEVELS_V1 = [0.25, 0.5, 0.75, 1.0]

# v2 (improved): skip tiny radii that always cause failures; add granularity
# near the optimal zone (0.8-1.0) where the coverage/latency trade-off lives.
_QL_LEVELS_V2 = [0.5, 0.65, 0.8, 0.9, 1.0]

# v3 (enriched): same action levels as v2, but richer state space
_QL_LEVELS_V3 = [0.5, 0.65, 0.8, 0.9, 1.0]


class QLearningAgent(Agent):
    """Tabular Q-learning — three versions for direct comparison.

    version="v1"  (original / baseline)
    ─────────────────────────────────────
    State  : (fail_bin, busy_bin, latency_bin)  — 2x3x3 = 18 states
    Problem: busy_frac is ALWAYS 1.0 and latency_norm is ALWAYS < 0.2 in
             this env, so the agent visits only ONE state all episode.
             Q-learning cannot differentiate and degenerates to a random walk.
    Actions: [0.25, 0.5, 0.75, 1.0]  — 4^n_stores combinations

    version="v2"  (improved)
    ─────────────────────────
    State  : (fail_bin, coverage_bin)  — 2x4 = 8 states
      fail_bin     — 0: no coverage failures this step, 1: some failures
      coverage_bin — average of obs[0..n-1] (agent's own last radii, normalised)
                     bucketed into 4 levels: [<0.5, 0.5-0.65, 0.65-0.85, 0.85+]
                     This mirrors the action levels so state reflects where the
                     agent currently IS in the action space.
    WHY: coverage_bin is the only feature that reliably varies with the
         agent's actions.  Profiling shows 5/8 states are visited during
         exploration vs 1/18 in v1.

    Actions: [0.5, 0.65, 0.8, 0.9, 1.0]  — 5^n_stores combinations
      Removes sub-0.5 radii (always catastrophic failures) and adds a 0.9
      level to resolve the coverage vs. cost trade-off more precisely.

    Q-init : optimistic (+1.0) — encourages the agent to try every action
             at least once before exploiting, preventing premature convergence
             to the first action that happened to give a positive reward.

    Training: 300 episodes (up from 200) to allow full convergence with the
              larger action space (25 vs 16 combos for 2 stores).

    version="v3"  (enriched state space)
    ────────────────────────────────────
    State  : (fail_bin, coverage_bin, delivery_bin)  — 2x4x2 = 16 states
      fail_bin      — 0: no coverage failures, 1: some failures
      coverage_bin  — [<0.5, 0.5-0.65, 0.65-0.85, 0.85+]  (same as v2)
      delivery_bin  — 0: delivery_rate < 0.2, 1: delivery_rate >= 0.2
                      Captures whether system is actively delivering vs stuck.
    WHY: delivery_bin adds temporal dynamics — distinguishes high-activity
         system states from low-activity / deadlock states.  Enables learning
         different policies for systems that are "flowing" vs "congested".
         Addresses limitation that v2 cannot distinguish "low coverage + no
         failures yet" (fast orders) from "low coverage + now failing" (orders
         arriving from far away). delivery_bin bridges the gap.

    Actions: [0.5, 0.65, 0.8, 0.9, 1.0]  — 5^n_stores combinations
      Same as v2 to isolate the effect of state space enrichment.

    Q-init : optimistic (+1.0) — same as v2.

    Training: 300 episodes (same as v2).
    """

    def __init__(
        self,
        n_stores: int,
        version: str = "v2",
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.05,
    ) -> None:
        self._n_stores = n_stores
        self._version = version
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min

        if version == "v3":
            levels = _QL_LEVELS_V3
            n_states = 16
        elif version == "v2":
            levels = _QL_LEVELS_V2
            n_states = 8
        else:
            levels = _QL_LEVELS_V1
            n_states = 18

        self._levels = levels
        self._combos: list[tuple[int, ...]] = list(
            itertools.product(range(len(levels)), repeat=n_stores)
        )
        self._n_actions = len(self._combos)
        self._n_states = n_states

        # Optimistic Q-init for v2/v3, zero-init for v1 (baseline)
        self._q: defaultdict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.ones(self._n_actions, dtype=np.float64)
            if version in ("v2", "v3")
            else np.zeros(self._n_actions, dtype=np.float64)
        )

        self._last_state: tuple[int, ...] | None = None
        self._last_action: int | None = None

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._last_state = None
        self._last_action = None

    def act(self, obs: np.ndarray, n_stores: int, max_r: float) -> np.ndarray:
        state = self._encode(obs)
        if np.random.random() < self._epsilon:
            idx = np.random.randint(self._n_actions)
        else:
            idx = int(np.argmax(self._q[state]))
        self._last_state = state
        self._last_action = idx
        return np.array(
            [self._levels[self._combos[idx][i]] * max_r for i in range(n_stores)],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, config: ScenarioConfig, n_episodes: int = 300) -> list[float]:
        """Train in-place; return per-episode reward list."""
        env = DeliveryEnv(config, render_mode="headless")
        n_stores = len(config.stores)
        max_r = config.max_coverage_radius
        rewards: list[float] = []

        print(
            f"\n  Training QLearningAgent-{self._version}  "
            f"({n_episodes} eps, {self._n_states} states, {self._n_actions} actions, "
            f"alpha={self._alpha} gamma={self._gamma} "
            f"eps {self._epsilon:.2f}->{self._epsilon_min:.2f})"
        )

        log_every = max(1, n_episodes // 4)

        for ep in range(n_episodes):
            self.reset()
            obs, _ = env.reset(seed=ep)
            total_reward = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                action = self.act(obs, n_stores, max_r)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                self._td_update(float(reward), next_obs, terminated or truncated)
                obs = next_obs
                total_reward += float(reward)

            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
            rewards.append(total_reward)

            if (ep + 1) % log_every == 0:
                window = rewards[max(0, ep - 19): ep + 1]
                avg = sum(window) / len(window)
                print(f"    ep {ep + 1:>4}  |  avg(last 20) = {avg:>7.2f}  |  eps = {self._epsilon:.3f}")

        env.close()
        self._epsilon = 0.0  # fully greedy for evaluation
        visited = sum(1 for v in self._q.values() if v.any())
        print(f"  Done. Q-table: {visited}/{self._n_states} states visited.\n")
        return rewards

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, obs: np.ndarray) -> tuple[int, ...]:
        n = self._n_stores
        delivery_rate = float(obs[n])
        fail_rate     = float(obs[n + 1])
        sentinel = abs(delivery_rate - 0.5) < 1e-6 and abs(fail_rate) < 1e-6
        fail_bin = 0 if (sentinel or fail_rate < 1e-6) else 1

        if self._version == "v3":
            # coverage_bin: average normalised radius (same as v2)
            cov_avg = float(np.mean(obs[:n]))
            if cov_avg < 0.5:
                cov_bin = 0
            elif cov_avg < 0.65:
                cov_bin = 1
            elif cov_avg < 0.85:
                cov_bin = 2
            else:
                cov_bin = 3
            # delivery_bin: system activity indicator
            delivery_bin = 0 if delivery_rate < 0.2 else 1
            return (fail_bin, cov_bin, delivery_bin)
        elif self._version == "v2":
            # coverage_bin: average normalised radius from agent's own last action.
            # Thresholds mirror _QL_LEVELS_V2 so the agent's state reflects
            # exactly which "level zone" it is currently operating in.
            cov_avg = float(np.mean(obs[:n]))
            if cov_avg < 0.5:
                cov_bin = 0
            elif cov_avg < 0.65:
                cov_bin = 1
            elif cov_avg < 0.85:
                cov_bin = 2
            else:
                cov_bin = 3
            return (fail_bin, cov_bin)
        else:
            # v1: original features (busy and latency — rarely vary in practice)
            busy_bin    = min(int(float(obs[n + 2]) * 3), 2)
            latency_bin = min(int(float(obs[n + 3]) * 3), 2)
            return (fail_bin, busy_bin, latency_bin)

    def _td_update(self, reward: float, next_obs: np.ndarray, done: bool) -> None:
        if self._last_state is None or self._last_action is None:
            return
        next_state = self._encode(next_obs)
        current_q = self._q[self._last_state][self._last_action]
        future = 0.0 if done else float(np.max(self._q[next_state]))
        target = reward + self._gamma * future
        self._q[self._last_state][self._last_action] += self._alpha * (target - current_q)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    agent: Agent,
    config: ScenarioConfig,
    seed: int,
    *,
    verbose: bool = False,
) -> dict[str, float | int]:
    """Run one episode and return a dict of episode-level results."""
    env = DeliveryEnv(config, render_mode="headless")
    n_stores = len(config.stores)
    max_r = config.max_coverage_radius

    agent.reset()
    obs, _ = env.reset(seed=seed)

    total_reward = 0.0
    terminated = truncated = False
    kpi: dict[str, float | int] = {}

    if verbose:
        _header()

    step = 0
    while not (terminated or truncated):
        action = agent.act(obs, n_stores, max_r)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1

        if verbose:
            _log_step(step, float(reward), obs, n_stores)

        if "kpi" in info:
            kpi = info["kpi"]

    env.close()

    if verbose:
        _footer(step, total_reward, kpi)

    return {
        "total_reward": total_reward,
        "steps": step,
        "delivered": int(kpi.get("delivered_orders", 0)),
        "failed": int(kpi.get("failed_orders", 0)),
        "total_orders": int(kpi.get("total_orders", 0)),
        "courier_utilization": float(kpi.get("courier_utilization", 0.0)),
        "mean_delivery_time": float(kpi.get("mean_delivery_time", 0.0)),
    }


# ---------------------------------------------------------------------------
# Multi-seed comparison
# ---------------------------------------------------------------------------

EVAL_SEEDS = [42, 123, 456, 789, 1337]


def compare(config: ScenarioConfig, agents: dict[str, Agent]) -> None:
    """Run every agent over EVAL_SEEDS and print a summary table."""
    print(f"\n=== Multi-seed comparison  ({len(EVAL_SEEDS)} seeds: {EVAL_SEEDS})")
    hdr = (
        f"{'Agent':<18}  {'Reward mean':>11}  {'+/-std':>7}  "
        f"{'Delivered':>9}  {'Failed':>6}  {'Orders':>7}  {'Util':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, agent in agents.items():
        results = [run_episode(agent, config, s) for s in EVAL_SEEDS]
        rewards   = [r["total_reward"] for r in results]
        delivered = [float(r["delivered"]) for r in results]
        failed    = [float(r["failed"]) for r in results]
        orders    = [float(r["total_orders"]) for r in results]
        util      = [r["courier_utilization"] for r in results]

        def _f(vals: list[float]) -> str:
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return f"{m:>7.2f} +/-{s:>4.2f}"

        print(
            f"{name:<18}  {_f(rewards):>19}  "
            f"{statistics.mean(delivered):>9.1f}  "
            f"{statistics.mean(failed):>6.1f}  "
            f"{statistics.mean(orders):>7.1f}  "
            f"{statistics.mean(util):>6.3f}"
        )


# ---------------------------------------------------------------------------
# Verbose helpers
# ---------------------------------------------------------------------------

def _header() -> None:
    print(
        f"{'Step':>4}  {'Reward':>7}  {'Coverage':>16}  "
        f"{'DelivRate':>9}  {'FailRate':>8}  {'Busy':>5}  {'Pending':>7}"
    )
    print("-" * 68)


def _log_step(step: int, reward: float, obs: np.ndarray, n_stores: int) -> None:
    coverage_str = " ".join(f"{obs[i]:.2f}" for i in range(n_stores))
    delivery_rate = float(obs[n_stores])
    fail_rate     = float(obs[n_stores + 1])
    busy_frac     = float(obs[n_stores + 2])
    pending_norm  = float(obs[n_stores + 4])
    sentinel = abs(delivery_rate - 0.5) < 1e-6 and abs(fail_rate) < 1e-6
    dr = "  --   " if sentinel else f"{delivery_rate:.3f}"
    fr = "  --  "  if sentinel else f"{fail_rate:.3f}"
    print(
        f"{step:>4}  {reward:>7.2f}  {coverage_str:>16}  "
        f"{dr:>9}  {fr:>8}  {busy_frac:>5.2f}  {pending_norm:>7.2f}"
    )


def _footer(steps: int, total_reward: float, kpi: dict[str, float | int]) -> None:
    print("-" * 68)
    print(f"Episode: {steps} steps  |  total reward = {total_reward:.2f}")
    if kpi:
        d = int(kpi["delivered_orders"])
        f = int(kpi["failed_orders"])
        t = int(kpi["total_orders"])
        print(
            f"Orders : {t} created  |  {d} delivered ({100*d/max(t,1):.1f}%)  "
            f"|  {f} failed  |  courier util = {kpi['courier_utilization']:.3f}"
        )
        if d:
            print(
                f"Latency: mean = {kpi['mean_delivery_time']:.0f}s  "
                f"|  p95 = {kpi['p95_delivery_time']:.0f}s"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from delivery_sim.config.schema import RewardConfig

    base_config = load_scenario(Path(__file__).parent.parent / "scenarios" / "example.yaml")
    config = base_config.model_copy(update={"max_steps": 5000})
    n_stores = len(config.stores)

    scenarios = [
        ("SparseDeliveryReward",   config),
        ("LatencyAwareReward",     config.model_copy(update={"reward": RewardConfig(function_type="LatencyAwareReward")})),
        ("OptimizedDeliveryReward",config.model_copy(update={"reward": RewardConfig(function_type="OptimizedDeliveryReward")})),
    ]

    for label, cfg in scenarios:
        print(f"\n{'#' * 68}")
        print(f"  REWARD: {label}")
        print(f"{'#' * 68}")

        ql_v1 = QLearningAgent(n_stores=n_stores, version="v1")
        ql_v2 = QLearningAgent(n_stores=n_stores, version="v2")
        ql_v3 = QLearningAgent(n_stores=n_stores, version="v3")
        ql_v1.train(cfg, n_episodes=300)
        ql_v2.train(cfg, n_episodes=300)
        ql_v3.train(cfg, n_episodes=300)

        agents: dict[str, Agent] = {
            "Random":        RandomAgent(seed=0),
            "MaxCoverage":   MaxCoverageAgent(),
            "Adaptive":      AdaptiveAgent(alpha_up=0.6, alpha_down=0.05, init_frac=0.5),
            "QL-v1(broken)": ql_v1,
            "QL-v2(fixed)":  ql_v2,
            "QL-v3(rich)":   ql_v3,
        }

        compare(cfg, agents)


if __name__ == "__main__":
    main()

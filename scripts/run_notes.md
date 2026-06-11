# PPO Training Run Notes — Balanced Scenario

## Exact train command

```bash
python scripts/train.py \
  --scenario scenarios/balanced.yaml \
  --total-timesteps 1000000 \
  --seed 7 \
  --obs-preset operational \
  --ent-coef 0.01 \
  --output-dir outputs/balanced_1M
```

Flags explained:
| Flag | Value | Why |
|---|---|---|
| `--scenario` | `scenarios/balanced.yaml` | 20 couriers, demand=0.025, horizon=4000 (balanced/stable scenario) |
| `--total-timesteps` | `1 000 000` | ~25 000 episodes, 488 rollout rows in CSV. Prior 100k run collapsed. |
| `--seed` | `7` | Training seed; disjoint from eval seeds [1000, 1001, …] |
| `--obs-preset` | `operational` | Richer per-store observation (coverage, utilization, demand rate); better signal for coverage-radius control than `standard` |
| `--ent-coef` | `0.01` | Entropy bonus enabling exploration of continuous radius action. Prior collapse at `ent_coef=0.0` is consistent with the agent immediately committing to r=0 and never leaving. |
| `--output-dir` | `outputs/balanced_1M` | Writes `ppo_delivery.zip`, `learning_curve.csv`, `learning_curve.png` |

**Reward (overridden from YAML):** `cost_aware` with defaults `delivery_reward=1.0, w_fail=0.5, w_cost=0.01`.  
The `w_cost=0.01` coefficient is deliberately weak so cost is measurable but doesn't
overwhelm the delivery signal — the interior optimum sits at r≈700, not lower.

**ACTION NORMALIZATION (critical — harness fix, not env change):**  
SB3 PPO's MlpPolicy initialises the action net with gain=0.01 → initial mean ≈ 0  
in native coordinates. With std=1 in action space [0, 1000], initial sampled actions  
cluster at radius ≈ 0.4 → effectively r=0 → all orders fail → gradient near zero → collapse.  
Fix applied in train.py: `RescaleAction(-1, 1)` wrapper maps policy output [-1,1]  
to env action [0, 1000]. With this wrapper, initial mean=0 → radius=500; std=1 covers  
the full action range symmetrically. The env itself is UNCHANGED.  
evaluate.py mirrors this wrapper so trained model actions are interpreted correctly.

**Compute estimate:** ~8–20 min on a modern CPU (MLP policy, no GPU needed).

## Eval command (run after training)

```bash
python scripts/evaluate.py \
  --scenario scenarios/balanced.yaml \
  --model-path outputs/balanced_1M/ppo_delivery \
  --n-eval-seeds 10 \
  --obs-preset operational
```

## Known optimum (the falsifiable target)

The balanced scenario has a diagnosed leverage curve:
- `r=0` → 0 % delivery / 100 % failure (no orders covered)
- `r≈700` → ~82 % delivery_rate — **the empirical peak**
- `r=1000` → ~80 % delivery_rate — lower than r=700 because `warehouse_a`
  (alphabetical-first) absorbs all orders at full radius; effective courier cycle
  rises to ~756 s; throughput falls

The `cost_aware` reward additionally penalises the higher real courier cost at
large radius, so the **trained agent's optimal action is r≈700, not r=max and not r=0**.

A naive "maximize coverage" agent scores worse than the correct interior-optimum agent.

## How to read the results

### Learning curve CSV (`outputs/balanced_1M/learning_curve.csv`)

- **Healthy convergence:** `mean_episode_return` rises over the first ~100 rollouts,
  then flattens or oscillates near a positive value.
- **Collapse signature:** return flat at ~0 or drifts negative from rollout 1 —
  agent set r=0 from the start and never explored.
- **Oscillation / overshooting:** return rises then crashes — try lowering
  `learning_rate` or increasing `n_steps`.

### Eval table verdict thresholds

| Verdict | Criterion |
|---|---|
| **SUCCESS** | `mean_output_radius` ∈ [500, 900] **AND** `episode_return` > `baseline_max_1000` |
| **PARTIAL** | Beats `baseline_min_0` / `baseline_mid_500` but ties `baseline_opt_700` or `baseline_max_1000` |
| **COLLAPSE** | `mean_output_radius` < 50 **OR** `delivery_rate` ≈ `baseline_min_0` |

### Failure-mode checklist

1. **COLLAPSE, radius→0:** `ent_coef` still 0 or reward signal too sparse — increase `ent_coef` to 0.05.
2. **PARTIAL, radius→max:** `w_cost` too weak; cost term not penalising large radius enough — try `w_cost=0.05`.
3. **PARTIAL, radius mid-range but DR < opt:** undertrained — try 5M timesteps.
4. **SUCCESS, radius ≈ 700:** training worked as expected.

## Hyperparameters (documented starting points, NOT tuned)

| Param | Value | Note |
|---|---|---|
| `n_steps` | 2048 | Rollout buffer (SB3 default) |
| `batch_size` | 64 | Minibatch size |
| `n_epochs` | 10 | Gradient passes per rollout |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_range` | 0.2 | PPO clip coefficient |
| `learning_rate` | 3e-4 | Adam LR |
| `ent_coef` | **0.01** | **Raised from 0.0; critical for exploration** |
| `vf_coef` | 0.5 | Value-function loss weight |
| `max_grad_norm` | 0.5 | Gradient norm clip |

## Files produced

- `outputs/balanced_1M/ppo_delivery.zip` — trained model
- `outputs/balanced_1M/learning_curve.csv` — per-rollout KPIs
- `outputs/balanced_1M/learning_curve.png` — learning curve plot (requires matplotlib)

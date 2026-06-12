# Increment B-realistic: Requirements

**Status:** Draft  
**Date:** 2026-06-12  
**Scope:** Demand patterns, store fulfillment capacity, probabilistic returns, scenario
presets, and minimal UI visibility — as a coherent realism layer on top of the existing
validated RL research environment.

---

## 1. Problem Statement

The current simulator produces a flat, stationary delivery world: demand arrives at a
constant Poisson rate, stores prepare orders in a fixed 30-second burst with no
queuing, every delivery attempt succeeds, and the only way to vary the scenario is to
hand-tune individual YAML knobs.

This limits the research questions a practitioner can ask. An agent trained on flat
demand and frictionless stores has never encountered:
- A lunch rush or end-of-day surge that temporarily overloads the system
- A store that is the bottleneck, not the courier fleet
- An order that returns mid-episode and re-enters the system

Without these, the trained policy is validated on only one kind of world, and a
researcher has no principled way to study how coverage-radius decisions change under
varying load shapes or compounded bottlenecks.

Increment B-realistic introduces four capabilities that together make the world richer
without adding new decision-making agents. Every capability must be researchable: it
must be observable via metrics, reproducible under a fixed seed, and switchable via a
single named preset.

---

## 2. Capability Requirements

### 2.1 Scenario Presets

**User-facing need**  
A researcher selects a named scenario preset ("light", "balanced", "heavy") and
immediately obtains a coherent, internally consistent world configuration — demand
pattern, store throughput, and return rate all set together. No manual cross-knob
tuning is required to get a well-posed episode.

**Acceptance criteria**

| # | Criterion |
|---|-----------|
| P-1 | The user can specify a preset by name (a single identifier in the scenario config or CLI flag); the system applies the full corresponding bundle of settings. |
| P-2 | Running "light" produces strictly lower `courier_utilization` and `mean_delivery_time` than "balanced" under the same seed and episode length. |
| P-3 | Running "heavy" produces strictly higher `mean_delivery_time` and `store_queue_depth` than "balanced" under the same seed and episode length. |
| P-4 | (Stretch) In the visual app, the user can switch the active preset by name without restarting the simulation; the episode restarts from the new config with a fresh seed. |

**Proving metrics**  
`courier_utilization` (existing), `mean_delivery_time` (existing), `store_queue_depth`
(new — see §2.3).

---

### 2.2 Time-Varying Demand

**User-facing need**  
Demand should have a *shape* across the episode, not just a level. A researcher
studying an agent's response to load transitions — ramp-up, peak saturation, recovery
— needs to specify that shape independently from how intense it is. At minimum, three
patterns must be available: a flat constant (the current behavior), a daily profile
with identifiable peaks and troughs, and a burst pattern with concentrated surges
separated by quiet intervals.

**Pattern definitions (qualitative)**

- **constant**: Order arrivals are uniformly distributed across the episode duration.
  This is current behavior; it must remain as a regression-free baseline.

- **daily-profile**: The arrival rate follows a within-episode cycle with two or more
  identifiable peaks (e.g., a lunch surge and a late-afternoon surge) and a quiet
  period between them. The total order volume across the episode is controlled by the
  intensity scale (see below), not by the shape itself. A researcher can set
  "daily-profile at 2× intensity" to get twice the orders with the same shape.

- **burst**: Long quiet periods are punctuated by one or more short, concentrated
  spikes where the arrival rate is many times the background rate. Between spikes the
  system is under-loaded; during a spike it may be transiently over-loaded. The
  duration and spacing of bursts are configurable.

**Intensity scale**  
An independent multiplier applies uniformly to the peak rate of the chosen pattern.
Doubling intensity doubles the total expected order count for the episode, regardless
of pattern. This lets a researcher change "how much" without changing "what shape."

**Acceptance criteria**

| # | Criterion |
|---|-----------|
| D-1 | With pattern=constant, behavior is identical to the current simulator (no regression in `total_orders` distribution or episode metrics). |
| D-2 | With pattern=daily-profile, the order-arrival density in designated peak windows is statistically higher than in off-peak windows within the same episode (verifiable by partitioning order `created_at` timestamps). |
| D-3 | With pattern=burst, there exist contiguous time windows with zero or near-zero arrivals and contiguous windows with arrival rates that exceed the off-peak rate by a configurable factor (verifiable from order timestamps). |
| D-4 | Doubling the intensity scale doubles the expected `total_orders` for the episode (within Poisson sampling noise; the mean value must double). |
| D-5 | Two runs with the same pattern, intensity, and seed produce identical order-arrival sequences. |

**Proving metrics**  
`total_orders` (existing), order `created_at` timestamps partitioned by window (derived
from existing order records), `delivery_rate` (existing; should degrade during peaks
under heavy load).

---

### 2.3 Store Fulfillment Capacity

**User-facing need**  
A store should have finite throughput, making it a potential bottleneck. When demand
arrives faster than a store can prepare orders, a queue forms: couriers wait at the
store, prep/wait times grow, and end-to-end delivery times climb. A researcher needs to
be able to distinguish "courier-limited" episodes from "store-limited" episodes.

Two aspects require specification:

**a) Configurable prep time**  
Prep time is currently fixed at a hardcoded 30 seconds and is not reachable via YAML.
That must change. Each store should have a prep time expressible in the scenario config.
Faster prep = store never the bottleneck; slower prep = store is easily saturated. This
is in scope for B-realistic because it is a prerequisite for meaningful capacity
experiments.

**b) Throughput capacity with queuing**  
A store's capacity sets how many orders it can be actively preparing simultaneously
(parallel slots). When all slots are full, new arriving couriers wait in a store queue.
Each queued courier departs only when a slot frees and the order's prep completes.
Total wait time = queue wait + prep time. The store is the bottleneck when the store
queue is non-empty for a sustained fraction of the episode.

**Acceptance criteria**

| # | Criterion |
|---|-----------|
| C-1 | `prep_time` is configurable per store in the scenario config and affects the time between a courier arriving at the store and the order being ready for pickup. |
| C-2 | Increasing `prep_time` from its current default increases observed `mean_pickup_latency` and `mean_delivery_time` proportionally. |
| C-3 | Under high demand with low store capacity, `store_queue_depth` grows above zero during the episode and `mean_store_wait_time` is measurably positive. |
| C-4 | Under low demand with high store capacity, `store_queue_depth` remains zero throughout the episode and `mean_store_wait_time` is zero. |
| C-5 | Orders queued at a store do not fail due to queueing alone; they wait and are eventually prepared (unless the episode horizon is reached). |
| C-6 | Changing store capacity changes observable `mean_store_wait_time` and `store_queue_depth`; no store capacity setting exists that changes neither metric under any load level. |

**Proving metrics**  
`mean_pickup_latency` (existing), `p95_delivery_time` (existing), plus new metrics:
- `store_queue_depth`: the time-averaged number of couriers waiting for a prep slot
  across all stores in the episode
- `mean_store_wait_time`: mean time an order spends in the store queue before prep
  begins (zero when queue never forms)

---

### 2.4 Probabilistic Returns

**User-facing need**  
When a courier reaches a customer, there is a small configurable probability that the
delivery attempt fails at the door — the customer is unavailable, refuses, or any other
real-world non-delivery reason. The order does not simply disappear: it is marked as
RETURNED, the courier carries it back toward the originating store, and the outcome is
tracked separately from both DELIVERED and FAILED orders. This is a distinct outcome
with its own lifecycle and metrics.

**Return lifecycle**

1. A courier arrives at the customer location (leg-2 complete, as today).
2. With probability `return_rate` (a configurable float in [0, 1)), the delivery attempt
   fails at the door.
3. The order transitions to the new **RETURNED** terminal state. It is not counted as
   delivered.
4. The courier enters a new **returning** phase: it travels back toward the originating
   store carrying the undelivered order.
5. Upon the courier's return to the store, the order is recorded as returned. The courier
   becomes free for new assignments.
6. The return leg distance is tracked for cost accounting.

**What "returned" means to the researcher**  
A returned order represents a full wasted courier cycle (two outbound legs + a return
leg) with no successful delivery. High return rates materially increase courier load and
could shift the agent's optimal coverage-radius decision. The researcher must be able to
set return rate per scenario and observe its effect on delivery efficiency metrics.

**Acceptance criteria**

| # | Criterion |
|---|-----------|
| R-1 | With `return_rate=0.0`, no orders enter the RETURNED state (`returned_orders == 0` at episode end). |
| R-2 | With `return_rate=1.0`, every order that reaches a customer transitions to RETURNED; `delivered_orders == 0`. |
| R-3 | With `return_rate=0.01` and at least 300 delivery attempts in the episode, the observed rate `returned_orders / delivery_attempts` lies within a statistically plausible range of 1% (e.g., [0.0%, 4.0%] — exact bounds are the architect's to derive from Binomial tail probabilities). |
| R-4 | Returned orders are not counted in `delivered_orders` and do not contribute to `delivery_rate`. |
| R-5 | Returned orders are not counted in `failed_orders`; they appear only in `returned_orders`. |
| R-6 | The courier performing a return is unavailable for new assignments during the return leg. |
| R-7 | The return-decision draw is fully seed-deterministic: same (config, seed) produces identical return events across runs. |
| R-8 | All returned-order metrics are present in the episode summary returned by the KPI collector. |

**New metrics**

| Metric | Definition |
|--------|-----------|
| `returned_orders` | Count of orders that entered the RETURNED state in the episode |
| `return_rate` | `returned_orders / (delivered_orders + returned_orders)` — the fraction of attempted deliveries that did not succeed at the door |
| `mean_return_leg_time` | Mean time from the failed delivery attempt to the courier's arrival back at the store |

**Preserving the existing failed_orders metric**  
`failed_orders` continues to count only orders that fail before or without a delivery
attempt (uncovered customer, courier never available, etc.). It must not subsume
RETURNED orders.

---

### 2.5 UI Visibility

**User-facing need**  
A researcher running the visual app should be able to identify entity states at a glance
without reading raw metric printouts. Two things must be visible: (a) what kind of
entity am I looking at and what phase is it in, and (b) what is the current load and
scenario state of the world.

**Entity visibility needs**

- Stores and couriers must be visually distinct from each other and from customer
  locations.
- A courier should look different depending on its phase: idle/free, traveling to
  store (leg 1), waiting at store, traveling to customer (leg 2), and returning with
  an undelivered order. Five distinguishable states.
- A store with a non-empty prep queue should be visually distinguishable from a store
  with an empty queue (queue buildup is a key phenomenon in heavy scenarios).

**World-state visibility needs**

- The active scenario name and demand pattern must be readable in the visual app during
  a run (so screenshots and recordings are self-documenting).
- The current demand intensity level (relative to the pattern's peak) should be
  readable — so a researcher can see "we are currently in the surge window."

**Acceptance criteria**

| # | Criterion |
|---|-----------|
| U-1 | A human observer viewing the visual app can identify stores, couriers (across all five phases), and pending customer locations without consulting documentation. |
| U-2 | A store with a queue depth > 0 is visually distinct from a store with queue depth = 0. |
| U-3 | The active scenario name and demand pattern type are displayed persistently in the visual app during an episode. |
| U-4 | The current demand intensity relative to the episode timeline is visible (e.g., a HUD indicator showing current rate vs. peak rate). |

**Note on scope**: This section states the product need. The Graphic Designer specifies
the exact visual treatment. No icon design, color scheme, sprite layout, or HUD
placement is mandated here.

---

## 3. Scenario Preset Table

The table below describes each preset qualitatively. Exact numeric values (rates,
capacities, prep times, return probabilities) are the Software Architect's and
Engineer's to derive from the system's capacity equations — as the balanced.yaml
comment already documents for the current scenario. The qualitative descriptions are
the binding acceptance anchor.

| Preset | Demand shape | Intensity | Store behavior | Return rate | What it feels like |
|--------|-------------|-----------|----------------|-------------|---------------------|
| **light** | constant | low | Ample capacity; stores never queue; fast prep | Near zero (trace) | Couriers are often idle. The store is never a bottleneck. Orders arrive slowly enough that a poorly-tuned coverage radius barely hurts. Used to verify the agent learns at all and to establish a low-friction baseline. |
| **balanced** | constant | moderate | Adequate capacity; brief queues under coincident arrivals; moderate prep | ~1% | Couriers work most of the time but have slack. Stores absorb demand with only occasional brief queuing. Returns happen rarely but accumulate over a long episode. The current research baseline; agent policy is meaningfully differentiated. |
| **heavy** | daily-profile or burst | high | Tight capacity; sustained queues during peaks; slower prep | ~3% | Couriers are almost always busy. Stores regularly queue multiple orders simultaneously; `mean_store_wait_time` is measurably non-zero. Return events are frequent enough to noticeably reduce effective throughput. Delivery times climb and p95 diverges from mean. Used to stress-test the agent under compounded bottlenecks. |

---

## 4. Reproducibility and No-Inert-Knobs

These are product-level acceptance criteria that apply to every setting introduced by
this increment. They are not optional:

**Reproducibility requirement**  
Given the same scenario configuration and seed, two independent runs of any B-realistic
scenario must produce byte-for-byte identical episode metrics. This includes demand
shape, order arrivals, store queue sequences, and return events. Any new source of
randomness (return decision, burst timing, profile interpolation) must draw exclusively
from a seeded RNG stream that is a deterministic child of the episode seed — never from
a global or time-based random source.

Acceptance criterion: `kpi_collector.summary()` is identical across two runs of the
same (config, seed) pair, for all three presets.

**No-inert-knobs requirement**  
Every new configurable setting introduced in B-realistic must change at least one
observable episode metric when changed. If a setting can be toggled without altering any
metric in any scenario, it must not exist.

Acceptance criteria (per setting):

| Setting | Must affect |
|---------|------------|
| Demand pattern | `total_orders` distribution shape within-episode (verifiable from timestamps) |
| Intensity scale | `total_orders` expected value |
| Burst duration / spacing | Time-windowed order density |
| `prep_time` | `mean_pickup_latency` |
| Store capacity (parallel slots) | `store_queue_depth` and `mean_store_wait_time` under appropriate load |
| `return_rate` | `returned_orders` and `return_rate` metric |

---

## 5. Out of Scope for This Increment

The following are explicitly deferred to future increments. Any design or code that
begins implementing these during B-realistic must be flagged as scope creep:

- **Customer purchasing strategies or store-selection logic**: customers arrive at a
  random location; how they "choose" a store is not modeled. Order routing remains the
  engine's responsibility (first covering store by ID).
- **Order-value modeling**: no revenue, pricing, or customer lifetime value.
- **In-store fulfillment agents**: the store prepares autonomously according to its
  throughput configuration. No RL or rule-based agent governs in-store preparation
  decisions.
- **Biker-manager as a decision-making agent**: the dispatcher remains a heuristic
  (first available courier). No new agent controls courier-to-order assignment.
- **Multi-agent RL**: the existing single-agent coverage-radius controller is the only
  RL agent.
- **Road-graph routing**: Euclidean routing is the sole routing model in B-realistic
  (road-graph is slated for a later step).
- **Courier heterogeneity beyond existing YAML fields**: new courier types, speed
  distributions, or capacity tiers are out of scope.
- **Returned orders re-entering the order queue as new orders**: a returned order is a
  terminal outcome (RETURNED state). Whether the customer re-orders is not modeled.

---

## 6. Handoff Notes

### For the Software Architect

B-realistic requires four design decisions that have non-trivial cross-layer
consequences:

1. **Demand pattern abstraction**: the `DemandGenerator` contract (`next_event`) is the
   right extension point, but time-varying demand requires the generator to be aware of
   `sim_time` relative to the episode horizon — something the current Poisson
   implementation ignores. The architect must design how the generator receives the
   episode duration (to compute normalized time) and how pattern + intensity are
   parameterized in the config schema without breaking the existing
   `PoissonDemandGenerator` as the constant-pattern implementation.

2. **Store fulfillment queue**: `BuiltinStore.start_preparation` currently does not
   queue — it records and returns `sim_time + prep_time` unconditionally. The architect
   must design how a store queue is represented and how the simulator event model
   handles the "courier waiting at store for a slot" state. A new event type (e.g.,
   `store_slot_available`) or a revised `ORDER_READY` scheduling path will be needed.
   The existing `can_prepare` guard is a natural hook but is currently uncalled by the
   simulator. `prep_time` must also be promoted from a hardcoded default to a
   per-store YAML field in `ScenarioConfig`.

3. **Probabilistic returns and new order state**: the order state machine currently has
   no path from `IN_TRANSIT` to anything but `DELIVERED` or `FAILED`. A new `RETURNED`
   terminal state is needed, along with a new courier phase (`returning`). The return
   decision RNG stream must be spawned as a new child of the episode `SeedSequence`
   (appended at the end of the existing spawn list to preserve seed-compatibility of
   existing scenarios with `return_rate=0`). The return leg must be accounted in cost
   metrics.

4. **Scenario preset system**: a preset is a named bundle of settings, not a file
   hierarchy. The architect must decide whether presets are compiled in (as typed
   dataclasses) or loaded from a presets registry file, and how a user-specified preset
   interacts with any per-field overrides in a custom YAML. The key constraint: a preset
   name must uniquely determine all B-realistic settings; no preset should leave any new
   setting at an ambiguous default.

### For the Graphic Designer

B-realistic introduces three new visual problems for the pygame renderer:

1. **Courier phase icons**: couriers now have five distinct operational states — idle,
   en-route to store, waiting at store, en-route to customer, and returning with an
   undelivered order. Each state should be visually unambiguous at a glance; the
   returning state is new and carries a different semantic (backward travel) that should
   read differently from the two forward-travel states.

2. **Store load indicator**: a store with a non-empty prep queue must look different
   from a store with an empty queue. The indicator must convey magnitude, not just
   binary (queue depth 1 vs. queue depth 8 should be distinguishable). Consider a
   badge, bar, or color shift — the specific treatment is yours to decide.

3. **Scenario / demand HUD**: the visual app needs a persistent overlay showing (a)
   the active scenario name and demand pattern type, and (b) a real-time indicator of
   current demand intensity relative to the pattern's peak. This could be a mini
   time-series bar, a label, or a status line — the requirement is that the information
   is present and readable without pausing the simulation. The HUD must not occlude
   the simulation area's primary entities.

The existing pygame renderer in `src/delivery_sim/render/pygame_renderer.py` is the
implementation target. The designer should produce annotated mockups (not code) that the
engineer can implement against.

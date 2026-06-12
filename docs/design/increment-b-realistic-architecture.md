# Increment B-realistic: Architecture Design

**Status:** Draft  
**Date:** 2026-06-12  
**Implements:** `docs/requirements/increment-b-realistic.md`  
**Architect rule:** Every design decision here must be traceable to a requirement
acceptance criterion. Every new config field must reach a real event. No redesign —
additive extension only.

---

## 0. Reading Map

| Section | What it answers |
|---------|----------------|
| §1 Design Philosophy | Constraints that govern every decision below |
| §2 Scenario Presets | How named presets are represented, selected, and (stretch) switched |
| §3 Time-Varying Demand | New DemandGenerator implementations; intensity multiplier |
| §4 Store Fulfillment Queue | Event-driven queueing; prep_time exposure |
| §5 Probabilistic Returns | Order state machine extension; return lifecycle; RNG stream |
| §6 Metrics | All new KPI fields; event-derivation path for each |
| §7 Snapshot Additions | New snapshot fields for the renderer |
| §8 Contract Amendments | Consolidated before/after for every amended contract |
| §9 RNG Streams | Authoritative spawn-index table |
| §10 Reproducibility & No-Inert-Knobs Audit | Per-knob traceability |
| §11 Build Step Sequence | Ordered steps with DoD; amendment steps flagged |

---

## 1. Design Philosophy

**Extend, don't redesign.** The event queue, registry, seeded-RNG spawn scheme,
`Order.transition()` chokepoint, `_on_order_terminal()` chokepoint, and snapshot
observer-invariance are all load-bearing. This design touches each only at its
documented extension seam.

**New time-based behavior = new events.** Every delay that matters (store queue wait,
return leg travel) is represented as a scheduled event in the event queue, not as
polling or timer state. This is what makes metrics dt-independent.

**Every new random draw = its own child stream.** The master `SeedSequence` spawns
one child per stochastic consumer, indexed by stable position. New consumers append
to the tail; existing indices are never reused.

**Backward compatibility baseline.** Any scenario YAML that loads and runs correctly
today must load and run correctly after B-realistic with `return_rate=0` (default),
`intensity=1.0` (default), and default store `prep_time`. The KPI fingerprint of
`balanced.yaml` must be preserved exactly, verified by a regression test.

---

## 2. Scenario Presets

### 2.1 Representation

A preset is a complete scenario YAML file in `scenarios/presets/`. There is no new
abstraction layer — the preset system is the existing YAML loader, applied to a
curated, committed set of files:

```
scenarios/
  presets/
    light.yaml
    balanced.yaml     ← current scenarios/balanced.yaml promoted here (or symlinked)
    heavy.yaml
```

Each preset YAML includes all B-realistic fields (`return_rate`, `prep_time` per
store, demand `intensity`, demand `generator_type`, etc.) set to values that produce
the qualitative behavior described in the requirements preset table.

The `balanced.yaml` preset must reproduce the existing validated training result
(mean_output_radius=676, return=69.63 > baseline_max). It is therefore the upgrade
of the current `scenarios/balanced.yaml`, not a replacement — its non-B-realistic
fields are unchanged.

**No config merging, no override system.** Each preset is a self-contained YAML.
Users who want to experiment (e.g., heavy scenario at lower return rate) copy the
preset and edit. This avoids introducing a merge/override mechanism that is not
needed by the requirements.

### 2.2 Selection

The existing `--scenario` CLI flag already accepts any YAML path. Preset selection
is:
```
python scripts/train.py --scenario scenarios/presets/heavy.yaml
```

A thin `--preset NAME` shortcut (resolves to `scenarios/presets/{NAME}.yaml`) can
be added to CLI scripts as a convenience. This is a script-level change, not an
engine change.

### 2.3 On-the-fly switching (stretch, eval/demo only)

**Determinism warning:** mid-episode preset switching is incompatible with
reproducibility. An episode's RNG state at step N depends on every random draw from
step 0 onward. Switching configs mid-episode would fork the episode history in a way
that cannot be reproduced from (new_config, seed) alone.

**Design constraint:** the visual app's "switch preset" action is always an **episode
restart** (call `sim.reset()` / `env.reset()` with the new config), never a mid-run
config mutation. The user experiences a seamless transition; the system starts a
fresh, fully reproducible episode. Flag in the UI: "Switching preset will restart
the episode."

Implementation surface: the pygame renderer receives a new optional
`switch_preset(name)` callback that the app wires to a key press or menu action.
The callback is invoked between episodes, never during `sim.run_until()`.

---

## 3. Time-Varying Demand

### 3.1 Design Principle

`DemandGenerator.next_event(sim_time, rng)` is already the correct extension seam:
it schedules the next arrival by drawing from `rng` and returning an absolute time.
Time-varying demand is just a different implementation of this method where the
inter-arrival delay depends on `sim_time` relative to the episode horizon. The ABC
does not change structurally; one parameter is added to `reset()`.

### 3.2 DemandGenerator ABC amendment (additive)

**Amendment DA-1 — `reset` signature:**

```python
# Before:
@abstractmethod
def reset(self, rng: np.random.Generator) -> None: ...

# After:
@abstractmethod
def reset(self, rng: np.random.Generator, horizon: float = 0.0) -> None: ...
```

`horizon` is the episode duration (= `max_steps * dt`). Time-varying generators
use it to compute `time_fraction = sim_time / horizon`. Stateless generators
(PoissonDemandGenerator) accept and ignore it; their reset bodies are no-ops and
require no change beyond the signature.

**How horizon is passed:** In `Simulator.reset()`, compute `self._horizon` before
calling `create(...)` for the demand generator. Add `horizon=self._horizon` to the
generator constructor kwargs. Additionally call `self._demand_generator.reset(self._gen_rng, horizon=self._horizon)`.

Both constructor-time and reset-time injection are used:
- Constructor: the generator may pre-compute a profile schedule at startup.
- Reset: the generator re-initialises internal state for the new episode (correct for multi-episode RL).

### 3.3 Intensity multiplier

**Amendment DA-2 — `intensity` in DemandConfig:**

```python
class DemandConfig(BaseModel):
    generator_type: str
    rate: float = 1.0
    intensity: float = Field(default=1.0, gt=0.0)   # NEW
    # pattern-specific fields below (§3.4, §3.5)
```

Every generator multiplies its base `rate` by `intensity` before computing
inter-arrival times:

```
effective_rate = config.rate × config.intensity
```

`PoissonDemandGenerator.__init__` gains `intensity: float = 1.0` (used in
`next_event` as `rate * intensity`). Existing YAML without `intensity` loads
unchanged via the Pydantic default.

**No-inert-knob proof:** `intensity=2.0` halves inter-arrival delay → doubles
expected `total_orders`. Verified by D-4 acceptance criterion.

### 3.4 DailyProfileDemandGenerator

**Registered as:** `"DailyProfileDemandGenerator"`

**New DemandConfig fields (additive, all defaulted):**

```python
class ProfileBreakpoint(BaseModel):
    time_fraction: float   # in [0.0, 1.0]; fraction of episode elapsed
    rate_factor: float     # multiplier on effective_rate at this point

class DemandConfig(BaseModel):
    ...
    profile: list[ProfileBreakpoint] = Field(default_factory=list)  # NEW
```

If `profile` is empty, the generator falls back to constant-rate Poisson (same as
`PoissonDemandGenerator`). This makes `DailyProfileDemandGenerator` a strict
superset.

**Rate function:** piecewise-linear interpolation between breakpoints, sorted by
`time_fraction`. The rate at `sim_time` is:

```
t_frac = sim_time / horizon
current_rate = effective_rate × piecewise_linear(profile, t_frac)
```

At the last breakpoint and beyond, the rate is held constant at the final value.

**next_event implementation:**  
At `sim_time`, compute `current_rate = effective_rate × factor(sim_time/horizon)`.
If `current_rate == 0`, step forward to the next breakpoint where rate > 0.
Otherwise, draw `delay ~ Exponential(1 / current_rate)` from `rng` and return
`sim_time + delay`.

**Note:** This is an approximation (the rate changes between draw and arrival time).
For research purposes this is sufficient — the piecewise constant approximation
matches the stated product need of "identifiable peaks and troughs." A future
increment could use thinning/superposition for exact non-stationary Poisson if
precision matters.

**RNG usage:** only `gen_rng` (child[0]). Draw order per event: (1) delay,
(2) customer_x, (3) customer_y. Same as `PoissonDemandGenerator`.

### 3.5 BurstDemandGenerator

**Registered as:** `"BurstDemandGenerator"`

**New DemandConfig fields (additive, all defaulted):**

```python
class DemandConfig(BaseModel):
    ...
    burst_rate_factor: float = Field(default=5.0, gt=1.0)          # NEW
    burst_duration_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)  # NEW
    burst_interval_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)  # NEW
```

**Burst schedule:** fully deterministic from config (no RNG draws for timing).
Given horizon H:
- Burst windows are at fixed offsets: burst starts at `burst_interval_fraction × H`,
  `2 × burst_interval_fraction × H`, etc. until the end of the episode.
- Each burst lasts `burst_duration_fraction × H` sim-seconds.
- During a burst: rate = `effective_rate × burst_rate_factor`
- Outside a burst: rate = `effective_rate`

Deterministic burst timing means the generator draws from `gen_rng` only for order
positions and inter-arrival delays, never for burst placement. This keeps the full
RNG draw sequence reproducible and requires no additional child stream.

**next_event implementation:** identical structure to `DailyProfileDemandGenerator`
but using the deterministic burst schedule as the rate function.

**Minimum config validation:** `burst_duration_fraction + burst_interval_fraction ≤ 1.0`
enforced by Pydantic validator (bursts cannot overlap).

### 3.6 Simulator wiring

In `Simulator.reset()`, the demand generator is already created via `create(...)`.
The additions are:

```python
# After amendment (additions highlighted with ← NEW)
horizon = self.config.max_steps * self.config.dt  # ← moved earlier
self._demand_generator = create(
    "demand_generator", self.config.demand.generator_type,
    rate=self.config.demand.rate,
    intensity=self.config.demand.intensity,      # ← NEW
    dt=self.config.dt,
    world_width=self.config.world.width,
    world_height=self.config.world.height,
    store_ids=[s.store_id for s in stores],
    horizon=horizon,                              # ← NEW
    profile=self.config.demand.profile,          # ← NEW
    burst_rate_factor=self.config.demand.burst_rate_factor,       # ← NEW
    burst_duration_fraction=self.config.demand.burst_duration_fraction, # ← NEW
    burst_interval_fraction=self.config.demand.burst_interval_fraction, # ← NEW
)
self._demand_generator.reset(self._gen_rng, horizon=horizon)  # ← horizon added
self._horizon = horizon  # ← was set here before; still valid (already computed)
```

Generators that don't use a kwarg (e.g., `PoissonDemandGenerator` ignores `profile`,
`burst_*`, `horizon`) accept and discard it via `**kwargs` in their constructors, or
store-but-don't-use it. Add `**_ignored: Any` to `PoissonDemandGenerator.__init__`
for forward compat.

---

## 4. Store Fulfillment Capacity and Queuing

This is the most significant engine change in B-realistic. The design uses three
principles:
1. The store is the authority on its slot occupancy and waiting queue; it never
   schedules events.
2. The simulator schedules all events. When a slot frees, it dequeues the next
   waiter and schedules their `order_ready` event.
3. The queue forms at `courier_arrived_store` and drains in `_handle_order_ready`.
   No new event type is needed.

### 4.1 StoreConfig amendment (additive)

```python
class StoreConfig(BaseModel):
    name: str
    x: float
    y: float
    capacity: int = 10
    prep_time: float = Field(default=30.0, gt=0.0)   # NEW — was hardcoded
    coverage_radius: float = Field(default=500.0, ge=0.0)
```

`prep_time` defaults to 30.0, preserving current behavior for all existing YAMLs.

### 4.2 Store ABC amendment (additive)

Four new abstract methods are added. All existing `Store` subclasses must implement
them (only `BuiltinStore` exists in the codebase):

```python
class Store(ABC):
    # --- EXISTING methods unchanged ---

    # --- NEW abstract methods ---

    @abstractmethod
    def enqueue_waiter(
        self, courier_id: str, order_id: str, arrived_at: float
    ) -> None:
        """Add a courier to the store's waiting queue (capacity was full at arrival).

        arrived_at is the sim_time when the courier arrived; used to compute
        wait duration when the waiter is later dequeued.
        """
        raise NotImplementedError

    @abstractmethod
    def dequeue_next_waiter(self) -> tuple[str, str, float] | None:
        """Pop the next waiting courier and return (courier_id, order_id, arrived_at).

        Returns None if the queue is empty. Called by the Simulator in
        _handle_order_ready after a prep slot is freed.
        """
        raise NotImplementedError

    @abstractmethod
    def complete_preparation(self, order_id: str) -> None:
        """Remove order_id from active_orders, freeing one prep slot.

        Called by the Simulator in _handle_order_ready, before dequeue_next_waiter.
        Must be idempotent if order_id is not present (defensive).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def queue_depth(self) -> int:
        """Current number of couriers waiting for a prep slot. Zero is normal."""
        raise NotImplementedError
```

### 4.3 BuiltinStore implementation

```python
from collections import deque

class BuiltinStore(Store):

    def __init__(self, ..., prep_time: float = 30.0, ...) -> None:
        ...
        self.prep_time = prep_time           # was hardcoded 30.0; now configurable
        self._active_orders: dict[str, float] = {}  # unchanged
        # NEW:
        self._waiting: deque[tuple[str, str, float]] = deque()
        # Each entry: (courier_id, order_id, arrived_at)

    def enqueue_waiter(self, courier_id, order_id, arrived_at):
        self._waiting.append((courier_id, order_id, arrived_at))

    def dequeue_next_waiter(self):
        return self._waiting.popleft() if self._waiting else None

    def complete_preparation(self, order_id):
        self._active_orders.pop(order_id, None)

    @property
    def queue_depth(self) -> int:
        return len(self._waiting)

    def reset(self) -> None:
        self._active_orders.clear()
        self._waiting.clear()           # NEW line added to existing reset
```

### 4.4 Simulator event flow changes

**`_handle_courier_arrived_store` (amended):**

```
BEFORE (current):
  store.start_preparation(order_id, sim_time)    # always succeeds, no guard
  order.transition(PREPARING, sim_time)
  schedule order_ready at sim_time + store.prep_time

AFTER:
  if store.can_prepare(order_id):
    ready_time = store.start_preparation(order_id, sim_time)
    order.transition(PREPARING, sim_time)
    courier_phase[courier_id] = "at-store"      # unchanged
    schedule order_ready(courier_id, order_id, store_id) at ready_time
  else:
    store.enqueue_waiter(courier_id, order_id, arrived_at=sim_time)
    courier_phase[courier_id] = "waiting-at-store"   # NEW phase value
    collector.on_order_queued_at_store(order_id, store_id, sim_time)  # NEW
    # No event is scheduled; the slot will free at the next order_ready
```

**`_handle_order_ready` (amended, new slot-free + dequeue block):**

```
BEFORE (current):
  order.transition(PICKED_UP, sim_time)
  order.transition(IN_TRANSIT, sim_time)
  courier.assign(... leg2 ...)
  schedule courier_arrived_customer

AFTER (new block inserted before the existing transition sequence):
  # Free the prep slot that just completed
  store.complete_preparation(order_id)           # NEW

  # Drain one waiting courier if any
  waiter = store.dequeue_next_waiter()           # NEW
  if waiter:
    w_courier_id, w_order_id, w_arrived_at = waiter
    w_order = world.active_orders[w_order_id]
    wait_time = sim_time - w_arrived_at
    w_ready_time = store.start_preparation(w_order_id, sim_time)
    w_order.transition(PREPARING, sim_time)
    courier_phase[w_courier_id] = "at-store"
    collector.on_order_dequeued_from_store(
        w_order_id, store_id, wait_time          # NEW
    )
    schedule order_ready(w_courier_id, w_order_id, store_id) at w_ready_time

  # Existing block (unchanged):
  order.transition(PICKED_UP, sim_time)
  order.transition(IN_TRANSIT, sim_time)
  courier.assign(... leg2 ...)
  schedule courier_arrived_customer
```

**Ordering proof:** `complete_preparation` is called before `dequeue_next_waiter`.
This ensures the newly freed slot is always available when start_preparation is
called for the waiter. `can_prepare` will return True because `len(active_orders) <
capacity` (just decremented). The ordering is invariant.

**Courier collector notifications for waiting couriers:**
The existing `on_courier_busy` is called when the courier is first dispatched
(in `_dispatch`). A courier waiting at a store is already "busy" in the utilisation
sense — it was counted as busy when assigned. No double-counting.

### 4.5 Backward compatibility

With the default `capacity=10` and existing scenarios' demand levels, `can_prepare`
always returns True throughout the episode. The `_waiting` deque is never populated.
`store.complete_preparation(order_id)` is called in `_handle_order_ready` but this
is a no-op in behaviour terms (it just removes from `_active_orders`, which was
previously never cleaned up — this is a silent bug fix, not a breaking change).

The KPI fingerprint of `balanced.yaml` is preserved because:
- `prep_time` defaults to 30.0 (current hardcoded value)
- `capacity=20` (ample for balanced demand)
- No waiter ever enqueues → `mean_store_wait_time = 0.0`

---

## 5. Probabilistic Returns

### 5.1 Order state machine amendment (additive)

**Amendment SM-1 — new `RETURNED` terminal state:**

```python
# Before:
class OrderStatus(Enum):
    CREATED = auto()
    ASSIGNED = auto()
    PREPARING = auto()
    PICKED_UP = auto()
    IN_TRANSIT = auto()
    DELIVERED = auto()
    FAILED = auto()

# After (one addition):
class OrderStatus(Enum):
    CREATED = auto()
    ASSIGNED = auto()
    PREPARING = auto()
    PICKED_UP = auto()
    IN_TRANSIT = auto()
    DELIVERED = auto()
    FAILED = auto()
    RETURNED = auto()    # NEW — terminal; order was refused at the customer's door
```

**Amendment SM-2 — allowed transitions:**

```python
# Before:
OrderStatus.IN_TRANSIT: frozenset({OrderStatus.DELIVERED, OrderStatus.FAILED}),

# After:
OrderStatus.IN_TRANSIT: frozenset({OrderStatus.DELIVERED, OrderStatus.FAILED,
                                    OrderStatus.RETURNED}),  # NEW path
OrderStatus.RETURNED: frozenset(),   # NEW — terminal (no outgoing)
```

**Amendment SM-3 — `is_terminal` property:**

```python
# Before:
@property
def is_terminal(self) -> bool:
    return self.status in (OrderStatus.DELIVERED, OrderStatus.FAILED)

# After:
@property
def is_terminal(self) -> bool:
    return self.status in (
        OrderStatus.DELIVERED, OrderStatus.FAILED, OrderStatus.RETURNED  # +RETURNED
    )
```

### 5.2 ScenarioConfig amendment (additive)

```python
class ScenarioConfig(BaseModel):
    ...
    return_rate: float = Field(default=0.0, ge=0.0, lt=1.0)   # NEW
```

Default 0.0 preserves existing behavior. Upper bound is strictly less than 1.0 to
keep at least one delivery attempt succeeding (a `return_rate=1.0` scenario is
allowed by requirement R-2 for testing; the `lt=1.0` bound should therefore be
`le=1.0` — see §10 discussion of R-2).

> **Correction:** accept `return_rate` in `[0.0, 1.0]` (inclusive). The Pydantic
> validator should use `ge=0.0, le=1.0`.

### 5.3 RNG stream (new child[1])

```python
# Before (Simulator.reset):
(gen_seed,) = np.random.SeedSequence(self.config.seed).spawn(1)
self._gen_rng = np.random.default_rng(gen_seed)

# After:
gen_seed, returns_seed = np.random.SeedSequence(self.config.seed).spawn(2)
self._gen_rng = np.random.default_rng(gen_seed)
self._returns_rng = np.random.default_rng(returns_seed)   # NEW
```

**Backward compatibility of spawn(2) vs spawn(1):**

`numpy.random.SeedSequence.spawn(n)` generates child seeds keyed by
`(parent_entropy, child_index)`. Child index 0 from `spawn(1)` and child index 0
from `spawn(2)` are derived identically because the child index, not the batch size,
determines the seed. This is verifiable with:

```python
import numpy as np
seed = 42
assert (np.random.SeedSequence(seed).spawn(1)[0].entropy ==
        np.random.SeedSequence(seed).spawn(2)[0].entropy)
```

This assertion must be added as a reproducibility smoke test (see Step B6). If it
fails on the project's numpy version, the design is invalidated and a workaround
(two separate `spawn(1)` calls on different parent seeds) must be adopted — but this
is not expected.

With `return_rate=0.0`, `self._returns_rng` is consumed zero times per episode.
The `gen_rng` sequence is therefore identical to pre-B-realistic, and all KPI
fingerprints are preserved.

### 5.4 Return lifecycle: event flow

**`_handle_courier_arrived_customer` (amended):**

```
BEFORE (current):
  order.transition(DELIVERED, sim_time)
  courier_phase[courier_id] = "free"
  _on_order_terminal(order, sim_time)
  collector.on_courier_free(courier_id, sim_time)

AFTER:
  if self.config.return_rate > 0.0:
    refused = self._returns_rng.random() < self.config.return_rate
  else:
    refused = False          # fast path; does not consume returns_rng

  if not refused:
    # Existing delivery path (unchanged):
    order.transition(DELIVERED, sim_time)
    courier_phase[courier_id] = "free"
    _on_order_terminal(order, sim_time)
    collector.on_courier_free(courier_id, sim_time)

  else:
    # Return path (NEW):
    return_store = world.store_index[order.store_id]

    # Add return leg cost before terminal transition
    return_dist = routing.distance(
        order.customer_x, order.customer_y,
        return_store.x, return_store.y
    )
    # courier is the assigned courier; cost() is the existing per-unit method
    _order_leg_cost[order_id] = (
        _order_leg_cost.get(order_id, 0.0) + courier.cost(return_dist)
    )

    order.transition(RETURNED, sim_time)
    # _on_order_terminal now handles RETURNED; pops cost, notifies collector
    _on_order_terminal(order, sim_time)

    # Assign return leg (courier travels from customer back to store)
    courier.assign(
        order_id=order_id,
        store_id=order.store_id,
        sim_time=sim_time,
        target_x=return_store.x,
        target_y=return_store.y,
        from_x=order.customer_x,
        from_y=order.customer_y,
    )
    courier_phase[courier_id] = "returning"    # NEW phase value

    eta = courier.arrival_time()
    assert eta is not None
    event_queue.push(Event(
        time=eta, priority=_PRI_ARRIVAL,
        event_type="courier_returned_to_store",   # NEW event type
        payload={
            "courier_id": courier_id,
            "order_id": order_id,
            "store_id": order.store_id,
        },
    ))
```

**Important ordering:** cost is accumulated **before** `_on_order_terminal`, which
pops it. `courier.assign()` is called **after** `_on_order_terminal`, because
assign() does not depend on the order's terminal state and the SETTLED precondition
holds (sim_time == arrival_time() for leg-2).

**New event handler `_handle_courier_returned_to_store`:**

```python
def _handle_courier_returned_to_store(self, event: Event) -> None:
    sim_time = self.clock.elapsed
    courier_id = event.payload["courier_id"]

    self.world.courier_phase[courier_id] = "free"
    if self._collector is not None:
        self._collector.on_courier_free(courier_id, sim_time)
        self._collector.on_courier_returned_to_store(courier_id, sim_time)
```

The store_id in the payload is available for store-level return metrics if needed in
a future increment. It is not used in B-realistic.

**`_dispatch_event` amendment:** add routing for the new event type:

```python
elif event.event_type == "courier_returned_to_store":
    self._handle_courier_returned_to_store(event)
```

### 5.5 `_on_order_terminal` amendment (additive branch)

```python
# Before:
if order.status == OrderStatus.DELIVERED:
    self._collector.on_order_delivered(order, sim_time, cost)
else:
    self._collector.on_order_failed(order, sim_time)

# After:
if order.status == OrderStatus.DELIVERED:
    self._collector.on_order_delivered(order, sim_time, cost)
elif order.status == OrderStatus.RETURNED:
    self._collector.on_order_returned(order, sim_time, cost)   # NEW
else:
    self._collector.on_order_failed(order, sim_time)
```

### 5.6 Courier phase summary (complete, including new values)

| Phase value | Meaning |
|-------------|---------|
| `"free"` | Idle, available for dispatch |
| `"en-route-store"` | Traveling to store (leg 1) |
| `"at-store"` | Waiting at store while order is being prepared |
| `"waiting-at-store"` | **NEW** — in store's overflow queue (prep slot full) |
| `"en-route-customer"` | Traveling to customer (leg 2) |
| `"returning"` | **NEW** — traveling back to store with a refused order |

---

## 6. Metrics

All new metrics follow the existing pattern: derived from event notifications, not
from snapshot polling. Dt-independence is automatic.

### 6.1 KPICollector: new notification methods

```python
# Store queue metrics
def on_order_queued_at_store(
    self, order_id: str, store_id: str, sim_time: float
) -> None:
    # Record queue entry time keyed by order_id
    self._store_queue_entry[order_id] = sim_time

def on_order_dequeued_from_store(
    self, order_id: str, store_id: str, wait_time: float
) -> None:
    # Accumulate wait time; track max queue depth
    self._store_wait_times.append(wait_time)
    self._store_queue_entry.pop(order_id, None)

# Return metrics
def on_order_returned(
    self, order: Order, sim_time: float, cost: float
) -> None:
    self._returned_orders += 1
    # Record sim_time for return_leg_time computation in on_courier_returned_to_store
    self._return_dispatch_times[order.order_id] = sim_time

def on_courier_returned_to_store(
    self, courier_id: str, sim_time: float
) -> None:
    # Note: we need order_id to look up dispatch time.
    # Design choice: pass order_id in this notification.
    # Amended signature:
    # on_courier_returned_to_store(courier_id, order_id, sim_time)
    pass
```

**Amended `on_courier_returned_to_store` signature:**

```python
def on_courier_returned_to_store(
    self, courier_id: str, order_id: str, sim_time: float
) -> None:
    dispatch_t = self._return_dispatch_times.pop(order_id, None)
    if dispatch_t is not None:
        self._return_leg_times.append(sim_time - dispatch_t)
```

The simulator passes `order_id` from the event payload to this call.

### 6.2 KPICollector: new state fields

```python
# Additions to __init__:
self._returned_orders: int = 0
self._return_dispatch_times: dict[str, float] = {}   # order_id → RETURNED sim_time
self._return_leg_times: list[float] = []
self._store_wait_times: list[float] = []
self._store_queue_entry: dict[str, float] = {}        # order_id → queued sim_time

# All cleared in reset().
```

### 6.3 KPICollector: new summary fields

```python
# In summary():

# Returns
returned = self._returned_orders
delivery_attempts = self._delivered_orders + returned
return_rate = returned / delivery_attempts if delivery_attempts > 0 else 0.0
mean_return_leg = (
    float(np.mean(self._return_leg_times))
    if self._return_leg_times else 0.0
)

# Store queue
mean_store_wait = (
    float(np.mean(self._store_wait_times))
    if self._store_wait_times else 0.0
)
max_store_queue_depth = (
    len(self._store_queue_entry)   # in-progress queue entries at episode end
    # NOTE: this measures only the final queue depth, not the maximum observed.
    # True max_queue_depth requires tracking max at every enqueue/dequeue.
    # See §6.4 for max tracking.
)
```

### 6.4 Max store queue depth tracking

The maximum instantaneous queue depth across all stores is useful for the "heavy"
preset. Track it in `on_order_queued_at_store`:

```python
def on_order_queued_at_store(self, order_id, store_id, sim_time):
    self._store_queue_entry[order_id] = sim_time
    self._max_store_queue_depth = max(
        self._max_store_queue_depth, len(self._store_queue_entry)
    )
```

This is a single-store global maximum (across all stores combined). If per-store
depth is needed in a future increment, the key can be `(store_id, order_id)`.

### 6.5 Complete new summary keys

| Key | Type | Definition | Existing? |
|-----|------|-----------|-----------|
| `returned_orders` | int | Orders that transitioned to RETURNED | NEW |
| `return_rate` | float | `returned / (delivered + returned)` | NEW |
| `mean_return_leg_time` | float | Mean travel time from refused delivery to store arrival | NEW |
| `mean_store_wait_time` | float | Mean time an order waited for a prep slot | NEW |
| `max_store_queue_depth` | int | Peak number of couriers waiting for prep slots simultaneously | NEW |

All five keys are always present in `summary()` even when zero (empty-episode
convention matching existing keys).

---

## 7. Snapshot Additions

All changes are additive. Frozen dataclasses gain new fields with defaults.
`WorldState.snapshot()` is updated to populate them.

### 7.1 `StoreSnapshot` amendment

```python
@dataclass(frozen=True)
class StoreSnapshot:
    store_id: str
    x: float
    y: float
    coverage_radius: float
    queue_depth: int = 0    # NEW — number of couriers in this store's waiting queue
```

Populated in `WorldState.snapshot()`:

```python
StoreSnapshot(
    store_id=s.store_id, x=s.x, y=s.y,
    coverage_radius=s.coverage_radius,
    queue_depth=s.queue_depth,     # NEW
)
```

### 7.2 `WorldSnapshot` amendment

```python
@dataclass(frozen=True)
class WorldSnapshot:
    tick: int
    elapsed: float
    stores: tuple[StoreSnapshot, ...]
    couriers: tuple[CourierSnapshot, ...]
    orders: tuple[OrderSnapshot, ...]
    scenario_name: str = ""           # NEW — active preset name for HUD
    demand_intensity: float = 0.0     # NEW — normalised current rate (0.0–1.0)
```

`scenario_name` is `self.config.name` from the Simulator (passed to
`WorldState.snapshot()` or stored on `WorldState` itself at reset time).

`demand_intensity` is the current rate as a fraction of the configured peak rate,
computed at snapshot time by the demand generator:

```python
# New method on DemandGenerator ABC:
def current_intensity(self, sim_time: float) -> float:
    """Return current rate / peak_rate in [0.0, 1.0]. Default: 1.0."""
    return 1.0
```

`DailyProfileDemandGenerator` and `BurstDemandGenerator` override this to return
`profile_factor(sim_time) × intensity / peak_intensity`. `PoissonDemandGenerator`
returns 1.0 always (constant rate).

The `WorldState` is given a reference to the active demand generator at reset time
(or the Simulator computes it and passes it to `snapshot()`).

### 7.3 Renderer impact

The pygame renderer (`render/pygame_renderer.py`) reads `WorldSnapshot` fields. New
fields are opt-in: the existing renderer continues to work unchanged (new fields have
defaults). The new visual features described in the requirements are implemented by
the renderer consuming the new fields.

`CourierSnapshot.status` already carries the phase string. The new phase values
(`"waiting-at-store"`, `"returning"`) are surfaced automatically via the existing
`courier_phase` mechanism. The renderer just needs to map these new strings to new
visual states.

---

## 8. Consolidated Contract Amendments

Each amendment is flagged: **ADDITIVE** (new fields/methods, all defaulted, no
existing callers break) or **AMENDMENT** (existing signature changes requiring all
callers to update).

### AM-1: `OrderStatus` enum — ADDITIVE
- Add `RETURNED = auto()`
- Add `IN_TRANSIT → RETURNED` to `ALLOWED_TRANSITIONS`
- Add `RETURNED: frozenset()` to `ALLOWED_TRANSITIONS`
- `Order.is_terminal`: add `RETURNED` to the in-check

### AM-2: `StoreConfig` — ADDITIVE
- Add `prep_time: float = Field(default=30.0, gt=0.0)`

### AM-3: `DemandConfig` — ADDITIVE
- Add `intensity: float = Field(default=1.0, gt=0.0)`
- Add `profile: list[ProfileBreakpoint] = []`
- Add `burst_rate_factor: float = 5.0`
- Add `burst_duration_fraction: float = 0.1`
- Add `burst_interval_fraction: float = 0.3`
- Add `ProfileBreakpoint` model (`time_fraction: float`, `rate_factor: float`)

### AM-4: `ScenarioConfig` — ADDITIVE
- Add `return_rate: float = Field(default=0.0, ge=0.0, le=1.0)`

### AM-5: `Store` ABC — ADDITIVE (new abstract methods)
- `enqueue_waiter(courier_id, order_id, arrived_at) → None`
- `dequeue_next_waiter() → tuple[str, str, float] | None`
- `complete_preparation(order_id) → None`
- `queue_depth` property → `int`

### AM-6: `BuiltinStore` — AMENDMENT (implements AM-5; constructor gains `prep_time`)
- Constructor: add `prep_time: float = 30.0` param
- Add `_waiting: deque` field
- Implement four new methods from AM-5
- Amend `reset()` to clear `_waiting`

### AM-7: `DemandGenerator` ABC — AMENDMENT (reset signature)
- `reset(rng, horizon: float = 0.0) → None` (default keeps callers valid)
- New abstract method: `current_intensity(sim_time: float) → float`

### AM-8: `PoissonDemandGenerator` — AMENDMENT (adapter for AM-7)
- Constructor: accept `intensity: float = 1.0`, `horizon: float = 0.0`, `**_ignored`
- `next_event`: multiply rate by `self.intensity`
- `reset`: accept new `horizon` kwarg (no-op)
- `current_intensity`: return `self.intensity` (constant)

### AM-9: `KPICollector` — ADDITIVE (new notification methods + summary keys)
- `on_order_queued_at_store(order_id, store_id, sim_time)`
- `on_order_dequeued_from_store(order_id, store_id, wait_time)`
- `on_order_returned(order, sim_time, cost)`
- `on_courier_returned_to_store(courier_id, order_id, sim_time)`
- Five new summary keys (§6.5)
- New state fields initialised in `__init__` and cleared in `reset()`

### AM-10: `Simulator` — AMENDMENT (multiple callers; review all usages)
- `reset()`: spawn(2) instead of spawn(1); add `returns_rng`; pass new params to
  store/generator constructors; compute `horizon` earlier
- `_handle_courier_arrived_store`: capacity guard + enqueue branch
- `_handle_order_ready`: `complete_preparation` + dequeue + waiter scheduling
- `_handle_courier_arrived_customer`: return-decision draw + return branch
- New handler: `_handle_courier_returned_to_store`
- `_dispatch_event`: route new event type
- `_on_order_terminal`: add RETURNED branch

### AM-11: `WorldState` / `WorldSnapshot` / `StoreSnapshot` — ADDITIVE
- `StoreSnapshot`: add `queue_depth: int = 0`
- `WorldSnapshot`: add `scenario_name: str = ""`, `demand_intensity: float = 0.0`
- `WorldState.snapshot()`: populate new fields

---

## 9. RNG Streams: Authoritative Spawn-Index Table

```
SeedSequence(config.seed).spawn(2)
  index 0 → gen_rng       → DemandGenerator.next_event (all demand events)
  index 1 → returns_rng   → return decision in _handle_courier_arrived_customer
```

**Rules:**
- Indexes are stable. Never reorder. New consumers append at the tail (index 2+).
- `gen_rng` is passed exclusively to `DemandGenerator.next_event`. No other consumer.
- `returns_rng` is consumed exactly once per delivery attempt when `return_rate > 0`.
- When `return_rate == 0`, `returns_rng` is unconsumed. The stream exists and is
  spawned; it just produces no draws. This is correct and does not alter `gen_rng`.

---

## 10. Reproducibility and No-Inert-Knobs Audit

### 10.1 Reproducibility checklist

| Assertion | Evidence |
|-----------|---------|
| `spawn(2)[0] == spawn(1)[0]` (same entropy) | Smoke test (Step B6) |
| `return_rate=0` → identical KPIs to pre-B run | Regression test against fingerprint |
| Same (config, seed) → identical order list | Existing determinism test; extend to cover new events |
| Burst timing is deterministic (no RNG) | By design in §3.5 |
| Profile interpolation is deterministic | Pure function of `sim_time / horizon` |

### 10.2 No-inert-knobs audit

| Knob | Lives in | Drives | Observable via |
|------|---------|--------|---------------|
| `intensity` | `DemandConfig` | `next_event` inter-arrival delay | `total_orders` |
| `profile` breakpoints | `DemandConfig` | Rate function at `sim_time / horizon` | `created_at` timestamp distribution |
| `burst_rate_factor` | `DemandConfig` | Rate during burst windows | Burst-window order density |
| `burst_duration_fraction` | `DemandConfig` | Length of burst windows | Burst-window order count |
| `burst_interval_fraction` | `DemandConfig` | Spacing between burst windows | Inter-burst quiet period length |
| `prep_time` (per store) | `StoreConfig` | `store.start_preparation` → `order_ready` time | `mean_pickup_latency` |
| `capacity` (per store) | `StoreConfig` | `store.can_prepare` gate → queue formation | `mean_store_wait_time`, `max_store_queue_depth` |
| `return_rate` | `ScenarioConfig` | Return decision draw in courier-arrived-customer | `returned_orders`, `return_rate` |

**Zero-effect cases that must not exist:**
- `profile=[]` with `generator_type=DailyProfileDemandGenerator` → falls back to
  constant (equivalent to PoissonDemandGenerator). This is acceptable: an empty
  profile is a degenerate case, not a separate knob.
- `burst_rate_factor`, `burst_duration_fraction`, `burst_interval_fraction` when
  `generator_type=PoissonDemandGenerator` → ignored. These fields are generator-
  type-scoped; they are not inert knobs but scoped knobs (they only apply when the
  matching generator is active). Document this clearly.

---

## 11. Build Step Sequence

Steps marked **[AMENDMENT]** require plan-then-stop: review and sign off before
implementing, because they touch public contracts that other tests depend on.

Steps marked **[IMPL]** can be implemented directly once the preceding amendment
steps are complete.

---

### Step B0: Regression Fingerprint (prerequisite, ~30 min)

**What:** Run `balanced.yaml` headlessly, capture `kpi_collector.summary()` as a
JSON fixture. This is the "golden" fingerprint all subsequent steps must preserve
(for `return_rate=0`, `intensity=1.0` defaults).

**Files:** `tests/fixtures/balanced_kpi_fingerprint.json`

**DoD:** Fixture committed. A test `test_balanced_fingerprint.py` asserts that a
fresh run of `balanced.yaml` with current code matches the fixture exactly.

---

### Step B1: Config Schema Amendments **[AMENDMENT]**

**What:** Extend the four Pydantic models per §8 (AM-2, AM-3, AM-4).

**Files changed:** `src/delivery_sim/config/schema.py`

**DoD:**
- All existing scenario YAMLs load without error (new fields have defaults).
- `pydantic v2` rejects: negative `prep_time`, `intensity ≤ 0`, `return_rate > 1.0`,
  `burst_duration_fraction + burst_interval_fraction > 1.0`.
- Tests: `tests/config/test_schema_b_realistic.py` covers the above validations.
- B0 fingerprint test still passes.

---

### Step B2: Order State Machine Amendment **[AMENDMENT]**

**What:** Add `RETURNED` to `OrderStatus`, `ALLOWED_TRANSITIONS`, and `is_terminal`
per AM-1.

**Files changed:** `src/delivery_sim/entities/order.py`

**DoD:**
- All existing order state machine tests pass unchanged.
- New tests: `RETURNED` reachable from `IN_TRANSIT`; `RETURNED` is terminal;
  `DELIVERED → RETURNED` raises; `FAILED → RETURNED` raises; `is_terminal` True
  for RETURNED.
- B0 fingerprint test still passes.

---

### Step B3: Store ABC and BuiltinStore Amendment **[AMENDMENT]**

**What:** Add four abstract methods to `Store` ABC and implement in `BuiltinStore`
(AM-5, AM-6). Expose `prep_time` in constructor (AM-6) and in `Simulator.reset()`
(part of AM-10).

**Files changed:**
- `src/delivery_sim/entities/store.py`
- `src/delivery_sim/engine/simulator.py` (prep_time passthrough only)

**DoD:**
- `BuiltinStore(prep_time=60.0)` produces `mean_pickup_latency ≈ 60s` (double the
  default), verified by a small integration test.
- `BuiltinStore.reset()` clears `_waiting`.
- B0 fingerprint test still passes (default `prep_time=30.0`).
- `StoreConfig.prep_time` round-trips through YAML load.

---

### Step B4: Store Queue Event Flow **[AMENDMENT]**

**What:** Wire the `can_prepare` / enqueue / dequeue / `complete_preparation`
logic into the Simulator event handlers per §4.4 (part of AM-10). New courier
phase `"waiting-at-store"`.

**Files changed:** `src/delivery_sim/engine/simulator.py`

**DoD (each verified by targeted integration tests):**
- With `capacity=1` and two simultaneous arriving couriers: one prepares, one
  waits; the waiter starts prep exactly when the first order is picked up.
- `mean_store_wait_time > 0` when capacity is intentionally saturated.
- B0 fingerprint test still passes (balanced.yaml has capacity=20, never saturates).
- No order is lost or double-prepared.
- `order.status` progression: ASSIGNED → (wait) → PREPARING → PICKED_UP →
  IN_TRANSIT, correct timestamps at each.

---

### Step B5: Store Queue Metrics **[IMPL]**

**What:** Add `on_order_queued_at_store` / `on_order_dequeued_from_store` to
`KPICollector`; add new summary keys per §6 (part of AM-9). Wire notifications
from Simulator.

**Files changed:**
- `src/delivery_sim/metrics/collector.py`
- `src/delivery_sim/engine/simulator.py` (add collector.on_* calls)

**DoD:**
- `summary()` always contains `mean_store_wait_time` and `max_store_queue_depth`.
- B0 fingerprint test: both values are `0` (no queue in balanced.yaml).
- Saturated-capacity test from B4 shows `mean_store_wait_time > 0`.
- `reset()` clears new state fields.

---

### Step B6: Probabilistic Returns — RNG Stream and Lifecycle **[AMENDMENT]**

**What:** Add `returns_rng` (spawn index 1) to `Simulator.reset()`. Amend
`_handle_courier_arrived_customer` with return decision and return branch. Add
`_handle_courier_returned_to_store` handler. Amend `_dispatch_event` and
`_on_order_terminal` per §5 (AM-10, AM-9 partial).

**Files changed:** `src/delivery_sim/engine/simulator.py`

**DoD (test-by-test):**
- Smoke test: `spawn(2)[0].entropy == spawn(1)[0].entropy` for a range of seeds.
- `return_rate=0.0`: `returned_orders == 0`; KPI fingerprint matches B0 exactly (R-1).
- `return_rate=1.0`: all orders that reach a customer are RETURNED; `delivered_orders == 0` (R-2).
- `return_rate=0.5`, fixed seed: two independent runs produce identical RETURNED
  order lists (R-7 determinism).
- Returning courier is unavailable (phase `"returning"`) until the return event
  fires (R-6).
- `_on_order_terminal` is called exactly once per returned order (no cost leaks).

---

### Step B7: Return Metrics **[IMPL]**

**What:** Complete the `KPICollector` additions for returns (AM-9 remainder).
`on_order_returned`, `on_courier_returned_to_store`; `returned_orders`,
`return_rate`, `mean_return_leg_time` in `summary()`.

**Files changed:** `src/delivery_sim/metrics/collector.py`

**DoD:**
- `summary()` always contains the three new return keys.
- `return_rate = returned / (delivered + returned)` (R-4, R-5 separation from
  `failed_orders`).
- `mean_return_leg_time` matches the manually-computable expected value for a
  known-geometry test case.
- `reset()` clears all new fields.

---

### Step B8: Time-Varying Demand Generators **[AMENDMENT]** (ABC) + **[IMPL]** (generators)

**What:** Amend `DemandGenerator` ABC (`reset` signature, `current_intensity`
method). Amend `PoissonDemandGenerator` (AM-7, AM-8). Implement
`DailyProfileDemandGenerator` and `BurstDemandGenerator` (§3.4, §3.5). Update
`Simulator.reset()` to pass new params.

**Files changed:**
- `src/delivery_sim/entities/demand_generator.py`
- `src/delivery_sim/engine/simulator.py` (pass intensity/horizon/profile/burst)

**DoD:**
- D-1: `PoissonDemandGenerator` at `intensity=1.0` with same seed → identical
  `total_orders` and `order.created_at` timestamps as pre-B8 (no regression).
- D-4: `intensity=2.0` → `total_orders` ≈ 2× at `intensity=1.0` (mean, across
  10 seeds; Poisson CLT gives tight bound).
- D-2: `DailyProfileDemandGenerator` with two-peak profile shows order density
  in peak windows > off-peak windows (paired t-test or simple ratio check).
- D-3: `BurstDemandGenerator` shows zero arrivals in inter-burst windows and
  high density in burst windows.
- D-5: same (config, seed) → identical order-created timestamps for all three
  generator types.

---

### Step B9: Snapshot Additions **[IMPL]**

**What:** Add `queue_depth` to `StoreSnapshot`, `scenario_name` and
`demand_intensity` to `WorldSnapshot`, update `WorldState.snapshot()` per §7
(AM-11). Add `current_intensity` to demand generator ABC (part of AM-7).

**Files changed:**
- `src/delivery_sim/render/protocol.py`
- `src/delivery_sim/engine/world_state.py`
- `src/delivery_sim/entities/demand_generator.py`

**DoD:**
- `WorldSnapshot` fields have defaults; existing renderer tests pass unchanged.
- `StoreSnapshot.queue_depth` is populated from `store.queue_depth`.
- `WorldSnapshot.scenario_name` matches `config.name`.
- `WorldSnapshot.demand_intensity` is in `[0.0, 1.0]` at every tick for all
  three generator types.
- Observer-invariance regression test still passes (renderer cannot perturb run).

---

### Step B10: Preset YAMLs and Preset Selection **[IMPL]**

**What:** Author `scenarios/presets/light.yaml` and `scenarios/presets/heavy.yaml`.
Update `scenarios/presets/balanced.yaml` (promote from `scenarios/balanced.yaml`,
adding B-realistic fields at their defaults so its KPI fingerprint is unchanged).
Optionally add `--preset NAME` shortcut to CLI scripts.

**Files changed / created:**
- `scenarios/presets/light.yaml` (new)
- `scenarios/presets/balanced.yaml` (new; `scenarios/balanced.yaml` kept as
  backward-compat symlink or copy)
- `scenarios/presets/heavy.yaml` (new)
- `scripts/train.py`, `scripts/evaluate.py` (optional `--preset` flag)

**DoD (preset acceptance criteria):**
- P-1: `--preset heavy` resolves and loads without error.
- P-2: light run (10 seeds) shows strictly lower `courier_utilization` mean and
  `mean_delivery_time` mean than balanced (same seed set).
- P-3: heavy run (10 seeds) shows strictly higher `mean_delivery_time` and
  `max_store_queue_depth` mean than balanced.
- B0 fingerprint: `balanced` preset run (seed=42) matches fixture exactly.

---

### Step B11 (Stretch): Visual Preset Switching **[IMPL]**

**What:** Add "switch preset + restart" interaction to pygame renderer. Episode
restart only; no mid-episode mutation.

**Files changed:** `src/delivery_sim/render/pygame_renderer.py`

**DoD:**
- Pressing a key (e.g., 1/2/3 for light/balanced/heavy) restarts the episode with
  the new preset. KPI counter resets to zero.
- Two presses of the same key with the same seed produce identical runs (restart
  is reproducible).
- A label in the HUD shows the active preset name, updating on switch.

---

## Appendix A: Reproducibility Verification Addenda

Three verifications raised during design review; each is a "looks fine, breaks under
load" class of defect. The relevant build steps carry these as explicit DoD items.

---

### V-1: Store Queue Structure and Tiebreak (carry into Step B4)

**Risk:** Under heavy load, multiple couriers may arrive at the same store at the
same `sim_time`. A non-FIFO or insertion-order-nondeterministic structure would dequeue
them in arbitrary order, breaking `(config, seed) → identical run`.

**Specification:**

`BuiltinStore._waiting` is a `collections.deque`. Each entry is a 4-tuple:

```python
(enqueue_seq: int, courier_id: str, order_id: str, arrived_at: float)
```

`enqueue_seq` is a monotonically increasing integer counter (`_enqueue_seq: int`
initialized to 0, incremented on every `enqueue_waiter` call). `dequeue_next_waiter`
calls `popleft()`, which is FIFO by the deque's own order — the counter is present
but is the *structural guarantee*, not a sort key.

**Why the counter is necessary:** the simulator processes all events via a totally-
ordered priority queue keyed by `(time, priority, insertion_counter)`. Two
`courier_arrived_store` events at the same `sim_time` and same priority are ordered
by their insertion counters, which are determined by dispatch order. Dispatch in
`_dispatch` is already deterministic (couriers sorted by `courier_id`). Therefore
insertion order into `_waiting` is itself deterministic, and `popleft()` on the deque
is FIFO-by-insertion.

The `enqueue_seq` counter makes this self-documenting and self-contained: even if
dispatch order ever changes, the queue entry's counter proves the dequeue order
without requiring knowledge of event-queue internals.

**DoD addition for B4:** test that two same-sim_time `courier_arrived_store` events
processed in known order (A before B) result in A being dequeued first; same run from
the same seed produces the same dequeue sequence.

---

### V-2: Returns RNG Draw Position (carry into Step B6)

**Risk:** If `returns_rng` is consumed at different positions relative to other
operations in `_handle_courier_arrived_customer` across different code paths, the
RNG state diverges between runs that should be identical, breaking reproducibility.

**Specification:**

The draw is pinned as the **first** operation in the return-decision branch, after
only two setup reads (no mutations, no other draws):

```python
def _handle_courier_arrived_customer(self, event: Event) -> None:
    sim_time = self.clock.elapsed          # 1. read clock
    courier_id = event.payload["courier_id"]   # 2. unpack payload
    order_id   = event.payload["order_id"]

    # Return-decision draw — FIXED POSITION, SOLE CONSUMER of returns_rng:
    refused = (
        self._returns_rng.random() < self.config.return_rate
        if self.config.return_rate > 0.0
        else False
    )
    # Fast path (return_rate == 0.0) does NOT consume returns_rng. This is
    # intentional: a scenario that adds return_rate=0.0 explicitly is identical
    # to one that omits the field (default 0.0) — no RNG state divergence.

    order   = self.world.active_orders[order_id]   # 3. resolve entities
    courier = self.world.courier_index[courier_id]
    ...
```

**Contract:** `returns_rng` is consumed at most once per `courier_arrived_customer`
event, always at the fixed position above, and never elsewhere. No other handler or
method may consume `returns_rng`. This mirrors the demand generator's pinned draw
order (delay → cx → cy).

**DoD addition for B6:** test that a scenario with `return_rate=0.5` produces
identical RETURNED order lists across two independent runs with the same seed;
test that `return_rate=0.0` produces zero `returns_rng` draws (the stream state
is identical before and after a full episode at rate=0.0 — verifiable by seeding a
fresh copy and checking downstream draws).

---

### V-3: `reset(horizon=0.0)` Safety (carry into Step B8)

**Risk:** `horizon=0.0` (the backward-compatible default on the amended `reset`
signature) is mathematically unsafe in time-varying generators: `sim_time / horizon`
divides by zero; `burst_interval_fraction × horizon` = 0 collapses all burst windows;
the entire profile schedule degenerates silently. A "backward-compatible default"
that breaks the math on first use is an inert-knob-class trap.

**Two-part specification:**

**Part 1 — Simulator always passes the real horizon (structural guarantee):**

In `Simulator.reset()`, `self._horizon` is computed *before* the demand generator is
constructed and reset:

```python
# In Simulator.reset() — horizon computed first, always:
self._horizon = self.config.max_steps * self.config.dt   # always > 0 (max_steps > 0, dt > 0)
self._demand_generator = create("demand_generator", ..., horizon=self._horizon)
self._demand_generator.reset(self._gen_rng, horizon=self._horizon)
```

`ScenarioConfig` enforces `max_steps > 0` (existing) and `dt > 0` (existing), so
`self._horizon` is always strictly positive when the simulator resets.

**Part 2 — Time-varying generators guard against unsafe call patterns:**

`DailyProfileDemandGenerator.reset()` and `BurstDemandGenerator.reset()` validate
their `horizon` argument:

```python
def reset(self, rng: np.random.Generator, horizon: float = 0.0) -> None:
    if horizon <= 0.0:
        raise ValueError(
            f"{type(self).__name__}.reset() requires horizon > 0; "
            f"got {horizon!r}. Always call via Simulator.reset(), "
            f"or pass the episode duration explicitly."
        )
    self._horizon = horizon
    ...
```

`PoissonDemandGenerator.reset()` remains a no-op with an ignored `horizon` argument;
the default is safe for it because it never divides by `horizon`.

**The default is not removed** from the ABC signature — it preserves backward
compatibility for callers of `PoissonDemandGenerator.reset(rng)`. But the default is
documented as "safe only for stateless generators; time-varying generators will raise."

**DoD addition for B8:** test that calling `DailyProfileDemandGenerator.reset(rng)`
without `horizon` (using the default `0.0`) raises `ValueError` immediately; test
that `BurstDemandGenerator.reset(rng)` likewise raises; test that
`PoissonDemandGenerator.reset(rng)` continues to be a silent no-op.

---

### B10 Preset Test Clarification

The preset acceptance test (Step B10) does not pass if a single metric is ordered
correctly — it passes only if light / balanced / heavy are **jointly ordered** across
a profile of metrics:

```python
def test_preset_ordering(light_kpis, balanced_kpis, heavy_kpis):
    # Delivery rate: light >= balanced >= heavy
    assert light_kpis["delivery_rate"] >= balanced_kpis["delivery_rate"] >= heavy_kpis["delivery_rate"]
    # Mean delivery time: light <= balanced <= heavy
    assert light_kpis["mean_delivery_time"] <= balanced_kpis["mean_delivery_time"] <= heavy_kpis["mean_delivery_time"]
    # Queue depth: light == 0, heavy > balanced >= 0
    assert light_kpis["max_store_queue_depth"] == 0
    assert heavy_kpis["max_store_queue_depth"] > balanced_kpis["max_store_queue_depth"]
    # Return count: light <= balanced <= heavy
    assert light_kpis["returned_orders"] <= balanced_kpis["returned_orders"] <= heavy_kpis["returned_orders"]
```

Run across ≥5 seeds, require all seeds to satisfy all four inequalities. This ensures
the presets are coherently ordered across the full KPI profile, not accidentally
ordered on one metric while inverted on another.

---

## Appendix B: Flag — One Requirement vs. Architecture Tension

**Requirement R-2** states: `return_rate=1.0` must transition every reaching-courier
order to RETURNED with `delivered_orders == 0`. The `ScenarioConfig` field as
specified uses `le=1.0` (inclusive), so `return_rate=1.0` is a legal config value.
The architect flags this as a **testing-only value** — a `return_rate=1.0` scenario
is a degenerate episode where no deliveries ever succeed and the agent cannot
receive positive reward. It should not appear in any preset YAML. The validator
enforces `ge=0.0, le=1.0`; documentation notes that values above `~0.05` are
unrealistic and above `0.5` are pathological.

No architecture change is needed; this flag is for the engineer to be aware of when
writing preset YAML values and test cases.

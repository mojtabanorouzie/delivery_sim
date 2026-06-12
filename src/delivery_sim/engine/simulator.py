"""
Simulator — event-driven simulation engine (ADR-001 / ADR-002).

Layer: Simulation Engine.

The Simulator owns the clock, event queue, and world state.  It drives the
world forward event-by-event: the run loop pops the next event, advances the
clock to that exact timestamp, and dispatches to the appropriate handler.
``dt`` is the observer / snapshot cadence only; it never governs advance size.

Event model
-----------
order_created       → find covering store (Store.covers); dispatch if courier
                      free; schedule next order_created via
                      DemandGenerator.next_event.
courier_arrived_store → store.start_preparation; transition ASSIGNED→PREPARING;
                        schedule order_ready at prep-completion time.
order_ready         → transition PREPARING→PICKED_UP→IN_TRANSIT; assign leg-2
                      (store→customer); schedule courier_arrived_customer.
courier_arrived_customer → transition IN_TRANSIT→DELIVERED; free courier.

RNG spawn scheme
----------------
SeedSequence(config.seed).spawn(1) → child[0] = gen_rng (sole consumer of
DemandGenerator.next_event; not shared with any other consumer).

Horizon scheduling convention
------------------------------
Events are scheduled only when arrival_time < horizon (strict <).  The run
loop terminates when the next event's time >= horizon or the queue is empty.
These two guards are consistent: an event exactly at horizon is never scheduled
and never processed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import delivery_sim.entities  # noqa: F401 — trigger @register decorators
import delivery_sim.routing  # noqa: F401 — trigger @register decorators
from delivery_sim.config.schema import ScenarioConfig
from delivery_sim.engine.clock import SimClock
from delivery_sim.engine.event_queue import Event, EventQueue
from delivery_sim.engine.world_state import WorldState
from delivery_sim.entities.demand_generator import DemandGenerator
from delivery_sim.entities.order import Order, OrderStatus
from delivery_sim.registry import create
from delivery_sim.routing.base import RoutingModel

if TYPE_CHECKING:
    from delivery_sim.entities.courier import Courier
    from delivery_sim.entities.store import Store
    from delivery_sim.metrics.collector import KPICollector
    from delivery_sim.render.protocol import SnapshotConsumer

# Event priority constants — lower integer = higher urgency at equal timestamps.
# Arrival events process before new demand when they coincide.
_PRI_ARRIVAL = 5
_PRI_DEMAND = 10


class Simulator:
    """Event-driven simulation engine.

    ``run()`` calls ``reset()`` first, then drains the event queue until empty
    or sim_time >= horizon.  An optional ``SnapshotConsumer`` receives
    ``WorldSnapshot`` objects at every ``dt`` boundary (observer cadence);
    these snapshots use ``position_at(t)`` — purely read-only — so the
    observer never influences sim state (observer-invariance guarantee).
    """

    def __init__(self, config: ScenarioConfig) -> None:
        self.config = config
        self.clock = SimClock(dt=config.dt)
        self.event_queue = EventQueue()
        self.world: WorldState | None = None
        self._renderer: SnapshotConsumer | None = None
        self._collector: KPICollector | None = None
        self._gen_rng: np.random.Generator | None = None
        self._returns_rng: np.random.Generator | None = None  # child[1] of SeedSequence
        self._demand_generator: DemandGenerator | None = None
        self._routing: RoutingModel | None = None
        self._order_counter: int = 0
        self._horizon: float = 0.0
        # Per-order accumulated delivery cost (both legs + optional return leg).
        # Cleared each reset().  Popped by _on_order_terminal at every terminal event.
        self._order_leg_cost: dict[str, float] = {}

    def attach_renderer(self, renderer: SnapshotConsumer) -> None:
        """Register a snapshot consumer that is called after each tick."""
        self._renderer = renderer

    def attach_collector(self, collector: KPICollector) -> None:
        """Register a KPICollector that receives event notifications.

        The collector is notified at the exact simulation timestamp of each
        order/courier event — never at snapshot/dt boundaries — so KPI values
        are dt-independent.  The collector never mutates world state.
        """
        self._collector = collector

    def _on_order_terminal(self, order: Order, sim_time: float) -> None:
        """Single exit point for cost accounting and collector notification.

        MUST be called immediately after every ``order.transition(DELIVERED)``
        or ``order.transition(FAILED)`` in this class, regardless of which
        handler triggers the transition.

        Pops the accumulated leg cost (0.0 when the order was never assigned,
        e.g. an uncovered FAILED) and dispatches to the collector.  Routing
        every terminal transition through here means a future handler that
        adds a new failure path cannot leak a ``_order_leg_cost`` entry by
        construction — it only needs to call this method.  The tripwire
        assertion at the end of ``run()`` catches any omission.
        """
        cost = self._order_leg_cost.pop(order.order_id, 0.0)
        order.delivery_cost = cost
        if self._collector is not None:
            if order.status == OrderStatus.DELIVERED:
                self._collector.on_order_delivered(order, sim_time, cost)
            elif order.status == OrderStatus.RETURNED:
                self._collector.on_order_returned(order, sim_time, cost)
            else:
                self._collector.on_order_failed(order, sim_time)

    def reset(self) -> None:
        """Initialise (or re-initialise) all entities from config + seed.

        Spawn scheme: ``SeedSequence(config.seed).spawn(1)`` yields one child
        stream (gen_rng) passed exclusively to ``DemandGenerator.next_event``.
        New RNG consumers must append to the spawn list (stable index order).
        """
        self.clock.reset()
        self.event_queue.clear()
        self._order_counter = 0
        self._order_leg_cost.clear()

        # Spawn two child streams.
        # index 0 → gen_rng    (demand generator, sole consumer)
        # index 1 → returns_rng (return decision in _handle_courier_arrived_customer)
        # spawn(2)[0] produces the same entropy as spawn(1)[0] so existing scenarios
        # with return_rate=0.0 preserve their full KPI fingerprint.
        gen_seed, returns_seed = np.random.SeedSequence(self.config.seed).spawn(2)
        self._gen_rng = np.random.default_rng(gen_seed)
        self._returns_rng = np.random.default_rng(returns_seed)

        # Routing model
        self._routing = create("routing", self.config.routing.model_type)

        # Stores
        stores: list[Store] = []
        for sc in self.config.stores:
            s = create(
                "store", "BuiltinStore",
                store_id=sc.name,
                x=sc.x,
                y=sc.y,
                capacity=sc.capacity,
                prep_time=sc.prep_time,
                coverage_radius=sc.coverage_radius,
            )
            s.reset()
            stores.append(s)

        # Couriers — all spawn at the world origin (0, 0)
        couriers: list[Courier] = []
        cid = 0
        for cc in self.config.couriers:
            for _ in range(cc.count):
                c = create(
                    "courier", cc.courier_type,
                    courier_id=f"c{cid:04d}",
                    x=0.0,
                    y=0.0,
                    routing=self._routing,
                    speed=cc.speed,
                    capacity=cc.capacity,
                    cost_per_unit=cc.cost_per_unit,
                )
                c.reset(origin_x=0.0, origin_y=0.0)
                couriers.append(c)
                cid += 1

        # Horizon must be computed before demand generator reset so time-varying
        # generators can validate and store it (V-3 reproducibility requirement).
        self._horizon = self.config.max_steps * self.config.dt

        # Demand generator
        self._demand_generator = create(
            "demand_generator", self.config.demand.generator_type,
            rate=self.config.demand.rate,
            dt=self.config.dt,
            world_width=self.config.world.width,
            world_height=self.config.world.height,
            store_ids=[s.store_id for s in stores],
            intensity=self.config.demand.intensity,
            profile=self.config.demand.profile,
            burst_rate_factor=self.config.demand.burst_rate_factor,
            burst_duration_fraction=self.config.demand.burst_duration_fraction,
            burst_interval_fraction=self.config.demand.burst_interval_fraction,
        )
        self._demand_generator.reset(self._gen_rng, horizon=self._horizon)

        # World state — courier_phase initialised to "free" for every courier
        self.world = WorldState(
            width=self.config.world.width,
            height=self.config.world.height,
            stores=stores,
            couriers=couriers,
            active_orders={},
            courier_phase={c.courier_id: "free" for c in couriers},
        )

        # Seed the demand stream
        result = self._demand_generator.next_event(0.0, self._gen_rng)
        if result is not None:
            arrival, attrs = result
            if arrival < self._horizon:
                self.event_queue.push(Event(
                    time=arrival, priority=_PRI_DEMAND,
                    event_type="order_created", payload=attrs,
                ))

    def step(self) -> None:
        """Single-event advance for RL wrappers.

        # TODO(step-5): implement for RL env integration.
        """
        raise NotImplementedError

    def run_until(self, target_time: float) -> None:
        """Advance the simulation to *target_time*, for use by RL env wrappers.

        Boundary convention — strict less-than, consistent with the horizon
        strict-< rule in ``run()``:

        * Events with ``event.time < target_time`` are processed in this call.
        * An event whose time equals *exactly* ``target_time`` is **not**
          processed here; it stays queued and is picked up by the next
          ``run_until`` window.  This guarantees every event is processed in
          exactly one window, never zero or two.

        After draining all eligible events the clock is pinned to
        ``target_time`` via ``advance_to``, so callers always observe
        ``clock.elapsed == target_time`` after each call, regardless of
        whether any events fell inside the window.

        Equivalence guarantee: N consecutive calls covering
        ``[0, d), [d, 2d), …, [(N-1)d, Nd=horizon)`` process exactly the
        same set of events as ``run()`` to the same horizon — same order,
        same timestamps, same KPI outcomes — because both use the strict-<
        convention and the underlying priority queue is never cleared between
        calls.

        Caller responsibilities:
        - Call ``reset()`` exactly once before the first ``run_until`` in an
          episode.
        - Call ``collector.finalize(num_couriers, horizon)`` once at episode
          end; ``run_until`` never calls it.
        """
        assert self.world is not None, "call reset() before run_until()"
        while not self.event_queue.is_empty():
            next_ev = self.event_queue.peek()
            assert next_ev is not None
            if next_ev.time >= target_time:
                break
            event = self.event_queue.pop()
            self.clock.advance_to(event.time)
            self._dispatch_event(event)
        if self.clock.elapsed < target_time:
            self.clock.advance_to(target_time)

    def run(
        self,
        consumer: SnapshotConsumer | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Run a full episode to the horizon.

        Calls ``reset()`` first.  *max_steps* overrides ``config.max_steps``
        for this call only; it does not mutate the config.

        The optional *consumer* receives ``WorldSnapshot`` at every ``dt``
        boundary (strictly before each event, then swept to horizon after the
        loop).  Snapshots call ``position_at(t)`` — read-only — so the
        observer cannot alter sim state.  A headless run and a consumer run
        with the same (config, seed) produce identical order histories
        (observer-invariance).
        """
        self.reset()
        if max_steps is not None:
            self._horizon = max_steps * self.config.dt

        assert self.world is not None
        horizon = self._horizon
        dt = self.clock.dt
        next_obs_t = dt   # first observer tick
        obs_tick = 0

        while not self.event_queue.is_empty():
            next_ev = self.event_queue.peek()
            assert next_ev is not None

            if next_ev.time >= horizon:
                break

            # Emit observer snapshots at dt boundaries strictly before the event
            if consumer is not None:
                while next_obs_t < next_ev.time and next_obs_t < horizon:
                    consumer.consume(self.world.snapshot(
                        obs_tick, next_obs_t,
                        scenario_name=self.config.name,
                        demand_intensity=self._demand_generator.current_intensity(next_obs_t)
                        if self._demand_generator is not None else 0.0,
                        demand_pattern=self.config.demand.generator_type,
                    ))
                    obs_tick += 1
                    next_obs_t += dt

            event = self.event_queue.pop()
            self.clock.advance_to(event.time)
            self._dispatch_event(event)

        # Sweep remaining observer ticks up to (not including) horizon
        if consumer is not None:
            while next_obs_t < horizon:
                consumer.consume(self.world.snapshot(
                    obs_tick, next_obs_t,
                    scenario_name=self.config.name,
                    demand_intensity=self._demand_generator.current_intensity(next_obs_t)
                    if self._demand_generator is not None else 0.0,
                    demand_pattern=self.config.demand.generator_type,
                ))
                obs_tick += 1
                next_obs_t += dt
            consumer.close()

        if self._collector is not None:
            self._collector.finalize(
                num_couriers=len(self.world.couriers),
                horizon=horizon,
            )

        # Tripwire — mirrors the SETTLED / advance_to invariant pattern.
        # In-flight orders (non-terminal at horizon) legitimately keep entries
        # in _order_leg_cost; they are dropped by reset().  Terminal orders
        # must have been cleaned up via _on_order_terminal().  Any entry here
        # for a terminal order means a handler transitioned the order without
        # calling _on_order_terminal — catch it immediately rather than letting
        # the accounting silently rot across episodes.
        _leaked = {
            oid for oid in self._order_leg_cost
            if oid in self.world.active_orders
            and self.world.active_orders[oid].is_terminal
        }
        assert not _leaked, (
            "_order_leg_cost has entries for terminal orders at end of run() — "
            "every terminal transition must call self._on_order_terminal(): "
            f"{sorted(_leaked)!r}"
        )

    # ------------------------------------------------------------------
    # Internal event router
    # ------------------------------------------------------------------

    def _dispatch_event(self, event: Event) -> None:
        if event.event_type == "order_created":
            self._handle_order_created(event)
        elif event.event_type == "courier_arrived_store":
            self._handle_courier_arrived_store(event)
        elif event.event_type == "order_ready":
            self._handle_order_ready(event)
        elif event.event_type == "courier_arrived_customer":
            self._handle_courier_arrived_customer(event)
        elif event.event_type == "courier_returned_to_store":
            self._handle_courier_returned_to_store(event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_order_created(self, event: Event) -> None:
        assert self.world is not None
        assert self._demand_generator is not None
        assert self._gen_rng is not None
        assert self._routing is not None

        sim_time = self.clock.elapsed
        cx: float = event.payload["customer_x"]
        cy: float = event.payload["customer_y"]

        # Covering-store resolution: first store (by store_id) whose covers()
        # returns True.  No random fallback — uncovered customer → FAILED.
        covering: Store | None = None
        for s in sorted(self.world.stores, key=lambda s: s.store_id):
            if s.covers(cx, cy, self._routing):
                covering = s
                break

        order_id = f"order_{self._order_counter:06d}"
        self._order_counter += 1
        order = Order(
            order_id=order_id,
            store_id=covering.store_id if covering else "",
            customer_x=cx,
            customer_y=cy,
            created_at=sim_time,
        )
        self.world.active_orders[order_id] = order

        if self._collector is not None:
            self._collector.on_order_created(order, sim_time)

        if covering is None:
            order.transition(OrderStatus.FAILED, sim_time)
            self._on_order_terminal(order, sim_time)
        else:
            self._dispatch(order, sim_time, covering)

        # Schedule next demand event — strict < horizon guard
        result = self._demand_generator.next_event(sim_time, self._gen_rng)
        if result is not None:
            arrival, attrs = result
            if arrival < self._horizon:
                self.event_queue.push(Event(
                    time=arrival, priority=_PRI_DEMAND,
                    event_type="order_created", payload=attrs,
                ))

    def _dispatch(self, order: Order, sim_time: float, store: Store) -> None:
        """Assign the first available courier to *order* for leg-1 (→ store).

        Availability: ``courier_phase == "free"`` implies SETTLED by construction.
        There are exactly two code paths that set a courier's phase to "free", and
        both guarantee the SETTLED precondition (arrival_time() is None or
        sim_time >= arrival_time()) at the moment assign() is called here:

        1. reset() — c.reset() clears _arrival_time_val to None, so
           arrival_time() is None → SETTLED trivially.

        2. _handle_courier_arrived_customer at event time T — sets phase "free"
           without clearing _arrival_time_val, which remains T.  _dispatch is
           only ever called from _handle_order_created, which processes demand
           events (priority _PRI_DEMAND=10).  Arrival events (_PRI_ARRIVAL=5)
           have a lower integer and therefore higher urgency, so any
           courier_arrived_customer at time T is dequeued and processed before
           any order_created at time T.  At this call site sim_time >= T ==
           arrival_time() → SETTLED.

        Tie-break: ascending ``courier_id`` string; never dict/set order.

        Placeholder; order→courier assignment is a future control decision,
        overridable at the env layer (step 5).
        """
        assert self.world is not None

        available: Courier | None = None
        for c in sorted(self.world.couriers, key=lambda c: c.courier_id):
            if self.world.courier_phase.get(c.courier_id) == "free":
                available = c
                break

        if available is None:
            return  # no courier available; order stays CREATED until step-5

        # Defensive assertion: the proof above guarantees this never fires, but
        # catches any future code that sets courier_phase="free" prematurely.
        _settled_eta = available.arrival_time()
        assert _settled_eta is None or sim_time >= _settled_eta, (
            f"SETTLED precondition violated: courier {available.courier_id!r} "
            f"has arrival_time={_settled_eta!r} but sim_time={sim_time!r}; "
            f"courier_phase was 'free' before its trajectory expired"
        )

        order.transition(OrderStatus.ASSIGNED, sim_time)
        order.assigned_courier_id = available.courier_id

        from_x, from_y = available.position_at(sim_time)
        available.assign(
            order_id=order.order_id,
            store_id=store.store_id,
            sim_time=sim_time,
            target_x=store.x,
            target_y=store.y,
            from_x=from_x,
            from_y=from_y,
        )
        self.world.courier_phase[available.courier_id] = "en-route-store"

        assert self._routing is not None
        leg1_dist = self._routing.distance(from_x, from_y, store.x, store.y)
        self._order_leg_cost[order.order_id] = available.cost(leg1_dist)
        if self._collector is not None:
            self._collector.on_courier_busy(available.courier_id, sim_time)

        eta = available.arrival_time()
        assert eta is not None
        self.event_queue.push(Event(
            time=eta, priority=_PRI_ARRIVAL,
            event_type="courier_arrived_store",
            payload={
                "courier_id": available.courier_id,
                "order_id": order.order_id,
                "store_id": store.store_id,
            },
        ))

    def _handle_courier_arrived_store(self, event: Event) -> None:
        assert self.world is not None

        sim_time = self.clock.elapsed
        courier_id: str = event.payload["courier_id"]
        order_id: str = event.payload["order_id"]
        store_id: str = event.payload["store_id"]

        order = self.world.active_orders[order_id]
        store = self.world.store_index[store_id]

        if store.can_prepare(order_id):
            ready_time = store.start_preparation(order_id, sim_time)
            order.transition(OrderStatus.PREPARING, sim_time)
            self.world.courier_phase[courier_id] = "at-store"
            self.event_queue.push(Event(
                time=ready_time, priority=_PRI_ARRIVAL,
                event_type="order_ready",
                payload={
                    "courier_id": courier_id,
                    "order_id": order_id,
                    "store_id": store_id,
                },
            ))
            if self._collector is not None:
                self._collector.on_order_queued_at_store(order_id, store_id, sim_time, queued=False)
        else:
            # All prep slots full — enqueue this courier; order stays ASSIGNED
            store.enqueue_waiter(courier_id, order_id, arrived_at=sim_time)
            self.world.courier_phase[courier_id] = "waiting-at-store"
            if self._collector is not None:
                self._collector.on_order_queued_at_store(order_id, store_id, sim_time, queued=True)

    def _handle_order_ready(self, event: Event) -> None:
        """Complete pickup and launch leg-2 (store → customer).

        SETTLED proof: order_ready fires at ready_time = arrival_at_store +
        prep_time >= courier.arrival_time(), so the SETTLED precondition for
        leg-2 assign() always holds at this handler.

        Queue drain: freeing a prep slot may unblock a waiting courier.
        complete_preparation() is called first so can_prepare() is True when
        start_preparation() is called for the next waiter.
        """
        assert self.world is not None

        sim_time = self.clock.elapsed
        courier_id: str = event.payload["courier_id"]
        order_id: str = event.payload["order_id"]
        store_id: str = event.payload["store_id"]

        order = self.world.active_orders[order_id]
        store = self.world.store_index[store_id]
        courier = self.world.courier_index[courier_id]

        # Free the prep slot before transitioning the current order
        store.complete_preparation(order_id)

        # Drain one waiting courier if any
        waiter = store.dequeue_next_waiter()
        if waiter is not None:
            w_courier_id, w_order_id, w_arrived_at = waiter
            w_order = self.world.active_orders[w_order_id]
            wait_time = sim_time - w_arrived_at
            w_ready_time = store.start_preparation(w_order_id, sim_time)
            w_order.transition(OrderStatus.PREPARING, sim_time)
            self.world.courier_phase[w_courier_id] = "at-store"
            if self._collector is not None:
                self._collector.on_order_dequeued_from_store(
                    w_order_id, store_id, wait_time
                )
            self.event_queue.push(Event(
                time=w_ready_time, priority=_PRI_ARRIVAL,
                event_type="order_ready",
                payload={
                    "courier_id": w_courier_id,
                    "order_id": w_order_id,
                    "store_id": store_id,
                },
            ))

        order.transition(OrderStatus.PICKED_UP, sim_time)
        order.transition(OrderStatus.IN_TRANSIT, sim_time)

        courier.assign(
            order_id=order_id,
            store_id=store_id,
            sim_time=sim_time,
            target_x=order.customer_x,
            target_y=order.customer_y,
            from_x=store.x,
            from_y=store.y,
        )
        self.world.courier_phase[courier_id] = "en-route-customer"

        assert self._routing is not None
        leg2_dist = self._routing.distance(
            store.x, store.y, order.customer_x, order.customer_y
        )
        self._order_leg_cost[order_id] = (
            self._order_leg_cost.get(order_id, 0.0) + courier.cost(leg2_dist)
        )

        eta = courier.arrival_time()
        assert eta is not None
        self.event_queue.push(Event(
            time=eta, priority=_PRI_ARRIVAL,
            event_type="courier_arrived_customer",
            payload={"courier_id": courier_id, "order_id": order_id},
        ))

    def _handle_courier_arrived_customer(self, event: Event) -> None:
        assert self.world is not None
        assert self._returns_rng is not None

        sim_time = self.clock.elapsed
        courier_id: str = event.payload["courier_id"]
        order_id: str = event.payload["order_id"]

        # Return-decision draw — FIXED POSITION, SOLE CONSUMER of _returns_rng.
        # One draw per event, always here, before any entity lookup or mutation.
        # Fast path (rate == 0.0) does not consume returns_rng at all.
        refused = (
            self._returns_rng.random() < self.config.return_rate
            if self.config.return_rate > 0.0
            else False
        )

        order = self.world.active_orders[order_id]
        courier = self.world.courier_index[courier_id]

        if not refused:
            order.transition(OrderStatus.DELIVERED, sim_time)
            self.world.courier_phase[courier_id] = "free"
            self._on_order_terminal(order, sim_time)
            if self._collector is not None:
                self._collector.on_courier_free(courier_id, sim_time)
        else:
            assert self._routing is not None
            return_store = self.world.store_index[order.store_id]

            # Accumulate return-leg cost BEFORE _on_order_terminal pops the entry
            return_dist = self._routing.distance(
                order.customer_x, order.customer_y,
                return_store.x, return_store.y,
            )
            self._order_leg_cost[order_id] = (
                self._order_leg_cost.get(order_id, 0.0) + courier.cost(return_dist)
            )

            order.transition(OrderStatus.RETURNED, sim_time)
            self._on_order_terminal(order, sim_time)

            # Assign return leg: courier travels from customer back to store
            courier.assign(
                order_id=order_id,
                store_id=order.store_id,
                sim_time=sim_time,
                target_x=return_store.x,
                target_y=return_store.y,
                from_x=order.customer_x,
                from_y=order.customer_y,
            )
            self.world.courier_phase[courier_id] = "returning"

            eta = courier.arrival_time()
            assert eta is not None
            self.event_queue.push(Event(
                time=eta, priority=_PRI_ARRIVAL,
                event_type="courier_returned_to_store",
                payload={
                    "courier_id": courier_id,
                    "order_id": order_id,
                    "store_id": order.store_id,
                },
            ))

    def _handle_courier_returned_to_store(self, event: Event) -> None:
        """Courier completed the return leg; free them for new assignments."""
        assert self.world is not None

        sim_time = self.clock.elapsed
        courier_id: str = event.payload["courier_id"]
        order_id: str = event.payload["order_id"]

        self.world.courier_phase[courier_id] = "free"
        if self._collector is not None:
            self._collector.on_courier_free(courier_id, sim_time)
            self._collector.on_courier_returned_to_store(courier_id, order_id, sim_time)

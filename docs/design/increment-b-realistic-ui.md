# Increment B-realistic: UI / Visualization Design Spec

**Status:** Draft  
**Date:** 2026-06-12  
**Implements requirements:** §2.5 (UI Visibility) of `docs/requirements/increment-b-realistic.md`  
**Consumes architecture:** `docs/design/increment-b-realistic-architecture.md` (snapshot fields)  
**Target:** `src/delivery_sim/render/pygame_renderer.py` — pure snapshot observer  
**Designer rule:** Every visual element must be backed by a named snapshot field.
Any needed field not yet in the spec is flagged in §6 as a requested amendment.

---

## 0. Design Principles

**Legibility over decoration.** This is a research and demo tool. The viewer's
question at every moment is: "what is the system doing right now?" — not "is this
beautiful?" Every new visual element earns its place by answering a research question.

**Colorblind-safe.** The existing palette has a green/red pair that is a deuteranopia
failure point. This spec corrects it and extends the palette using colors and shapes
that survive the three most common deficiencies (deuteranopia, protanopia,
tritanopia). Shape differentiation is used for the three terminal order states
(delivered / failed / returned) so color alone is never the only signal.

**Observer purity.** The renderer reads only `WorldSnapshot` fields. The scenario
switch affordance is a key-capture gesture that surfaces a callback to the
application layer; the renderer draws the affordance and the callback label, but
initiates nothing inside the engine.

**Backward compat.** The existing window size (900×700) is extended minimally. All
new visual elements use defaults when new snapshot fields are absent (zero queue,
empty scenario name, intensity 0.0), so the renderer degrades gracefully if old
snapshots are replayed.

---

## 1. Window and Layout

### 1.1 Window Dimensions

```
window_w = 900 px   (unchanged)
window_h = 760 px   (was 700; +60 px for extended HUD)
vp_h     = 580 px   (was 588; slight reduction to give HUD headroom)
hud_h    = 180 px   (was 112; +68 px for three new HUD rows + scenario block)
```

The viewport height reduction (8 px) is imperceptible in practice. The HUD is the
main change.

### 1.2 Overall Layout

```
+═══════════════════════════════════════════════════════════+ y=0
║                                                           ║
║   VIEWPORT  (900 × 580 px)                                ║
║                                                           ║
║   ┌──────────────────────────────┐  ← scenario overlay   ║
║   │  balanced                   │    top-left, x=10 y=10  ║
║   │  ▶ ████████░░░  72%          │    semi-transparent    ║
║   └──────────────────────────────┘    ~200 × 48 px        ║
║                                                           ║
║   world entities:                                         ║
║     coverage circles (translucent)                        ║
║     order dots (customer locations)                       ║
║     courier circles                                       ║
║     store squares + queue badges                          ║
║                                                           ║
+═══════════════════════════════════════════════════════════+ y=580
║  ─── HUD SEPARATOR LINE ───────────────────────────────── ║
║                                                           ║
║   COL A  (x=0–279)    COL B  (x=292–559)  COL C (x=560–) ║
║                                                           ║
║   Episode metrics      Scenario / Demand   Legend         ║
║                                                           ║
+═══════════════════════════════════════════════════════════+ y=760
```

---

## 2. Entity Icon and Color Specification

### 2.1 Color Palette

Full palette including replacements and new entries. Constants for the engineer
(replace/extend the existing `_COURIER` and `_ORDER` dicts):

```python
# --- Background / chrome (unchanged) ---
_BG        = (20, 20, 30)
_HUD_BG    = (30, 30, 45)
_HUD_LINE  = (70, 70, 95)
_TEXT      = (210, 210, 210)
_TEXT_DIM  = (130, 130, 155)
_DIVIDER   = (70, 70, 95)

# --- Store (unchanged) ---
_STORE_COLOR    = (255, 220, 50)   # amber-yellow
_COVERAGE_FILL  = (255, 220, 50, 22)
_COVERAGE_BORDER= (255, 220, 50, 140)
_QUEUE_BADGE    = (210, 40, 40)    # NEW: red badge on busy stores

# --- Couriers ---
# CHANGE: "free" replaced from (70,200,70) → teal to decouple green from DELIVERED
_COURIER = {
    "free":              (30, 190, 160),   # teal
    "en-route-store":    (60, 120, 240),   # blue          (unchanged)
    "at-store":          (160, 70, 220),   # violet        (unchanged)
    "waiting-at-store":  (200, 140, 0),    # amber         (NEW)
    "en-route-customer": (255, 130, 40),   # orange        (unchanged)
    "returning":         (210, 75, 55),    # coral-red     (NEW)
}

# --- Orders ---
# CHANGE: DELIVERED → teal (matches free-courier teal for semantic match)
# CHANGE: DELIVERED shape → hollow ring (see §2.3)
# CHANGE: FAILED shape → cross (see §2.3)
# NEW: RETURNED color + shape
_ORDER = {
    "CREATED":   (160, 160, 165),   # grey          (approx unchanged)
    "ASSIGNED":  (60, 120, 240),    # blue          (unchanged)
    "PREPARING": (160, 70, 220),    # violet        (unchanged)
    "PICKED_UP": (255, 130, 40),    # orange        (unchanged)
    "IN_TRANSIT":(255, 130, 40),    # orange        (unchanged)
    "DELIVERED": (30, 190, 160),    # teal          (CHANGED from green)
    "FAILED":    (210, 45, 45),     # red           (unchanged)
    "RETURNED":  (210, 75, 55),     # coral-red     (NEW, matches "returning" courier)
}

# Intensity bar colors (see §4)
_INTENSITY_LOW    = (60, 120, 240)   # blue
_INTENSITY_MED    = (200, 140, 0)    # amber
_INTENSITY_HIGH   = (210, 75, 55)    # coral
```

### 2.2 Stores

**Shape:** filled square, 14 × 14 px (half = 7 px), centered at screen position.
White 1 px outline. Drawn topmost so always visible over coverage circles.

**Label:** store_id drawn `font_sm` above the square, y − 7 − 10, centered
horizontally on the store. Color `_TEXT_DIM`. (Only draw if `store_id` length ≤ 12
to avoid collision; truncate with "…" otherwise.)

**Queue badge (NEW):**
Driven by `StoreSnapshot.queue_depth`.
- When `queue_depth == 0`: no badge drawn.
- When `queue_depth >= 1`: draw a filled circle (radius 7 px) at position
  (sx + 7, sy − 7) — top-right corner of the square. Color `_QUEUE_BADGE`.
  Draw the digit (or "9+") in white `font_sm`, centered inside the badge.
  This is the pygame notification-badge pattern.

**Queue stress tint (NEW):**
The store square fill color shifts based on `queue_depth` to give a second signal
visible from across the viewport:
- `queue_depth == 0`: `_STORE_COLOR` (amber-yellow, normal)
- `queue_depth == 1–2`: `(255, 175, 30)` (deeper amber, mild stress)
- `queue_depth == 3–4`: `(255, 130, 30)` (orange, moderate stress)
- `queue_depth >= 5`:   `(220, 70, 30)` (red-orange, heavy stress)

The tint communicates magnitude at a glance; the badge communicates the exact count.
Both derive from `store.queue_depth`.

**⚑ FLAG to Architect — Amendment request SA-1:**
`StoreSnapshot.active_prep_count: int` — the number of orders currently occupying
prep slots (being actively prepared, NOT waiting in the queue). Without this, a store
at 95% slot occupancy with zero overflow queue looks identical to an empty store.
The designer requests this field to drive a secondary load indicator (a thin fill bar
beneath the store square, §2.5). If this field cannot be added in B-realistic, the
queue badge alone is the fallback.

### 2.3 Couriers

**Shape:** filled circle, radius 7 px. 1 px white outline. (Unchanged from current.)

**Phase color:** from `_COURIER[courier.status]` (extended table in §2.1).

**Authoritative courier phase strings** — six values emitted by the engine.
The two marked NEW are B-realistic additions (architecture doc §5.6):

| `courier.status` string | Meaning | New in B? |
|-------------------------|---------|-----------|
| `"free"` | Idle, available for dispatch | — |
| `"en-route-store"` | Traveling to store (leg 1) | — |
| `"at-store"` | At store; order being prepared | — |
| `"waiting-at-store"` | At store; waiting for prep slot (queue overflow) | **NEW** |
| `"en-route-customer"` | Traveling to customer (leg 2) | — |
| `"returning"` | Traveling back to store with refused order | **NEW** |

Both `"waiting-at-store"` and `"returning"` are already defined in the architecture
design (§4.4 and §5.4 respectively) and emitted by the Simulator. No snapshot
amendment is required for the renderer to consume them — the renderer maps them
through `_COURIER[courier.status]` exactly as it does today.

**"Returning" secondary mark (NEW):**
For `courier.status == "returning"` only: draw a small left-pointing chevron (‹)
centered inside the courier circle. The chevron is 2 white line segments, total span
4 × 3 px, drawn at the circle center. This is the only secondary mark; other phases
rely on color alone.

Implementation: two `pg.draw.line` calls from (cx+2, cy) → (cx−1, cy−2) and
(cx+2, cy) → (cx−1, cy+2), 1 px wide, color white. Readable at the 7 px radius.

**"Waiting-at-store" indicator (NEW):**
For `courier.status == "waiting-at-store"`: amber fill (from palette) is the primary
signal. Additionally draw a small hollow square (4 × 4 px, 1 px white border,
centered at circle center) to add a shape cue suggesting "blocked / waiting."

Implementation: `pg.draw.rect(screen, white, (cx-2, cy-2, 4, 4), 1)`.

### 2.4 Orders

**Active states** (non-terminal): filled circle, radius 5 px, color from `_ORDER`.

**Terminal states** — three completely distinct shapes (colorblind safety: shape is
never the sole differentiator, but each shape is unambiguous on its own):

| Status | Shape | Color | Draw call |
|--------|-------|-------|-----------|
| DELIVERED | Hollow ring, r=4 px, border 1 px | Teal `(30,190,160)` | `draw.circle(… r=4, width=1)` |
| FAILED | Cross / ✕ — two lines, span 6 px × 6 px | Red `(210,45,45)` | `draw.line` × 2 (diagonal, ±3 px from center) |
| RETURNED | Hollow diamond ◇ — 4-point polygon, half-size 4 px | Coral `(210,75,55)` | `draw.polygon(… points, width=1)` |

**Why a diamond for RETURNED (not a second ring):** DELIVERED and RETURNED both
represent orders that reached a customer. Using the same hollow-ring shape but
different colors risks confusion under any color deficiency (particularly tritanopia,
where teal and coral can both appear brownish). A diamond is an unambiguous shape
signal: "ring = successfully delivered / closed" vs. "diamond = refused at door."
FAILED uses a cross because the order never completed a delivery leg at all.

**Diamond draw call (4 vertices, half-size r=4 px):**
```python
points = [(ox, oy - 4), (ox + 4, oy), (ox, oy + 4), (ox - 4, oy)]
pg.draw.polygon(screen, color, points, 1)   # width=1 → hollow
```

**Order-to-courier link for RETURNED (NEW):**
When `order.status == "RETURNED"` and `order.assigned_courier_id` matches a courier
with `status == "returning"`, draw a faint dashed line from the returning courier's
current screen position to the order's customer position. This line visually connects
the "refused delivery" location to the courier carrying the return. Both
`OrderSnapshot.assigned_courier_id` and `CourierSnapshot.x/y` are already in the
snapshot; no architecture amendment is required for this feature.

Draw: 4 px line segments, 4 px gaps, color coral with alpha ~80 (drawn on the alpha
surface). The engineer can approximate dashes with a loop using `draw.line` for each
segment chunk.

This uses: `OrderSnapshot.customer_x/y`, `OrderSnapshot.assigned_courier_id`,
`CourierSnapshot.x/y` (matched by courier_id). All fields are present in the
current snapshot spec.

### 2.5 Store Utilization Bar (requires SA-1)

If `StoreSnapshot.active_prep_count` is added (Amendment SA-1), draw a thin
utilization bar beneath each store square:
- Position: (sx − STORE_HALF, sy + STORE_HALF + 2), width 14 px, height 3 px
- Background: dark grey (50, 50, 65)
- Fill width: `int(14 * active_prep_count / capacity)` px (requires `capacity` too,
  Amendment SA-2 below)
- Fill color: same three-state tint used for the store square body

**⚑ FLAG to Architect — Amendment request SA-2:**
`StoreSnapshot.capacity: int` — total number of prep slots. Required alongside
`active_prep_count` to normalize the utilization bar. Without this, a raw
active_prep_count number could be displayed as text but cannot be shown as a
proportional bar.

---

## 3. Returns Visualization

### 3.1 Returning courier

Visual state: coral circle + white ‹ chevron inside (see §2.3).
In the legend: labeled "returning" with coral swatch.

### 3.2 Returned order marker

Visual state: hollow coral diamond ◇ at the customer's location (see §2.4).
In the legend: labeled "RETURNED" with coral hollow-diamond swatch.

### 3.3 Return path line

When an order is RETURNED and its courier is still "returning" (en route to store),
draw the faint dashed coral line described in §2.4. This makes the return mechanic
spatially legible: the viewer sees a courier heading back toward a store with a
dashed line connecting it to the original delivery attempt location.

When the courier arrives at the store and transitions to "free", the dashed line
disappears (order's courier_id will no longer match a "returning" courier).

### 3.4 Return rate in HUD (computable from snapshot)

```python
n_returned  = sum(1 for o in snapshot.orders if o.status == "RETURNED")
n_delivered = sum(1 for o in snapshot.orders if o.status == "DELIVERED")
attempts    = n_delivered + n_returned
live_return_rate = n_returned / attempts if attempts > 0 else 0.0
```

Displayed in HUD Column A as: `returned: X    rate: X.X%`

This is computable directly from `snapshot.orders`. No new snapshot field required.

---

## 4. Scenario / Demand State

### 4.1 Viewport overlay (top-left)

A semi-transparent box in the top-left corner of the viewport gives the scenario
name and demand intensity at a glance, making screenshots and recordings self-
documenting without requiring the viewer to look at the HUD.

**Position:** x=10, y=10
**Dimensions:** 198 × 50 px
**Background:** `(20, 20, 30, 180)` (semi-transparent dark, drawn on alpha surface)
**Border:** 1 px, color `(70, 70, 95)` (matches HUD line color)

**Content:**
```
Row 1 (y+8):  scenario name  [font_lg, _TEXT]
              e.g. "balanced"
Row 2 (y+28): "▶ " + intensity bar (80 × 8 px) + " NN%" [font_sm, _TEXT_DIM]
```

**Intensity bar:**
- Background rect: (ox+18, oy+28, 80, 8), color (50, 55, 75)
- Filled rect: (ox+18, oy+28, int(80 * demand_intensity), 8)
  - Fill color: `_INTENSITY_LOW` if < 0.34, `_INTENSITY_MED` if < 0.67, else `_INTENSITY_HIGH`
- Percentage text: drawn at (ox+102, oy+26), format `f"{demand_intensity*100:.0f}%"`

When `snapshot.scenario_name == ""` or `snapshot.demand_intensity == 0.0` (default
values from snapshot spec), the overlay is suppressed or shows placeholder dashes.

**⚑ FLAG to Architect — Amendment request SA-3:**
`WorldSnapshot.demand_pattern: str` — the active generator type as a short label
("constant", "daily-profile", "burst"). Currently `scenario_name` is available
(e.g., "balanced") but not the pattern type. The designer needs this to display:
```
Row 3 (y+40): "pattern: daily-profile"  [font_sm, _TEXT_DIM]
```
Without SA-3, Row 3 is omitted. The overlay renders in 2-row mode (50 px → 38 px tall).

### 4.2 Scenario block in HUD (Column B)

HUD Column B (x=292–551, width=259):

```
y+580+6   "SCENARIO"         font_sm, _TEXT_DIM  ← section label
y+580+20  scenario_name      font_lg, _TEXT      ← e.g. "balanced" (prominent)
y+580+40  "intensity  "      font_sm, _TEXT_DIM
           [intensity bar 80×8 px]  "NN%"
y+580+58  "pattern: …"       font_sm, _TEXT_DIM  ← SA-3 flagged; shows "—" if absent
y+580+76  ─────── divider line ──────────────────
y+580+84  "1=light  2=balanced  3=heavy"
                              font_sm, _TEXT
y+580+100 "↵ restarts episode"
                              font_sm, _TEXT_DIM
y+580+118 (spare)
```

The vertical divider between Col A and Col B is a 1 px line at x=280,
y=580 to y=760, color `_DIVIDER`.
The vertical divider between Col B and Col C is a 1 px line at x=560.

### 4.3 Scenario switch affordance

**Key bindings (for viewport key events):**
- `1` → select "light" preset
- `2` → select "balanced" preset
- `3` → select "heavy" preset

**Renderer behavior:** captures key events in the existing `pg.event.get()` loop
inside `consume()`. When a preset key is detected, the renderer calls
`self._on_preset_switch(name)` if the callback has been registered.

**App-level wiring (not designed here):** the application constructs the renderer
with an `on_preset_switch` callback:
```python
renderer = PygameRenderer(
    ...,
    on_preset_switch=lambda name: restart_episode(name)
)
```

**Visual confirmation:** when a switch key is pressed, the renderer displays a
brief confirmation banner in the viewport center: a dark semi-transparent rect
(400 × 36 px, centered) with text "Switching to {name}… (restart)" for 1.0 s
(tracked by storing `_switch_msg` and `_switch_msg_until` as renderer state,
elapsed from `snapshot.elapsed`).

**Determinism note (from Architect):** live switching is **always** an episode
restart with the new preset config. It is not a mid-episode config mutation.
The banner text "restart" makes this visible to the user. This is confirmed
demo/eval behavior only; the engine's reproducibility guarantees hold for any
complete episode.

---

## 5. Extended HUD Specification

### 5.1 HUD Dimensions

```python
_HUD_H     = 180   # was 112
_TOTAL_H   = 760   # was 700
_VP_H      = 580   # was 588
```

### 5.2 Column A — Episode Metrics (x=12, width=256)

Vertical rhythm: rows start at y=`_VP_H + offset`, advancing by 18 px.

```
+8   t = 12345.0 s    tick = 3800          font_lg, _TEXT
+26  delivered: 82   failed:  3            font_sm, _TEXT
+42  pending:  21    (in-flight)           font_sm, _TEXT_DIM
+58  RETURNED:  4    rate:  4.7%           font_sm, _TEXT  [← new]
+74  queue:  2  couriers waiting           font_sm, _TEXT  [← new, see §5.4]
+90  mean coverage_radius: 682.5           font_sm, _TEXT
+110 ─────────────────────────────────     divider, _DIVIDER
```

The "queue: N couriers waiting" line shows `sum(s.queue_depth for s in snapshot.stores)`.
It is zero and shown dimly when no stores are queued:
- Zero: draw in `_TEXT_DIM`
- Non-zero: draw in `_TEXT` (same brightness as other lines)

### 5.3 Column B — Scenario (see §4.2)

### 5.4 Column C — Legend (x=564, width=332)

Two-column layout within the legend area. Swatch + label, 13 px row height.

```
y+6    LEGEND                 font_sm, _TEXT_DIM (header)

       Sub-col left (x=568)           Sub-col right (x=732)

y+20   ■ store                        ● free courier
y+33   ● courier →store               ● courier at-store
y+46   ● waiting at store    (NEW)    ● courier →customer
y+59   ● returning           (NEW)    ○ delivered (ring)
y+72   ✕ failed (cross)               ◇ RETURNED  (diamond, NEW)
y+85   ● order (active)               (spare)
```

Swatches:
- Store: 8 × 8 px filled rect, `_STORE_COLOR`
- Couriers: circle radius 5, respective color from `_COURIER`
- Delivered: hollow circle radius 4, 1 px border, teal
- Failed: two 4 px diagonal lines (cross), red
- RETURNED: hollow diamond ◇ half-size 4 px, 1 px border, coral
- Order (active): circle radius 5, grey

Labels drawn in `_TEXT_DIM` using `font_sm`. Swatch at (lx, hy + row_y + 6); label
at (lx + 12, hy + row_y).

### 5.5 Computed HUD values (all from snapshot)

| HUD value | Snapshot source | Formula |
|-----------|----------------|---------|
| `t = …` | `snapshot.elapsed` | format `f"{elapsed:,.1f}"` |
| `tick = …` | `snapshot.tick` | direct |
| `delivered: …` | `snapshot.orders` | `sum(o.status == "DELIVERED")` |
| `failed: …` | `snapshot.orders` | `sum(o.status == "FAILED")` |
| `pending: …` | `snapshot.orders` | `total – delivered – failed – returned` |
| `RETURNED: …` | `snapshot.orders` | `sum(o.status == "RETURNED")` |
| `rate: …%` | derived | `returned / (delivered + returned)` if > 0 |
| `queue: N` | `snapshot.stores` | `sum(s.queue_depth for s in stores)` |
| `mean coverage_radius` | `snapshot.stores` | `mean(s.coverage_radius)` |
| scenario name | `snapshot.scenario_name` | direct |
| intensity bar | `snapshot.demand_intensity` | direct (0.0–1.0) |

All values are computable frame-by-frame from snapshot fields. No accumulated state
is held in the renderer. This is consistent with the observer-invariance guarantee.

### 5.6 HUD values that are NOT in the snapshot

The requirements mention `mean_store_wait_time` and courier utilization. These are
episode-accumulated KPIs from `KPICollector.summary()`, not real-time snapshot data.
The renderer does not have access to them.

**⚑ FLAG to Architect — Amendment request SA-4 (optional):**
`WorldSnapshot.episode_kpis: dict[str, float]` — a thin, optional dict of episode-
to-date KPI values (mean_store_wait_time, courier_utilization, mean_return_leg_time)
populated by the Simulator/env layer at each snapshot. This would allow the HUD to
show episode averages in real time. Without it, the HUD shows only the instantaneous
proxies listed in §5.5.

This is flagged as "optional" because the proxies (live queue depth, live return
rate from order counts, demand intensity bar) already satisfy the requirements'
acceptance criteria U-3 and U-4. The accumulated KPIs are bonus information.

---

## 6. Flagged Architecture Amendments

Summary of all fields the designer requires that are not currently in the Architect's
snapshot spec. In priority order:

| # | Field | On | Required for | Priority |
|---|-------|----|-------------|----------|
| SA-1 | `active_prep_count: int` | `StoreSnapshot` | Store slot utilization bar (§2.5) | Medium |
| SA-2 | `capacity: int` | `StoreSnapshot` | Normalize utilization bar (§2.5) | Medium (depends on SA-1) |
| SA-3 | `demand_pattern: str` | `WorldSnapshot` | Display pattern type in overlay and HUD (§4.1, §4.2) | Low (graceful fallback if absent) |
| SA-4 | `episode_kpis: dict[str, float]` | `WorldSnapshot` | Show accumulated wait/utilization in HUD (§5.6) | Optional |

**Not flagged — already available:**
`OrderSnapshot.assigned_courier_id` (used in §2.4 for the return-path dashed line)
is already defined in `protocol.py` and populated in `WorldState.snapshot()` at line
106. The courier phase strings `"waiting-at-store"` and `"returning"` are already
emitted by the Simulator and surfaced through `CourierSnapshot.status` — no new
snapshot field is required for either.

**SA-1 and SA-2 are a pair.** Both are needed to show a meaningful utilization bar.
If neither is added, the queue badge on the store (§2.2) is still a clear bottleneck
signal; the utilization bar is bonus readability.

**SA-3 is a single string.** The Architect already computes `current_intensity` on
the demand generator; `demand_pattern` would be the generator's type label. It is one
line of code to add to `WorldState.snapshot()`. Without it, the overlay still shows
scenario name + intensity bar (two of three rows), which satisfies acceptance criteria.

**SA-4 is truly optional.** The live-proxy approach in §5.5 satisfies all stated
acceptance criteria (U-3 and U-4). SA-4 is listed here for the Architect's awareness
in case a future increment adds a richer HUD.

---

## 7. ASCII Layout Reference

### 7.1 Viewport (900 × 580 px)

```
+═══════════════════════════════════════════════════════════+ y=0
║  ┌──────────────────────────┐                             ║
║  │ balanced                 │   ← scenario overlay       ║
║  │ ▶ ████████░░░  72%        │     x=10, y=10, 198×50 px ║
║  └──────────────────────────┘                             ║
║                                                           ║
║     ╭────────────────────────────────────────╮           ║
║     │  coverage circle (translucent fill)     │           ║
║     │                                         │           ║
║     │        ■ warehouse_a              ■[3]  │  ← store  ║
║     │       (yellow sq, no badge)     (badge) │           ║
║     │                  ◌ returned order        │           ║
║     │                 …                        │           ║
║     │      ● free      ●→store                 │           ║
║     │      (teal)      (blue)                  │           ║
║     │                                          │           ║
║     │             ✕ failed order               │           ║
║     │             ○ delivered                  │           ║
║     ╰────────────────────────────────────────╯           ║
║                                                           ║
+═══════════════════════════════════════════════════════════+ y=580
```

### 7.2 HUD (900 × 180 px, y=580–760)

```
──────────────────────────────────────────────────────────── y=580 (separator)
  COL A (x=12–268)          │ COL B (x=292–551)  │ COL C (x=564–888)
                             │                    │
  t = 12345.0 s  tick=3800   │ SCENARIO           │ LEGEND
  delivered: 82  failed: 3   │ balanced           │ ■store    ●free
  pending:   21              │ ▶ ████████░░░ 72%  │ ●→store   ●at-store
  RETURNED:   4  rate: 4.7%  │ pattern: burst     │ ●waiting  ●→cust
  queue:  2  waiting         │ ─────────────────  │ ●return   ○delivd
  mean_r: 682.5              │ 1=light 2=bal 3=hvy│ ✕failed   ◇RETND
                             │ ↵ restarts episode  │ ●active
──────────────────────────────────────────────────────────── y=760
```

Column dividers: 1 px vertical lines at x=280 and x=560, full HUD height.

---

## 8. Transition from Current Renderer

The following is a change summary for the engineer, mapping current
`pygame_renderer.py` constructs to their replacements:

| Current | Replacement / Addition |
|---------|----------------------|
| `_HUD_H = 112` | `_HUD_H = 180` |
| `window_h = 700` | `window_h = 760` |
| `_COURIER["free"] = (70,200,70)` | `(30, 190, 160)` — teal (colorblind fix) |
| `_ORDER["DELIVERED"] = (70,200,70)` | `(30, 190, 160)` — teal (colorblind fix) |
| No `_ORDER["RETURNED"]` | Add `"RETURNED": (210, 75, 55)` |
| No `_COURIER["waiting-at-store"]` | Add `"waiting-at-store": (200, 140, 0)` |
| No `_COURIER["returning"]` | Add `"returning": (210, 75, 55)` |
| Terminal orders: all `radius=3`, filled circles | DELIVERED → hollow teal ring; FAILED → red cross; RETURNED → hollow coral diamond ◇ |
| `_draw_hud` — 3 text rows + 8-item right legend | Extend to 6 rows, 3-column layout, 11-item legend |
| No store queue badge | Add badge when `store.queue_depth > 0` |
| No scenario overlay | Add semi-transparent top-left overlay (§4.1) |
| No key-based scenario switching | Add key event handling in `consume()` event loop |

**No changes to:** coverage circle drawing, `_to_screen` / `_r_to_px` helpers,
snapshot consumption protocol, `_ensure_init`, `close()`, lazy pygame import.

---

## 9. What Is Not Designed Here

The following items are explicitly out of this spec:

- The callback interface for `on_preset_switch` — that is an API contract for the
  engineer, not a visual design decision.
- The exact ordering of scenario YAML loading when a switch is requested — that is
  application-level logic, not renderer logic.
- Any visual treatment of the B11 stretch goal (in-episode preset switch animation)
  beyond the 1-second confirmation banner described in §4.3.
- Font face or size adjustments. The existing `pygame.font.Font(None, 15/17)` system
  font is retained; line spacing is designed to fit within 18 px rows.
- The pygame renderer's test scaffolding — no change to the headless renderer or
  test isolation patterns.

---

## 10. Engineer Handoff Note

**Pure-additive draws (no new snapshot fields required):** Every visual change in
this spec can be implemented today against the snapshot fields already defined in
`protocol.py` and populated by `WorldState.snapshot()`. This includes: the store
queue badge and stress-tint (from `store.queue_depth`), all six courier phase colors
and secondary marks (from `courier.status` — `"waiting-at-store"` and `"returning"`
are already emitted by the Simulator), all three terminal-order shapes DELIVERED/
FAILED/RETURNED (from `order.status`), the return-path dashed line (from
`order.assigned_courier_id` matched to courier positions), the scenario overlay (from
`snapshot.scenario_name` and `snapshot.demand_intensity`), and the extended HUD
columns.

**Changes requiring a new snapshot field:** Three of the four flagged amendments in
§6 affect visual features that degrade gracefully without the field. SA-1 + SA-2
(`active_prep_count` / `capacity` on `StoreSnapshot`) are needed only for the store
slot-utilization bar in §2.5; the queue badge alone already satisfies acceptance
criterion U-2. SA-3 (`demand_pattern: str` on `WorldSnapshot`) is needed for the
third row of the scenario overlay and the HUD pattern label; without it, both areas
show "—" and the two-row fallback still satisfies U-3. SA-4 (`episode_kpis`) is
purely optional and is not needed for any acceptance criterion.

**Recommended implementation order:** (1) palette constants + courier phase map
extension; (2) terminal-order shapes (ring / cross / diamond); (3) store queue badge
and tint; (4) HUD column layout + scenario overlay; (5) return-path dashed line;
(6) scenario-switch key handler + confirmation banner. Steps 1–4 are independent of
any new snapshot fields. Step 5 depends only on `assigned_courier_id`, which is
already in the snapshot. Step 6 requires the app-layer `on_preset_switch` callback
wiring, which is out of scope for this spec.

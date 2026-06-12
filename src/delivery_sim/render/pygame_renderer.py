"""
PygameRenderer — live pygame visualization of the WorldSnapshot stream.

Layer: Visualization (pure consumer; never calls back into engine or env).

pygame is OPTIONAL (``pip install 'delivery-sim[render]'``).  This module may
be imported without pygame installed; pygame is lazy-loaded inside __init__ so
that the core tests stay pygame-free.

Color / shape legend
---------------------
World viewport
  Amber square (tint shifts red under load)   store; badge shows queue depth
  Yellow translucent circle                   store coverage_radius
  Teal   filled circle                        courier — free
  Blue   filled circle                        courier — en-route-store
  Violet filled circle                        courier — at-store
  Amber  filled circle + □                    courier — waiting-at-store (queue)
  Orange filled circle                        courier — en-route-customer
  Coral  filled circle + ‹                    courier — returning (refused delivery)
  Grey   dot                                  order — CREATED
  Blue   dot                                  order — ASSIGNED
  Violet dot                                  order — PREPARING / PICKED_UP
  Orange dot                                  order — IN_TRANSIT
  Teal   hollow ring ○                        order — DELIVERED
  Red    cross ✕                              order — FAILED
  Coral  hollow diamond ◇                     order — RETURNED (refused at door)
  Coral  dashed line                          return-path: courier ↔ refused location

HUD (bottom 180 px, 3 columns)
  Col A  (x=12–268):  sim time + tick / delivered / failed / RETURNED + return
                      rate / pending / total queue depth / mean coverage_radius
  Col B  (x=292–551): active scenario name / demand intensity bar /
                      switch-preset affordance (keys 1/2/3)
  Col C  (x=564–888): two-sub-column legend
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from delivery_sim.render.protocol import WorldSnapshot

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_BG              = (20, 20, 30)
_HUD_BG          = (30, 30, 45)
_HUD_LINE        = (70, 70, 95)
_DIVIDER         = (70, 70, 95)
_TEXT            = (210, 210, 210)
_TEXT_DIM        = (130, 130, 155)
_WHITE           = (255, 255, 255)

_STORE_COLOR     = (255, 220, 50)
_COVERAGE_FILL:   tuple[int, int, int, int] = (255, 220, 50, 22)
_COVERAGE_BORDER: tuple[int, int, int, int] = (255, 220, 50, 140)
_QUEUE_BADGE     = (210, 40, 40)

_COURIER: dict[str, tuple[int, int, int]] = {
    "free":               (30, 190, 160),   # teal
    "en-route-store":     (60, 120, 240),   # blue
    "at-store":           (160, 70, 220),   # violet
    "waiting-at-store":   (200, 140, 0),    # amber  (NEW — store queue)
    "en-route-customer":  (255, 130, 40),   # orange
    "returning":          (210, 75, 55),    # coral  (NEW — return leg)
}
_ORDER: dict[str, tuple[int, int, int]] = {
    "CREATED":   (160, 160, 165),
    "ASSIGNED":  (60, 120, 240),
    "PREPARING": (160, 70, 220),
    "PICKED_UP": (255, 130, 40),
    "IN_TRANSIT": (255, 130, 40),
    "DELIVERED": (30, 190, 160),   # teal  (was green; decoupled from free-courier)
    "FAILED":    (210, 45, 45),    # red
    "RETURNED":  (210, 75, 55),    # coral  (NEW)
}

_INTENSITY_LOW  = (60, 120, 240)   # blue  (< 34 %)
_INTENSITY_MED  = (200, 140, 0)    # amber (34–67 %)
_INTENSITY_HIGH = (210, 75, 55)    # coral (> 67 %)

# Human-readable labels for generator_type strings (SA-3 resolved)
_PATTERN_LABELS: dict[str, str] = {
    "PoissonDemandGenerator":      "constant",
    "DailyProfileDemandGenerator": "daily-profile",
    "BurstDemandGenerator":        "burst",
}

_HUD_H       = 180
_STORE_HALF  = 7
_BANNER_DUR  = 1.0   # seconds the switch-confirmation banner stays visible


def _store_fill(queue_depth: int) -> tuple[int, int, int]:
    """Store square fill color; tint shifts toward red as queue grows."""
    if queue_depth >= 5:
        return (220, 70, 30)
    if queue_depth >= 3:
        return (255, 130, 30)
    if queue_depth >= 1:
        return (255, 175, 30)
    return _STORE_COLOR


def _intensity_color(intensity: float) -> tuple[int, int, int]:
    if intensity >= 0.67:
        return _INTENSITY_HIGH
    if intensity >= 0.34:
        return _INTENSITY_MED
    return _INTENSITY_LOW


class PygameRenderer:
    """Pygame-based live renderer for the WorldSnapshot stream.

    Satisfies the SnapshotConsumer protocol without inheriting from it.
    The renderer reads ONLY snapshot fields; it never touches the simulator,
    world state, or any other engine internals.

    Scenario switching (stretch goal B11): if *on_preset_switch* is supplied,
    pressing 1/2/3 fires the callback with "light"/"balanced"/"heavy" and shows
    a 1-second banner.  The callback must trigger an episode restart — it never
    mutates a live run.
    """

    def __init__(
        self,
        world_width: float,
        world_height: float,
        window_w: int = 900,
        window_h: int = 760,
        fps: int = 30,
        on_preset_switch: Callable[[str], None] | None = None,
    ) -> None:
        """Create the renderer (window opens lazily on the first consume() call).

        Args:
            world_width:       Simulation world width in world-units.
            world_height:      Simulation world height in world-units.
            window_w:          Total window pixel width.
            window_h:          Total window pixel height (viewport + HUD).
            fps:               Target frame rate; 0 = unlimited (CI-friendly).
            on_preset_switch:  Optional callback for preset switching.  Called
                               with the preset name when the user presses 1/2/3.
                               Should restart the episode; never mutates a live
                               run mid-episode.
        """
        try:
            import pygame as _pg  # noqa: PLC0415
            self._pg: Any = _pg
        except ImportError as exc:
            raise ImportError(
                "PygameRenderer requires pygame. "
                "Install with: pip install 'delivery-sim[render]'"
            ) from exc

        self._world_w = world_width
        self._world_h = world_height
        self._win_w = window_w
        self._vp_h = window_h - _HUD_H
        self._total_h = window_h
        self._fps = fps
        self._on_preset_switch = on_preset_switch
        self._switch_msg: str = ""
        self._switch_msg_until: float = 0.0
        self._initialized = False

        self._screen: Any = None
        self._alpha_surf: Any = None
        self._font_sm: Any = None
        self._font_lg: Any = None
        self._clock: Any = None

    # ------------------------------------------------------------------
    # SnapshotConsumer protocol
    # ------------------------------------------------------------------

    def consume(self, snapshot: WorldSnapshot) -> None:
        """Draw *snapshot* to the window.

        Reads only snapshot fields.  Drains the OS event queue so the window
        stays responsive; handles preset-switch keys 1/2/3 if a callback was
        registered.
        """
        self._ensure_init()
        pg = self._pg

        _PRESET_KEYS = {pg.K_1: "light", pg.K_2: "balanced", pg.K_3: "heavy"}
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return
            if event.type == pg.KEYDOWN:
                name = _PRESET_KEYS.get(event.key)
                if name is not None:
                    self._switch_msg = f"Switching to {name}… (restart)"
                    self._switch_msg_until = snapshot.elapsed + _BANNER_DUR
                    if self._on_preset_switch is not None:
                        self._on_preset_switch(name)

        self._draw(snapshot)
        pg.display.flip()
        self._clock.tick(self._fps)

    def close(self) -> None:
        """Shut down pygame and release the display."""
        if self._initialized:
            self._pg.quit()
            self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        pg = self._pg
        pg.init()
        pg.display.set_caption("delivery_sim  |  B-realistic visualizer")
        self._screen = pg.display.set_mode((self._win_w, self._total_h))
        self._alpha_surf = pg.Surface((self._win_w, self._vp_h), pg.SRCALPHA)
        self._font_sm = pg.font.Font(None, 15)
        self._font_lg = pg.font.Font(None, 17)
        self._clock = pg.time.Clock()
        self._initialized = True

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        """Map world-space (wx, wy) to viewport pixel coords (Y-flipped)."""
        sx = int(wx / self._world_w * self._win_w)
        sy = int((1.0 - wy / self._world_h) * self._vp_h)
        return sx, sy

    def _r_to_px(self, r: float) -> int:
        """Scale a world-unit radius to pixels along the horizontal axis."""
        return max(1, int(r / self._world_w * self._win_w))

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, snapshot: WorldSnapshot) -> None:
        pg = self._pg
        screen = self._screen
        alpha = self._alpha_surf

        screen.fill(_BG)
        alpha.fill((0, 0, 0, 0))

        # --- Alpha surface: coverage circles + return-path dashed lines ---
        # Both drawn here so they appear behind the opaque entity markers.

        for store in snapshot.stores:
            cx, cy = self._to_screen(store.x, store.y)
            r_px = self._r_to_px(store.coverage_radius)
            pg.draw.circle(alpha, _COVERAGE_FILL, (cx, cy), r_px)
            pg.draw.circle(alpha, _COVERAGE_BORDER, (cx, cy), r_px, 2)

        returning_map = {
            c.courier_id: c for c in snapshot.couriers if c.status == "returning"
        }
        for order in snapshot.orders:
            if (
                order.status == "RETURNED"
                and order.assigned_courier_id is not None
                and order.assigned_courier_id in returning_map
            ):
                cr = returning_map[order.assigned_courier_id]
                self._draw_dashed_line(
                    alpha,
                    (210, 75, 55, 80),
                    self._to_screen(cr.x, cr.y),
                    self._to_screen(order.customer_x, order.customer_y),
                )

        screen.blit(alpha, (0, 0))

        # --- Orders: terminal states use distinct shapes ---

        for order in snapshot.orders:
            ox, oy = self._to_screen(order.customer_x, order.customer_y)
            color = _ORDER.get(order.status, (160, 160, 165))
            st = order.status
            if st == "DELIVERED":
                pg.draw.circle(screen, color, (ox, oy), 4, 1)           # hollow ring
            elif st == "FAILED":
                pg.draw.line(screen, color, (ox - 3, oy - 3), (ox + 3, oy + 3), 1)
                pg.draw.line(screen, color, (ox + 3, oy - 3), (ox - 3, oy + 3), 1)
            elif st == "RETURNED":
                pts = [(ox, oy - 4), (ox + 4, oy), (ox, oy + 4), (ox - 4, oy)]
                pg.draw.polygon(screen, color, pts, 1)                   # hollow diamond
            else:
                pg.draw.circle(screen, color, (ox, oy), 5)

        # --- Couriers: filled circle + phase-specific inner marks ---

        for courier in snapshot.couriers:
            cx, cy = self._to_screen(courier.x, courier.y)
            color = _COURIER.get(courier.status, (200, 200, 200))
            pg.draw.circle(screen, color, (cx, cy), 7)
            pg.draw.circle(screen, _WHITE, (cx, cy), 7, 1)

            if courier.status == "returning":
                # left-pointing chevron ‹ centred inside the circle
                pg.draw.line(screen, _WHITE, (cx + 2, cy), (cx - 1, cy - 2), 1)
                pg.draw.line(screen, _WHITE, (cx + 2, cy), (cx - 1, cy + 2), 1)
            elif courier.status == "waiting-at-store":
                # hollow 4×4 square — "blocked" cue
                pg.draw.rect(screen, _WHITE, (cx - 2, cy - 2, 4, 4), 1)

        # --- Stores: on top, queue tint + badge + ID label ---

        for store in snapshot.stores:
            sx, sy = self._to_screen(store.x, store.y)
            q = store.queue_depth
            fill = _store_fill(q)
            rect = (sx - _STORE_HALF, sy - _STORE_HALF, _STORE_HALF * 2, _STORE_HALF * 2)
            pg.draw.rect(screen, fill, rect)
            pg.draw.rect(screen, _WHITE, rect, 1)

            lbl = store.store_id if len(store.store_id) <= 12 else store.store_id[:11] + "…"
            ls = self._font_sm.render(lbl, True, _TEXT_DIM)
            screen.blit(ls, (sx - ls.get_width() // 2, sy - _STORE_HALF - 11))

            if q > 0:
                bx, by = sx + _STORE_HALF, sy - _STORE_HALF
                pg.draw.circle(screen, _QUEUE_BADGE, (bx, by), 7)
                bt = str(q) if q <= 9 else "9+"
                bs = self._font_sm.render(bt, True, _WHITE)
                screen.blit(bs, (bx - bs.get_width() // 2, by - bs.get_height() // 2))

        # --- Viewport overlays ---

        if snapshot.scenario_name:
            self._draw_scenario_overlay(snapshot)

        if self._switch_msg and snapshot.elapsed <= self._switch_msg_until:
            self._draw_switch_banner()

        self._draw_hud(snapshot)

    def _draw_scenario_overlay(self, snapshot: WorldSnapshot) -> None:
        """Semi-transparent scenario + intensity + pattern overlay at top-left."""
        pg = self._pg
        screen = self._screen
        ox, oy, ow, oh = 10, 10, 198, 62   # 62 px: 3 rows now that SA-3 is resolved

        surf = pg.Surface((ow, oh), pg.SRCALPHA)
        surf.fill((20, 20, 30, 180))
        pg.draw.rect(surf, (70, 70, 95, 255), (0, 0, ow, oh), 1)
        screen.blit(surf, (ox, oy))

        # Row 1: scenario name
        s = self._font_lg.render(snapshot.scenario_name, True, _TEXT)
        screen.blit(s, (ox + 6, oy + 8))

        # Row 2: intensity bar
        s = self._font_sm.render("▶", True, _TEXT_DIM)
        screen.blit(s, (ox + 6, oy + 26))
        bar_x, bar_y, bar_w, bar_h = ox + 20, oy + 28, 80, 8
        pg.draw.rect(screen, (50, 55, 75), (bar_x, bar_y, bar_w, bar_h))
        intensity = max(0.0, min(1.0, snapshot.demand_intensity))
        filled = int(bar_w * intensity)
        if filled > 0:
            pg.draw.rect(screen, _intensity_color(intensity), (bar_x, bar_y, filled, bar_h))
        s = self._font_sm.render(f"{intensity * 100:.0f}%", True, _TEXT_DIM)
        screen.blit(s, (bar_x + bar_w + 4, oy + 26))

        # Row 3: demand pattern (SA-3 resolved — field now in snapshot)
        pattern_lbl = _PATTERN_LABELS.get(
            snapshot.demand_pattern, snapshot.demand_pattern
        ) or "—"
        s = self._font_sm.render(f"pattern: {pattern_lbl}", True, _TEXT_DIM)
        screen.blit(s, (ox + 6, oy + 44))

    def _draw_switch_banner(self) -> None:
        """Centred 1-second banner confirming a preset-switch restart."""
        pg = self._pg
        screen = self._screen
        bw, bh = 400, 36
        bx = (self._win_w - bw) // 2
        by = (self._vp_h - bh) // 2

        banner = pg.Surface((bw, bh), pg.SRCALPHA)
        banner.fill((20, 20, 30, 210))
        pg.draw.rect(banner, (70, 70, 95, 255), (0, 0, bw, bh), 1)
        screen.blit(banner, (bx, by))

        s = self._font_lg.render(self._switch_msg, True, _TEXT)
        screen.blit(s, (bx + (bw - s.get_width()) // 2, by + (bh - s.get_height()) // 2))

    def _draw_dashed_line(
        self,
        surface: Any,
        color: tuple[int, int, int, int],
        start: tuple[int, int],
        end: tuple[int, int],
        seg: int = 4,
        gap: int = 4,
    ) -> None:
        """Draw a dashed line on an SRCALPHA *surface*."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        pos = 0.0
        drawing = True
        while pos < length:
            step = float(seg if drawing else gap)
            next_pos = min(pos + step, length)
            if drawing:
                self._pg.draw.line(
                    surface, color,
                    (int(start[0] + ux * pos),       int(start[1] + uy * pos)),
                    (int(start[0] + ux * next_pos),   int(start[1] + uy * next_pos)),
                    1,
                )
            pos = next_pos
            drawing = not drawing

    # ------------------------------------------------------------------
    # HUD (3 columns)
    # ------------------------------------------------------------------

    def _draw_hud(self, snapshot: WorldSnapshot) -> None:
        """Draw the 3-column HUD strip."""
        pg = self._pg
        screen = self._screen
        hy = self._vp_h

        pg.draw.rect(screen, _HUD_BG, (0, hy, self._win_w, _HUD_H))
        pg.draw.line(screen, _HUD_LINE, (0, hy), (self._win_w, hy), 1)
        pg.draw.line(screen, _DIVIDER, (280, hy), (280, hy + _HUD_H), 1)
        pg.draw.line(screen, _DIVIDER, (560, hy), (560, hy + _HUD_H), 1)

        self._draw_hud_col_a(snapshot, hy)
        self._draw_hud_col_b(snapshot, hy)
        self._draw_hud_col_c(hy)

    def _draw_hud_col_a(self, snapshot: WorldSnapshot, hy: int) -> None:
        """Column A (x=12): episode metrics."""
        screen = self._screen
        x = 12
        fs = self._font_sm
        fl = self._font_lg

        n_del  = sum(1 for o in snapshot.orders if o.status == "DELIVERED")
        n_fail = sum(1 for o in snapshot.orders if o.status == "FAILED")
        n_ret  = sum(1 for o in snapshot.orders if o.status == "RETURNED")
        n_pend = len(snapshot.orders) - n_del - n_fail - n_ret
        attempts = n_del + n_ret
        rate = n_ret / attempts if attempts > 0 else 0.0
        total_q = sum(s.queue_depth for s in snapshot.stores)
        mean_r = (
            sum(s.coverage_radius for s in snapshot.stores) / len(snapshot.stores)
            if snapshot.stores else 0.0
        )

        rows = [
            (fl, _TEXT,
             f"t = {snapshot.elapsed:,.1f} s   tick = {snapshot.tick}"),
            (fs, _TEXT,
             f"delivered: {n_del:4d}   failed: {n_fail:3d}"),
            (fs, _TEXT_DIM,
             f"pending:   {n_pend:4d}"),
            (fs, _TEXT if n_ret > 0 else _TEXT_DIM,
             f"RETURNED:  {n_ret:4d}   rate: {rate:4.1f}%"),
            (fs, _TEXT if total_q > 0 else _TEXT_DIM,
             f"queue:  {total_q}  waiting"),
            (fs, _TEXT,
             f"mean radius: {mean_r:.1f}"),
        ]
        for i, (fnt, color, text) in enumerate(rows):
            s = fnt.render(text, True, color)
            screen.blit(s, (x, hy + 8 + i * 18))

    def _draw_hud_col_b(self, snapshot: WorldSnapshot, hy: int) -> None:
        """Column B (x=292): scenario name, intensity bar, switch affordance."""
        pg = self._pg
        screen = self._screen
        x = 292
        fs = self._font_sm
        fl = self._font_lg

        screen.blit(fs.render("SCENARIO", True, _TEXT_DIM), (x + 4, hy + 6))

        name = snapshot.scenario_name or "—"
        screen.blit(fl.render(name, True, _TEXT), (x + 4, hy + 20))

        screen.blit(fs.render("intensity", True, _TEXT_DIM), (x + 4, hy + 38))
        bar_x, bar_y, bar_w, bar_h = x + 66, hy + 40, 80, 8
        pg.draw.rect(screen, (50, 55, 75), (bar_x, bar_y, bar_w, bar_h))
        intensity = max(0.0, min(1.0, snapshot.demand_intensity))
        filled = int(bar_w * intensity)
        if filled > 0:
            pg.draw.rect(screen, _intensity_color(intensity), (bar_x, bar_y, filled, bar_h))
        screen.blit(
            fs.render(f"{intensity * 100:.0f}%", True, _TEXT_DIM),
            (bar_x + bar_w + 4, hy + 38),
        )

        pattern_lbl = _PATTERN_LABELS.get(
            snapshot.demand_pattern, snapshot.demand_pattern
        ) or "—"
        screen.blit(fs.render(f"pattern: {pattern_lbl}", True, _TEXT_DIM), (x + 4, hy + 56))

        pg.draw.line(screen, _DIVIDER, (x + 4, hy + 70), (551, hy + 70), 1)

        screen.blit(fs.render("1=light  2=balanced  3=heavy", True, _TEXT), (x + 4, hy + 78))
        screen.blit(fs.render("↵ restarts episode", True, _TEXT_DIM), (x + 4, hy + 93))

    def _draw_hud_col_c(self, hy: int) -> None:
        """Column C (x=564): two-sub-column legend."""
        pg = self._pg
        screen = self._screen
        fs = self._font_sm
        lx_l = 568
        lx_r = 732

        screen.blit(fs.render("LEGEND", True, _TEXT_DIM), (lx_l, hy + 6))

        def circle(x: int, y: int, color: tuple[int, ...]) -> None:
            pg.draw.circle(screen, color, (x, y), 5)

        def ring(x: int, y: int, color: tuple[int, ...]) -> None:
            pg.draw.circle(screen, color, (x, y), 4, 1)

        def rect_swatch(x: int, y: int, color: tuple[int, ...]) -> None:
            pg.draw.rect(screen, color, (x - 4, y - 4, 8, 8))

        def cross(x: int, y: int, color: tuple[int, ...]) -> None:
            pg.draw.line(screen, color, (x - 3, y - 3), (x + 3, y + 3), 1)
            pg.draw.line(screen, color, (x + 3, y - 3), (x - 3, y + 3), 1)

        def diamond(x: int, y: int, color: tuple[int, ...]) -> None:
            pts = [(x, y - 4), (x + 4, y), (x, y + 4), (x - 4, y)]
            pg.draw.polygon(screen, color, pts, 1)

        def lbl(lx: int, dy: int, text: str) -> None:
            s = fs.render(text, True, _TEXT_DIM)
            screen.blit(s, (lx + 14, hy + dy - 5))

        left_rows = [
            (20,  rect_swatch, _STORE_COLOR,                    "store"),
            (33,  circle,      _COURIER["en-route-store"],      "→store"),
            (46,  circle,      _COURIER["waiting-at-store"],    "waiting"),
            (59,  circle,      _COURIER["returning"],           "returning"),
            (72,  cross,       _ORDER["FAILED"],                "failed"),
            (85,  circle,      _ORDER["CREATED"],               "active order"),
        ]
        right_rows = [
            (20,  circle,      _COURIER["free"],                "free"),
            (33,  circle,      _COURIER["at-store"],            "at-store"),
            (46,  circle,      _COURIER["en-route-customer"],   "→customer"),
            (59,  ring,        _ORDER["DELIVERED"],             "delivered"),
            (72,  diamond,     _ORDER["RETURNED"],              "RETURNED"),
        ]

        for dy, draw_fn, color, label in left_rows:
            draw_fn(lx_l + 6, hy + dy, color)
            lbl(lx_l, dy, label)

        for dy, draw_fn, color, label in right_rows:
            draw_fn(lx_r + 6, hy + dy, color)
            lbl(lx_r, dy, label)

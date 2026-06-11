"""
PygameRenderer — live pygame visualization of the WorldSnapshot stream.

Layer: Visualization (pure consumer; never calls back into engine or env).

pygame is OPTIONAL (``pip install 'delivery-sim[render]'``).  This module may
be imported without pygame installed; pygame is lazy-loaded inside __init__ so
that the core 310 tests stay pygame-free.

Color legend
------------
World viewport
  Yellow square   store location marker
  Yellow circle   store coverage_radius (translucent fill + solid border)
  Green  dot      courier  — free
  Blue   dot      courier  — en-route-store
  Purple dot      courier  — at-store
  Orange dot      courier  — en-route-customer
  Grey   dot      order    — CREATED
  Blue   dot      order    — ASSIGNED
  Purple dot      order    — PREPARING / PICKED_UP
  Orange dot      order    — IN_TRANSIT
  Green  dot      order    — DELIVERED  (small)
  Red    dot      order    — FAILED     (small)

HUD (bottom strip)
  sim time, tick, delivered / failed / pending counts, mean coverage_radius
"""

from __future__ import annotations

from typing import Any

from delivery_sim.render.protocol import WorldSnapshot

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
_BG = (20, 20, 30)
_HUD_BG = (30, 30, 45)
_HUD_LINE = (70, 70, 95)
_TEXT = (210, 210, 210)
_TEXT_DIM = (150, 150, 150)

_STORE_COLOR = (255, 220, 50)
_COVERAGE_FILL: tuple[int, int, int, int] = (255, 220, 50, 22)
_COVERAGE_BORDER: tuple[int, int, int, int] = (255, 220, 50, 140)

_COURIER: dict[str, tuple[int, int, int]] = {
    "free": (70, 200, 70),
    "en-route-store": (70, 130, 255),
    "at-store": (180, 70, 255),
    "en-route-customer": (255, 140, 30),
}
_ORDER: dict[str, tuple[int, int, int]] = {
    "CREATED": (170, 170, 170),
    "ASSIGNED": (70, 130, 255),
    "PREPARING": (180, 70, 255),
    "PICKED_UP": (255, 140, 30),
    "IN_TRANSIT": (255, 140, 30),
    "DELIVERED": (70, 200, 70),
    "FAILED": (255, 60, 60),
}

_HUD_H = 112
_STORE_HALF = 7


class PygameRenderer:
    """Pygame-based live renderer for the WorldSnapshot stream.

    Satisfies the SnapshotConsumer protocol without inheriting from it.
    The renderer reads ONLY snapshot fields; it never touches the simulator,
    world state, or any other engine internals.
    """

    def __init__(
        self,
        world_width: float,
        world_height: float,
        window_w: int = 900,
        window_h: int = 700,
        fps: int = 30,
    ) -> None:
        """Create the renderer (window opens lazily on the first consume() call).

        Args:
            world_width:  Simulation world width in world-units (from ScenarioConfig).
            world_height: Simulation world height in world-units.
            window_w:     Total window pixel width.
            window_h:     Total window pixel height (viewport + HUD).
            fps:          Target frame rate; 0 = unlimited (useful for CI tests).
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
        self._initialized = False

        # Assigned in _ensure_init()
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
        stays responsive; exits cleanly on a close event.
        """
        self._ensure_init()
        pg = self._pg

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return

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
        pg.display.set_caption("delivery_sim  |  coverage visualizer")
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

        # Background
        screen.fill(_BG)
        alpha.fill((0, 0, 0, 0))

        # Coverage circles (translucent fill + opaque border) on alpha surface
        for store in snapshot.stores:
            cx, cy = self._to_screen(store.x, store.y)
            r_px = self._r_to_px(store.coverage_radius)
            pg.draw.circle(alpha, _COVERAGE_FILL, (cx, cy), r_px)
            pg.draw.circle(alpha, _COVERAGE_BORDER, (cx, cy), r_px, 2)
        screen.blit(alpha, (0, 0))

        # Orders — active ones at full size, terminal ones as tiny dots
        for order in snapshot.orders:
            ox, oy = self._to_screen(order.customer_x, order.customer_y)
            color = _ORDER.get(order.status, (170, 170, 170))
            terminal = order.status in ("DELIVERED", "FAILED")
            pg.draw.circle(screen, color, (ox, oy), 3 if terminal else 5)

        # Couriers — dot with white outline
        for courier in snapshot.couriers:
            cx, cy = self._to_screen(courier.x, courier.y)
            color = _COURIER.get(courier.status, (200, 200, 200))
            pg.draw.circle(screen, color, (cx, cy), 7)
            pg.draw.circle(screen, (255, 255, 255), (cx, cy), 7, 1)

        # Store markers on top (always visible even when inside a coverage circle)
        for store in snapshot.stores:
            sx, sy = self._to_screen(store.x, store.y)
            rect = (sx - _STORE_HALF, sy - _STORE_HALF, _STORE_HALF * 2, _STORE_HALF * 2)
            pg.draw.rect(screen, _STORE_COLOR, rect)
            pg.draw.rect(screen, (255, 255, 255), rect, 1)

        self._draw_hud(snapshot)

    def _draw_hud(self, snapshot: WorldSnapshot) -> None:
        pg = self._pg
        screen = self._screen
        hy = self._vp_h

        pg.draw.rect(screen, _HUD_BG, (0, hy, self._win_w, _HUD_H))
        pg.draw.line(screen, _HUD_LINE, (0, hy), (self._win_w, hy), 1)

        n_del = sum(1 for o in snapshot.orders if o.status == "DELIVERED")
        n_fail = sum(1 for o in snapshot.orders if o.status == "FAILED")
        n_pend = len(snapshot.orders) - n_del - n_fail
        mean_r = (
            sum(s.coverage_radius for s in snapshot.stores) / len(snapshot.stores)
            if snapshot.stores else 0.0
        )

        lines = [
            (self._font_lg, f"t = {snapshot.elapsed:9.1f} s    tick = {snapshot.tick}"),
            (self._font_sm, f"delivered={n_del:4d}  failed={n_fail:4d}  pending={n_pend:4d}"),
            (self._font_sm, f"mean coverage_radius = {mean_r:.1f}"),
        ]
        for i, (font, text) in enumerate(lines):
            surf = font.render(text, True, _TEXT)
            screen.blit(surf, (12, hy + 7 + i * 22))

        # Legend (right column)
        legend: list[tuple[str, tuple[int, int, int]]] = [
            ("store", _STORE_COLOR),
            ("free", _COURIER["free"]),
            ("→store", _COURIER["en-route-store"]),
            ("at-store", _COURIER["at-store"]),
            ("→customer", _COURIER["en-route-customer"]),
            ("order (active)", _ORDER["CREATED"]),
            ("delivered", _ORDER["DELIVERED"]),
            ("failed", _ORDER["FAILED"]),
        ]
        lx = self._win_w - 165
        font = self._font_sm
        for i, (label, color) in enumerate(legend):
            pg.draw.circle(screen, color, (lx + 6, hy + 11 + i * 13), 5)
            s = font.render(label, True, _TEXT_DIM)
            screen.blit(s, (lx + 15, hy + 5 + i * 13))

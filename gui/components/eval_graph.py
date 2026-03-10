"""Evaluation sparkline used in analysis mode."""

from __future__ import annotations

import pygame

from gui.constants import ACCENT, BG_CARD, DANGER, MUTED, SUCCESS, WHITE_COL


class EvalGraph:
    """Draws a small filled evaluation graph with hover selection."""

    def __init__(self, rect: pygame.Rect | tuple[int, int, int, int]) -> None:
        self.rect = pygame.Rect(rect)
        self.evals: list[float] = []
        self.hovered = -1
        self._font: pygame.font.Font | None = None

    def set_evals(self, eval_list: list[float]) -> None:
        """Replace the evaluation history."""
        self.evals = list(eval_list)

    def handle_event(self, event: pygame.event.Event) -> int | None:
        """Return the hovered or clicked point index when inside the graph."""
        if not self.evals:
            return None
        if event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN):
            pos = getattr(event, "pos", None)
            if pos and self.rect.collidepoint(pos):
                rel_x = pos[0] - self.rect.x
                idx = int(rel_x / max(self.rect.w, 1) * len(self.evals))
                idx = max(0, min(len(self.evals) - 1, idx))
                self.hovered = idx
                return idx
            if event.type == pygame.MOUSEMOTION:
                self.hovered = -1
        return None

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the evaluation graph and current hover marker."""
        if self._font is None:
            self._font = pygame.font.SysFont("segoeui", 12)

        pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=4)
        if not self.evals:
            return

        center_y = self.rect.centery
        step = self.rect.w / max(len(self.evals) - 1, 1)
        pygame.draw.line(surface, MUTED, (self.rect.left, center_y), (self.rect.right, center_y), 1)

        area = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
        points: list[tuple[int, int]] = []
        for i, ev in enumerate(self.evals):
            ev_c = max(-600, min(600, ev))
            px = int(i * step)
            py = int(self.rect.h / 2 - ev_c / 600 * (self.rect.h / 2 - 4))
            points.append((px, py))

        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            raw_total = self.evals[i] + self.evals[i + 1]
            col = SUCCESS if raw_total >= 0 else DANGER
            pygame.draw.polygon(
                area,
                (*col, 80),
                [(p0[0], self.rect.h // 2), p0, p1, (p1[0], self.rect.h // 2)],
            )
        surface.blit(area, self.rect.topleft)

        translated = [(self.rect.x + x, self.rect.y + y) for x, y in points]
        if len(translated) >= 2:
            pygame.draw.lines(surface, WHITE_COL, False, translated, 2)

        if 0 <= self.hovered < len(translated):
            hx, hy = translated[self.hovered]
            pygame.draw.circle(surface, ACCENT, (hx, hy), 5)
            label = f"{self.evals[self.hovered] / 100:+.1f}"
            txt = self._font.render(label, True, WHITE_COL)
            surface.blit(txt, (hx + 8, hy - 10))

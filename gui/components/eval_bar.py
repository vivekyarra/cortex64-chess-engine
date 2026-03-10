"""Vertical evaluation bar with smooth interpolation."""

from __future__ import annotations

import pygame

from gui.constants import BG_CARD, DANGER, GOLD, MUTED, WHITE_COL


class EvalBar:
    """Shows White/Black advantage as a vertical filled bar."""

    def __init__(self, rect: pygame.Rect | tuple[int, int, int, int]) -> None:
        self.rect = pygame.Rect(rect)
        self.target_eval = 0.0
        self.display_eval = 0.0
        self._label_font: pygame.font.Font | None = None

    def set_eval(self, eval_cp: float) -> None:
        """Set the target evaluation in centipawns."""
        self.target_eval = max(-1000.0, min(1000.0, float(eval_cp)))

    def update(self, dt_ms: float) -> None:
        """Smoothly lerp the displayed score to the latest target."""
        speed = dt_ms / 250.0
        self.display_eval += (self.target_eval - self.display_eval) * min(1.0, speed)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the bar and score label."""
        pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=4)

        ratio = (self.display_eval + 1000.0) / 2000.0
        ratio = max(0.02, min(0.98, ratio))
        white_h = int(self.rect.h * ratio)
        black_h = self.rect.h - white_h

        pygame.draw.rect(
            surface,
            WHITE_COL,
            (self.rect.x, self.rect.y, self.rect.w, white_h),
            border_top_left_radius=4,
            border_top_right_radius=4,
        )
        pygame.draw.rect(
            surface,
            (30, 30, 30),
            (self.rect.x, self.rect.y + white_h, self.rect.w, black_h),
            border_bottom_left_radius=4,
            border_bottom_right_radius=4,
        )

        if self._label_font is None:
            self._label_font = pygame.font.SysFont("segoeui", 13)
        ev = self.display_eval / 100.0
        if abs(ev) >= 9.9:
            label = "M" if ev > 0 else "-M"
        elif ev >= 0:
            label = f"+{ev:.1f}"
        else:
            label = f"{ev:.1f}"
        color = DANGER if ev < -0.5 else GOLD if ev > 0.5 else MUTED
        txt = self._label_font.render(label, True, color)
        surface.blit(txt, txt.get_rect(centerx=self.rect.centerx, top=self.rect.bottom + 5))

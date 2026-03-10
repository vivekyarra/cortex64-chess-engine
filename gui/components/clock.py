"""Chess clock component with increment and low-time warning styling."""

from __future__ import annotations

import pygame

from gui.constants import BG_CARD, DANGER, WHITE_COL


class ChessClock:
    """Tracks remaining time for White and Black."""

    def __init__(self, white_ms: int, black_ms: int, increment_ms: int = 0) -> None:
        self.times = {"white": int(white_ms), "black": int(black_ms)}
        self.increment = int(increment_ms)
        self.active: str | None = None
        self.running = False
        self.flagged: str | None = None
        self._font: pygame.font.Font | None = None

    def start(self, color_to_move: str) -> None:
        """Start the clock for the side to move."""
        self.active = color_to_move
        self.running = True

    def press(self, color_that_just_moved: str) -> None:
        """Switch turns and apply increment after a completed move."""
        if self.running:
            self.times[color_that_just_moved] += self.increment
            self.active = "black" if color_that_just_moved == "white" else "white"

    def pause(self) -> None:
        """Pause both clocks."""
        self.running = False

    def resume(self) -> None:
        """Resume counting down for the active side."""
        self.running = True

    def update(self, dt_ms: int) -> str | None:
        """Tick the active clock and return the flagged color if time expires."""
        if not self.running or self.active is None:
            return None
        self.times[self.active] = max(0, self.times[self.active] - int(dt_ms))
        if self.times[self.active] == 0:
            self.running = False
            self.flagged = self.active
            return self.active
        return None

    def format_time(self, color: str) -> str:
        """Return the time string for a color."""
        ms = self.times[color]
        secs = ms // 1000
        mins, seconds = divmod(secs, 60)
        if mins >= 10:
            return f"{mins}:{seconds:02d}"
        tenths = (ms % 1000) // 100
        return f"{mins}:{seconds:02d}" if mins > 0 else f"{seconds}.{tenths}"

    def draw(self, surface: pygame.Surface, rect: pygame.Rect | tuple[int, int, int, int], color: str) -> None:
        """Draw one half of the clock stack."""
        rect = pygame.Rect(rect)
        if self._font is None:
            self._font = pygame.font.SysFont("segoeui", 28, bold=True)
        is_low = self.times[color] < 30_000
        is_active = self.active == color and self.running
        bg_col = (60, 10, 20) if is_low else BG_CARD
        pygame.draw.rect(surface, bg_col, rect, border_radius=6)
        if is_active:
            pygame.draw.rect(surface, DANGER if is_low else WHITE_COL, rect, width=2, border_radius=6)
        txt_col = DANGER if is_low else WHITE_COL
        text = self._font.render(self.format_time(color), True, txt_col)
        surface.blit(text, text.get_rect(center=rect.center))

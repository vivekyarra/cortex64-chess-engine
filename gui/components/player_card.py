"""Player card UI component used in the sidebars."""

from __future__ import annotations

import pygame

from gui.constants import ACCENT, BG_CARD, GOLD, MUTED, WHITE_COL


class PlayerCard:
    """Displays avatar, name, rating, and active-turn glow."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        name: str = "Player",
        rating: int = 1200,
        color: str = "white",
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.name = name
        self.rating = rating
        self.color = color
        self.active = False
        self._font: pygame.font.Font | None = None
        self._sm_font: pygame.font.Font | None = None

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the player card."""
        if self._font is None:
            self._font = pygame.font.SysFont("segoeui", 16, bold=True)
            self._sm_font = pygame.font.SysFont("segoeui", 13)

        pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=8)
        if self.active:
            border_col = GOLD if self.color == "white" else ACCENT
            pygame.draw.rect(surface, border_col, (self.rect.x, self.rect.y, 4, self.rect.h), border_radius=4)

        av_cx = self.rect.x + 28
        av_cy = self.rect.centery
        fill = ACCENT if self.color == "white" else (60, 60, 90)
        pygame.draw.circle(surface, fill, (av_cx, av_cy), 18)
        initial = self._font.render((self.name[:1] or "?").upper(), True, WHITE_COL)
        surface.blit(initial, initial.get_rect(center=(av_cx, av_cy)))

        name_surf = self._font.render(self.name, True, WHITE_COL)
        surface.blit(name_surf, (av_cx + 26, self.rect.y + 8))
        rat_surf = self._sm_font.render(str(self.rating), True, MUTED)
        surface.blit(rat_surf, (av_cx + 26, self.rect.y + 28))

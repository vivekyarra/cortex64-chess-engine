"""Reusable button component with hover glow and pressed states."""

from __future__ import annotations

import pygame

from gui.constants import ACCENT, BG_CARD, MUTED, WHITE_COL


class Button:
    """Simple rounded button with hover, press, and disabled visuals."""

    def __init__(
        self,
        rect: pygame.Rect | tuple[int, int, int, int],
        label: str,
        color: tuple[int, int, int] | None = None,
        text_color: tuple[int, int, int] | None = None,
        font_size: int = 16,
        radius: int = 8,
        icon: str | None = None,
        disabled: bool = False,
    ) -> None:
        self.rect = pygame.Rect(rect)
        self.label = label
        self.color = color or ACCENT
        self.text_color = text_color or WHITE_COL
        self.font_size = font_size
        self.radius = radius
        self.icon = icon
        self.disabled = disabled
        self.hovered = False
        self.pressed = False
        self._font: pygame.font.Font | None = None

    def _get_font(self) -> pygame.font.Font:
        if self._font is None:
            self._font = pygame.font.SysFont("segoeui", self.font_size, bold=True)
        return self._font

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button to the given surface."""
        col = MUTED if self.disabled else self.color
        bg = BG_CARD if self.disabled else col
        if not self.disabled:
            if self.hovered:
                bg = tuple(min(255, channel + 35) for channel in bg)
            if self.pressed:
                bg = tuple(max(0, channel - 25) for channel in bg)

        if self.hovered and not self.disabled:
            glow = pygame.Surface((self.rect.w + 20, self.rect.h + 20), pygame.SRCALPHA)
            pygame.draw.rect(
                glow,
                (*self.color, 55),
                (0, 0, self.rect.w + 20, self.rect.h + 20),
                border_radius=self.radius + 6,
            )
            surface.blit(glow, (self.rect.x - 10, self.rect.y - 10))

        pygame.draw.rect(surface, bg, self.rect, border_radius=self.radius)
        border_color = self.color if not self.disabled else MUTED
        pygame.draw.rect(surface, border_color, self.rect, width=1, border_radius=self.radius)

        text = f"{self.icon}  {self.label}" if self.icon else self.label
        rendered = self._get_font().render(text, True, self.text_color if not self.disabled else WHITE_COL)
        surface.blit(rendered, rendered.get_rect(center=self.rect.center))

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Return 'clicked', 'pressed', or None."""
        if self.disabled:
            return None
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.hovered = True
                self.pressed = True
                return "pressed"
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            self.hovered = self.rect.collidepoint(event.pos)
            if was_pressed and self.hovered:
                return "clicked"
        return None

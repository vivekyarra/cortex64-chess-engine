"""Scrollable move list with optional quality badges."""

from __future__ import annotations

import pygame

from gui.constants import ACCENT, BG_CARD, BG_SURFACE, DANGER, MUTED, SUCCESS, WARNING, WHITE_COL

QUALITY_BADGE = {
    "good": ("✓", SUCCESS),
    "inaccuracy": ("△", WARNING),
    "mistake": ("✗", DANGER),
    None: ("", MUTED),
}


class MoveList:
    """Scrollable half-move list with click-to-select support."""

    def __init__(self, rect: pygame.Rect | tuple[int, int, int, int]) -> None:
        self.rect = pygame.Rect(rect)
        self.entries: list[dict] = []
        self.scroll_y = 0
        self.current = -1
        self._font: pygame.font.Font | None = None
        self._bold_font: pygame.font.Font | None = None
        self._badge_font: pygame.font.Font | None = None
        self.row_h = 28

    def set_entries(self, entries: list[dict]) -> None:
        """Replace the full entry list."""
        self.entries = list(entries)
        self.current = min(self.current, len(self.entries) - 1)

    def add_move(self, move_san: str, color: str, move_num: int, quality: str | None = None) -> None:
        """Append a new move entry and scroll to it."""
        self.entries.append(
            {
                "move_san": move_san,
                "color": color,
                "move_num": move_num,
                "quality": quality,
            }
        )
        self.current = len(self.entries) - 1
        self._scroll_to_current()

    def set_quality(self, index: int, quality: str | None) -> None:
        """Update a move quality badge."""
        if 0 <= index < len(self.entries):
            self.entries[index]["quality"] = quality

    def _scroll_to_current(self) -> None:
        visible_rows = self.rect.h // self.row_h
        if self.current >= visible_rows:
            self.scroll_y = max(0, (self.current - visible_rows + 2) * self.row_h)

    def handle_event(self, event: pygame.event.Event) -> int | None:
        """Handle scroll or click and return a clicked half-move index."""
        if event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            max_scroll = max(0, len(self.entries) * self.row_h - self.rect.h)
            self.scroll_y = max(0, min(max_scroll, self.scroll_y - event.y * self.row_h))
            return None
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            rel_y = event.pos[1] - self.rect.y + self.scroll_y
            clicked = rel_y // self.row_h
            if 0 <= clicked < len(self.entries):
                self.current = clicked
                self._scroll_to_current()
                return clicked
        return None

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the move list contents within the component rect."""
        if self._font is None:
            self._font = pygame.font.SysFont("segoeui", 15)
            self._bold_font = pygame.font.SysFont("segoeui", 15, bold=True)
            self._badge_font = pygame.font.SysFont("segoeui", 13, bold=True)

        pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=10)

        clip = surface.get_clip()
        surface.set_clip(self.rect)

        for i, entry in enumerate(self.entries):
            row_y = self.rect.y + i * self.row_h - self.scroll_y
            if row_y + self.row_h < self.rect.y or row_y > self.rect.bottom:
                continue

            bg = BG_CARD if i % 2 == 0 else BG_SURFACE
            pygame.draw.rect(surface, bg, (self.rect.x, row_y, self.rect.w, self.row_h))
            if i == self.current:
                pygame.draw.rect(surface, ACCENT, (self.rect.x, row_y, 4, self.row_h))

            if entry["color"] == "white":
                num_txt = self._bold_font.render(f"{entry['move_num']}.", True, MUTED)
                surface.blit(num_txt, (self.rect.x + 8, row_y + 6))

            x_off = 38 if entry["color"] == "white" else 92
            mov = self._font.render(entry["move_san"], True, WHITE_COL)
            surface.blit(mov, (self.rect.x + x_off, row_y + 6))

            symbol, color = QUALITY_BADGE.get(entry.get("quality"), ("", MUTED))
            if symbol:
                badge = self._badge_font.render(symbol, True, color)
                surface.blit(badge, (self.rect.right - 22, row_y + 7))

        surface.set_clip(clip)

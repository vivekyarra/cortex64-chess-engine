"""Settings screen for theme, sound, animation, and accessibility toggles."""

from __future__ import annotations

import pygame

from gui.components import Button
from gui.constants import ACCENT, ALL_SKINS, BG_CARD, BG_DARK, GOLD, MUTED, WHITE_COL, WINDOW_H, WINDOW_W
from gui.state import DEFAULT_SETTINGS, save_settings
from gui import theme


class SettingsScreen:
    """Persisted settings editor with immediate writes to settings.json."""

    def __init__(self, screen: pygame.Surface, app_state, return_screen) -> None:
        self.screen = screen
        self.app = app_state
        self.return_screen = return_screen
        self.title_font = pygame.font.SysFont("segoeui", 34, bold=True)
        self.body_font = pygame.font.SysFont("segoeui", 18)
        self.label_font = pygame.font.SysFont("segoeui", 20, bold=True)
        self.back_button = Button((48, 40, 120, 40), "Back", icon="←")
        self.reset_button = Button((WINDOW_W - 240, WINDOW_H - 82, 180, 44), "Reset to Defaults", color=GOLD, text_color=BG_DARK)

    def _set(self, key: str, value) -> None:
        self.app.settings[key] = value
        if key == "skin_index":
            theme.set_skin(int(value))
        if self.app.sound_manager is not None and key == "sound":
            self.app.sound_manager.set_enabled(bool(value))
        save_settings(self.app.settings)

    def update(self, events: list[pygame.event.Event], dt_ms: int):
        _ = dt_ms
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return self.return_screen
            if self.back_button.handle_event(event) == "clicked":
                return self.return_screen
            if self.reset_button.handle_event(event) == "clicked":
                self.app.settings.update(DEFAULT_SETTINGS)
                theme.set_skin(int(self.app.settings["skin_index"]))
                if self.app.sound_manager is not None:
                    self.app.sound_manager.set_enabled(bool(self.app.settings["sound"]))
                save_settings(self.app.settings)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)
        return self

    def _handle_click(self, pos: tuple[int, int]) -> None:
        x0 = 220
        chip_w = 140
        chip_h = 34
        start_y = 150
        row_gap = 78
        option_gap = 16

        rows = [
            ("skin_index", [skin["name"] for skin in ALL_SKINS]),
            ("sound", ["ON", "OFF"]),
            ("animations", ["Fast", "Normal", "Off"]),
            ("show_legal_moves", ["ON", "OFF"]),
            ("show_eval_bar", ["ON", "OFF"]),
            ("font_size", ["Normal", "Large"]),
        ]

        for row_index, (key, values) in enumerate(rows):
            y = start_y + row_index * row_gap
            for index, label in enumerate(values):
                rect = pygame.Rect(x0 + index * (chip_w + option_gap), y, chip_w, chip_h)
                if not rect.collidepoint(pos):
                    continue
                if key == "skin_index":
                    self._set(key, index)
                elif key in {"sound", "show_legal_moves", "show_eval_bar"}:
                    self._set(key, index == 0)
                elif key == "animations":
                    self._set(key, label.lower())
                elif key == "font_size":
                    self._set(key, label.lower())

    def _draw_row(self, surface: pygame.Surface, y: int, label: str, key: str, values: list[str]) -> None:
        text = self.label_font.render(label, True, WHITE_COL)
        surface.blit(text, (70, y + 2))
        x = 220
        chip_w = 140
        option_gap = 16
        current = self.app.settings.get(key)
        for index, value in enumerate(values):
            rect = pygame.Rect(x + index * (chip_w + option_gap), y, chip_w, 34)
            selected = (
                (key == "skin_index" and current == index)
                or (key in {"sound", "show_legal_moves", "show_eval_bar"} and bool(current) == (index == 0))
                or (str(current).lower() == value.lower())
            )
            fill = ACCENT if selected else BG_CARD
            pygame.draw.rect(surface, fill, rect, border_radius=16)
            pygame.draw.rect(surface, ACCENT if selected else MUTED, rect, width=1, border_radius=16)
            color = WHITE_COL if selected else MUTED
            txt = self.body_font.render(value, True, color)
            surface.blit(txt, txt.get_rect(center=rect.center))

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DARK)
        panel = pygame.Rect(32, 24, WINDOW_W - 64, WINDOW_H - 48)
        pygame.draw.rect(surface, BG_CARD, panel, border_radius=24)
        pygame.draw.rect(surface, ACCENT, panel, width=2, border_radius=24)

        self.back_button.draw(surface)
        title = self.title_font.render("Settings", True, WHITE_COL)
        surface.blit(title, (70, 90))

        self._draw_row(surface, 150, "Board Skin", "skin_index", [skin["name"] for skin in ALL_SKINS])
        self._draw_row(surface, 228, "Sound", "sound", ["ON", "OFF"])
        self._draw_row(surface, 306, "Animations", "animations", ["Fast", "Normal", "Off"])
        self._draw_row(surface, 384, "Show legal moves", "show_legal_moves", ["ON", "OFF"])
        self._draw_row(surface, 462, "Show eval bar", "show_eval_bar", ["ON", "OFF"])
        self._draw_row(surface, 540, "Font size", "font_size", ["Normal", "Large"])

        note = self.body_font.render("All changes are saved immediately to gui/data/settings.json", True, MUTED)
        surface.blit(note, (70, WINDOW_H - 108))
        self.reset_button.draw(surface)

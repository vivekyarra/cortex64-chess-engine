"""Game setup modal screen for AI and local human play."""

from __future__ import annotations

import math

import pygame

from gui.components import Button
from gui.constants import ACCENT, ANIM_MODAL_MS, BG_CARD, BG_DARK, GOLD, MUTED, WHITE_COL, WINDOW_H, WINDOW_W
from gui.state import DIFFICULTY_PRESETS, GameConfig, TIME_CONTROL_PRESETS, TimeControl, choose_side, save_profile


class GameSetupScreen:
    """Slide-in modal overlay used to configure a new game."""

    def __init__(self, screen: pygame.Surface, app_state, return_screen, mode: str = "ai") -> None:
        self.screen = screen
        self.app = app_state
        self.return_screen = return_screen
        self.mode = mode
        self.elapsed_ms = 0
        self.side_choice = str(self.app.profile.get("preferred_side", "white")).lower()
        self.difficulty_index = 2
        self.time_choice = 0
        self.custom_minutes = "10"
        self.custom_increment = "0"
        self.active_input = "username"
        self.username = str(self.app.profile.get("username", "Soprano"))

        self.title_font = pygame.font.SysFont("segoeui", 30, bold=True)
        self.label_font = pygame.font.SysFont("segoeui", 18, bold=True)
        self.body_font = pygame.font.SysFont("segoeui", 16)
        self.start_button = Button((0, 0, 100, 48), "START GAME", font_size=18, radius=12)

    def _modal_rect(self) -> pygame.Rect:
        width = 640
        height = 520
        progress = min(1.0, self.elapsed_ms / ANIM_MODAL_MS)
        eased = 1.0 - (1.0 - progress) ** 3
        base_y = WINDOW_H // 2 - height // 2
        y = int(WINDOW_H + 40 - (WINDOW_H // 2 + height // 2 + 40) * eased)
        return pygame.Rect(WINDOW_W // 2 - width // 2, max(40, y), width, height)

    def _current_time_control(self):
        label, value = TIME_CONTROL_PRESETS[self.time_choice]
        if value == "custom":
            return TimeControl(minutes=max(0, int(self.custom_minutes or "0")), increment=max(0, int(self.custom_increment or "0")))
        return value

    def _build_config(self) -> GameConfig:
        side = choose_side(self.side_choice)
        config = GameConfig.from_preset(
            self.difficulty_index,
            mode=self.mode,
            human=side,
            username=self.username.strip() or "Soprano",
            settings=self.app.settings,
            time_control=self._current_time_control(),
            model=self.app.model_path,
        )
        return config

    def _handle_text_input(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_TAB:
            self.active_input = "minutes" if self.active_input == "username" else "username"
            return
        if event.key == pygame.K_BACKSPACE:
            if self.active_input == "username":
                self.username = self.username[:-1]
            elif self.active_input == "minutes":
                self.custom_minutes = self.custom_minutes[:-1]
            elif self.active_input == "increment":
                self.custom_increment = self.custom_increment[:-1]
            return
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return
        if self.active_input == "username":
            if event.unicode and event.unicode.isprintable() and len(self.username) < 16:
                self.username += event.unicode
            return
        if event.unicode.isdigit():
            if self.active_input == "minutes" and len(self.custom_minutes) < 2:
                self.custom_minutes += event.unicode
            if self.active_input == "increment" and len(self.custom_increment) < 2:
                self.custom_increment += event.unicode

    def update(self, events: list[pygame.event.Event], dt_ms: int):
        self.elapsed_ms = min(self.elapsed_ms + dt_ms, ANIM_MODAL_MS)
        modal = self._modal_rect()
        start_rect = pygame.Rect(modal.x + 44, modal.bottom - 70, modal.w - 88, 48)
        self.start_button.rect = start_rect

        side_rects = {
            "white": pygame.Rect(modal.x + 44, modal.y + 92, 164, 74),
            "black": pygame.Rect(modal.x + 238, modal.y + 92, 164, 74),
            "random": pygame.Rect(modal.x + 432, modal.y + 92, 164, 74),
        }
        username_rect = pygame.Rect(modal.x + 44, modal.y + 352, modal.w - 88, 42)
        custom_minutes_rect = pygame.Rect(modal.x + 372, modal.y + 330, 86, 34)
        custom_increment_rect = pygame.Rect(modal.x + 474, modal.y + 330, 86, 34)

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return self.return_screen
                self._handle_text_input(event)

            if self.start_button.handle_event(event) == "clicked":
                config = self._build_config()
                self.app.profile["username"] = config.username
                self.app.profile["preferred_side"] = config.human
                save_profile(self.app.profile)
                from gui.game import GameScreen

                return GameScreen(self.screen, self.app, config)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not modal.collidepoint(event.pos):
                    return self.return_screen
                if username_rect.collidepoint(event.pos):
                    self.active_input = "username"
                elif TIME_CONTROL_PRESETS[self.time_choice][1] == "custom" and custom_minutes_rect.collidepoint(event.pos):
                    self.active_input = "minutes"
                elif TIME_CONTROL_PRESETS[self.time_choice][1] == "custom" and custom_increment_rect.collidepoint(event.pos):
                    self.active_input = "increment"

                for key, rect in side_rects.items():
                    if rect.collidepoint(event.pos):
                        self.side_choice = key

                slider_x0 = modal.x + 54
                slider_x1 = modal.right - 54
                slider_y = modal.y + 220
                slider_rect = pygame.Rect(slider_x0, slider_y - 12, slider_x1 - slider_x0, 24)
                if slider_rect.collidepoint(event.pos):
                    ratio = (event.pos[0] - slider_x0) / max(1, slider_x1 - slider_x0)
                    self.difficulty_index = max(0, min(4, int(round(ratio * 4))))

                for index, (_label, value) in enumerate(TIME_CONTROL_PRESETS):
                    pill = pygame.Rect(modal.x + 44 + index * 112, modal.y + 286, 98, 34)
                    if pill.collidepoint(event.pos):
                        self.time_choice = index
                        if value != "custom":
                            self.active_input = "username"
        return self

    def draw(self, surface: pygame.Surface) -> None:
        self.return_screen.draw(surface)
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((*BG_DARK, 180))
        surface.blit(overlay, (0, 0))

        modal = self._modal_rect()
        pygame.draw.rect(surface, BG_CARD, modal, border_radius=24)
        pygame.draw.rect(surface, ACCENT, modal, width=2, border_radius=24)

        title = self.title_font.render("Game Setup" if self.mode == "ai" else "Local Two-Player Setup", True, WHITE_COL)
        surface.blit(title, (modal.x + 32, modal.y + 24))
        subtitle = self.body_font.render(
            "Choose your side, difficulty, and time control.",
            True,
            MUTED,
        )
        surface.blit(subtitle, (modal.x + 34, modal.y + 58))

        side_rects = {
            "white": pygame.Rect(modal.x + 44, modal.y + 92, 164, 74),
            "black": pygame.Rect(modal.x + 238, modal.y + 92, 164, 74),
            "random": pygame.Rect(modal.x + 432, modal.y + 92, 164, 74),
        }
        labels = {"white": "White", "black": "Black", "random": "Random"}
        icons = {"white": "♔", "black": "♚", "random": "?"}
        for key, rect in side_rects.items():
            selected = self.side_choice == key
            fill = (38, 38, 66) if selected else (32, 32, 54)
            pygame.draw.rect(surface, fill, rect, border_radius=18)
            pygame.draw.rect(surface, ACCENT if selected else MUTED, rect, width=2, border_radius=18)
            icon = self.title_font.render(icons[key], True, GOLD if selected else WHITE_COL)
            label = self.label_font.render(labels[key], True, WHITE_COL)
            surface.blit(icon, (rect.x + 16, rect.y + 16))
            surface.blit(label, (rect.x + 64, rect.y + 28))

        diff_title = self.label_font.render("Difficulty", True, WHITE_COL)
        surface.blit(diff_title, (modal.x + 44, modal.y + 186))
        slider_x0 = modal.x + 54
        slider_x1 = modal.right - 54
        slider_y = modal.y + 220
        pygame.draw.line(surface, MUTED, (slider_x0, slider_y), (slider_x1, slider_y), 3)
        notch_gap = (slider_x1 - slider_x0) / 4
        knob_x = slider_x0 + notch_gap * self.difficulty_index
        for index, preset in enumerate(DIFFICULTY_PRESETS):
            cx = slider_x0 + notch_gap * index
            pygame.draw.circle(surface, ACCENT if index <= self.difficulty_index else MUTED, (int(cx), slider_y), 8)
            label = self.body_font.render(preset["label"], True, GOLD if index == self.difficulty_index else MUTED)
            surface.blit(label, label.get_rect(center=(int(cx), slider_y + 30)))

        current = DIFFICULTY_PRESETS[self.difficulty_index]
        details = self.body_font.render(
            f"Depth {current['depth']}  ·  {current['move_time']:.1f}s move budget",
            True,
            WHITE_COL,
        )
        surface.blit(details, (modal.x + 44, modal.y + 250))

        tc_title = self.label_font.render("Time Control", True, WHITE_COL)
        surface.blit(tc_title, (modal.x + 44, modal.y + 286 - 28))
        for index, (label, value) in enumerate(TIME_CONTROL_PRESETS):
            pill = pygame.Rect(modal.x + 44 + index * 112, modal.y + 286, 98, 34)
            selected = self.time_choice == index
            pygame.draw.rect(surface, ACCENT if selected else (32, 32, 54), pill, border_radius=17)
            pygame.draw.rect(surface, ACCENT if selected else MUTED, pill, width=1, border_radius=17)
            txt = self.body_font.render(label, True, WHITE_COL if selected else MUTED)
            surface.blit(txt, txt.get_rect(center=pill.center))

        if TIME_CONTROL_PRESETS[self.time_choice][1] == "custom":
            minutes_rect = pygame.Rect(modal.x + 372, modal.y + 330, 86, 34)
            increment_rect = pygame.Rect(modal.x + 474, modal.y + 330, 86, 34)
            for rect, value, active, suffix in (
                (minutes_rect, self.custom_minutes or "0", self.active_input == "minutes", "min"),
                (increment_rect, self.custom_increment or "0", self.active_input == "increment", "inc"),
            ):
                pygame.draw.rect(surface, (38, 38, 66), rect, border_radius=12)
                pygame.draw.rect(surface, ACCENT if active else MUTED, rect, width=1, border_radius=12)
                txt = self.body_font.render(f"{value} {suffix}", True, WHITE_COL)
                surface.blit(txt, txt.get_rect(center=rect.center))

        username_title = self.label_font.render("Username", True, WHITE_COL)
        surface.blit(username_title, (modal.x + 44, modal.y + 324))
        username_rect = pygame.Rect(modal.x + 44, modal.y + 352, modal.w - 88, 42)
        pygame.draw.rect(surface, (32, 32, 54), username_rect, border_radius=14)
        pygame.draw.rect(surface, ACCENT if self.active_input == "username" else MUTED, username_rect, width=1, border_radius=14)
        caret = "|" if self.active_input == "username" and int(pygame.time.get_ticks() / 400) % 2 == 0 else ""
        name_txt = self.body_font.render((self.username or "Soprano") + caret, True, WHITE_COL)
        surface.blit(name_txt, (username_rect.x + 12, username_rect.y + 10))

        note = self.body_font.render("ESC or click outside the modal to cancel.", True, MUTED)
        surface.blit(note, (modal.x + 44, modal.bottom - 96))

        self.start_button.draw(surface)

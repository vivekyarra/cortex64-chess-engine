"""Main menu screen for the Cortex64 v2 interface."""

from __future__ import annotations

import math

import pygame

from gui.components import Button
from gui.constants import ACCENT, BG_CARD, BG_DARK, GOLD, MUTED, WHITE_COL, WINDOW_H, WINDOW_W
from gui.ui_utils import blur_surface, load_piece_images


class MainMenuScreen:
    """Animated main menu with primary navigation buttons."""

    def __init__(self, screen: pygame.Surface, app_state) -> None:
        self.screen = screen
        self.app = app_state
        self.elapsed_ms = 0.0
        self.show_profile = False

        self.title_font = pygame.font.SysFont("segoeui", 48, bold=True)
        self.subtitle_font = pygame.font.SysFont("segoeui", 24, bold=True)
        self.body_font = pygame.font.SysFont("segoeui", 18)
        self.footer_font = pygame.font.SysFont("georgia", 14, italic=True)

        button_w = 280
        button_h = 52
        start_y = 270
        gap = 14
        center_x = WINDOW_W // 2 - button_w // 2
        self.buttons = {
            "play_ai": Button((center_x, start_y, button_w, button_h), "Play vs AI", icon="▶"),
            "play_human": Button((center_x, start_y + (button_h + gap), button_w, button_h), "Play vs Human", icon="♟"),
            "analysis": Button((center_x, start_y + 2 * (button_h + gap), button_w, button_h), "Analysis Mode", icon="📖"),
            "settings": Button((center_x, start_y + 3 * (button_h + gap), button_w, button_h), "Settings", icon="⚙"),
            "profile": Button((center_x, start_y + 4 * (button_h + gap), button_w, button_h), "Profile & Stats", icon="🏆"),
        }

        piece_images = load_piece_images(96)
        self.silhouettes = [
            {"surf": self._tint(piece_images[5]), "pos": (90, 120), "amp": 14, "phase": 0.0, "speed": 0.0015},
            {"surf": self._tint(piece_images[-2]), "pos": (980, 150), "amp": 18, "phase": 0.8, "speed": 0.0012},
            {"surf": self._tint(piece_images[6]), "pos": (150, 520), "amp": 12, "phase": 1.6, "speed": 0.0018},
            {"surf": self._tint(piece_images[-5]), "pos": (1010, 500), "amp": 16, "phase": 2.4, "speed": 0.0014},
        ]

    def _tint(self, surface: pygame.Surface) -> pygame.Surface:
        tinted = pygame.transform.smoothscale(surface, (120, 120)).copy()
        tinted.fill((110, 110, 140, 180), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted

    def update(self, events: list[pygame.event.Event], dt_ms: int):
        self.elapsed_ms += dt_ms
        for event in events:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                continue
            if self.show_profile and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                card = pygame.Rect(WINDOW_W // 2 - 200, WINDOW_H // 2 - 150, 400, 300)
                if not card.collidepoint(event.pos):
                    self.show_profile = False

            for key, button in self.buttons.items():
                action = button.handle_event(event)
                if action != "clicked":
                    continue
                if key == "play_ai":
                    from gui.screens.game_setup import GameSetupScreen

                    return GameSetupScreen(self.screen, self.app, self, mode="ai")
                if key == "play_human":
                    from gui.screens.game_setup import GameSetupScreen

                    return GameSetupScreen(self.screen, self.app, self, mode="human")
                if key == "analysis":
                    from gui.screens.analysis import AnalysisScreen

                    return AnalysisScreen(self.screen, self.app, self.app.last_summary, self)
                if key == "settings":
                    from gui.screens.settings import SettingsScreen

                    return SettingsScreen(self.screen, self.app, self)
                if key == "profile":
                    self.show_profile = not self.show_profile
        return self

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DARK)
        gradient = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        pygame.draw.circle(gradient, (*ACCENT, 28), (WINDOW_W // 2, 130), 240)
        pygame.draw.circle(gradient, (*GOLD, 16), (WINDOW_W // 2 + 120, 560), 180)
        surface.blit(gradient, (0, 0))

        for silhouette in self.silhouettes:
            pulse = 34 + int((math.sin(self.elapsed_ms * silhouette["speed"] + silhouette["phase"]) + 1) * 18)
            offset = math.sin(self.elapsed_ms * silhouette["speed"] * 1.4 + silhouette["phase"]) * silhouette["amp"]
            piece = silhouette["surf"].copy()
            piece.set_alpha(pulse)
            surface.blit(piece, (silhouette["pos"][0], int(silhouette["pos"][1] + offset)))

        glow = self.title_font.render("⬡ CORTEX64", True, ACCENT)
        glow_surf = pygame.Surface((glow.get_width() + 32, glow.get_height() + 24), pygame.SRCALPHA)
        glow_surf.blit(glow, (16, 12))
        glow_surf = blur_surface(glow_surf, 0.2)
        glow_surf.set_alpha(160)
        surface.blit(glow_surf, (WINDOW_W // 2 - glow_surf.get_width() // 2, 108))
        title = self.title_font.render("⬡ CORTEX64", True, ACCENT)
        surface.blit(title, title.get_rect(center=(WINDOW_W // 2, 146)))

        subtitle = self.subtitle_font.render("v2.0 — AI Chess Engine", True, GOLD)
        surface.blit(subtitle, subtitle.get_rect(center=(WINDOW_W // 2, 190)))

        for button in self.buttons.values():
            button.draw(surface)

        footer = self.footer_font.render(
            "Built from scratch in Python · No Stockfish · No External Chess Libraries",
            True,
            MUTED,
        )
        surface.blit(footer, footer.get_rect(center=(WINDOW_W // 2, WINDOW_H - 28)))

        if self.show_profile:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            surface.blit(overlay, (0, 0))
            card = pygame.Rect(WINDOW_W // 2 - 200, WINDOW_H // 2 - 150, 400, 300)
            pygame.draw.rect(surface, BG_CARD, card, border_radius=20)
            pygame.draw.rect(surface, ACCENT, card, width=2, border_radius=20)
            header = self.subtitle_font.render(self.app.profile.get("username", "Soprano"), True, WHITE_COL)
            surface.blit(header, (card.x + 26, card.y + 24))

            stats = [
                f"Games Played: {self.app.profile.get('games_played', 0)}",
                f"Wins: {self.app.profile.get('wins', 0)}",
                f"Losses: {self.app.profile.get('losses', 0)}",
                f"Draws: {self.app.profile.get('draws', 0)}",
                f"Preferred Side: {str(self.app.profile.get('preferred_side', 'white')).title()}",
            ]
            for index, line in enumerate(stats):
                text = self.body_font.render(line, True, WHITE_COL if index == 0 else MUTED)
                surface.blit(text, (card.x + 28, card.y + 84 + index * 34))

            detail = self.body_font.render("Profile data stays backward compatible with v1.", True, MUTED)
            surface.blit(detail, (card.x + 28, card.bottom - 46))

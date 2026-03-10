"""Post-game results screen with summary stats and follow-up actions."""

from __future__ import annotations

from dataclasses import replace

import pygame

from gui.components import Button
from gui.constants import ACCENT, BG_CARD, BG_DARK, DANGER, GOLD, MUTED, SUCCESS, WHITE_COL, WINDOW_H, WINDOW_W
from gui.state import GameConfig, export_summary_pgn
from gui.ui_utils import blur_surface


class ResultsScreen:
    """Displays the finished game result and navigation options."""

    def __init__(self, screen: pygame.Surface, app_state, summary) -> None:
        self.screen = screen
        self.app = app_state
        self.summary = summary
        self.background = blur_surface(summary.background or pygame.Surface((WINDOW_W, WINDOW_H)))
        self.title_font = pygame.font.SysFont("segoeui", 42, bold=True)
        self.subtitle_font = pygame.font.SysFont("segoeui", 20, bold=True)
        self.body_font = pygame.font.SysFont("segoeui", 18)
        self.small_font = pygame.font.SysFont("segoeui", 15)
        self.toast = ""
        self.toast_until = 0

        btn_w = 180
        btn_h = 42
        gap = 12
        left = WINDOW_W // 2 - (btn_w * 3 + gap * 2) // 2
        top = 560
        self.buttons = {
            "rematch": Button((left, top, btn_w, btn_h), "Rematch", icon="🔄"),
            "switch": Button((left + btn_w + gap, top, btn_w, btn_h), "Switch Sides", icon="↕"),
            "analyze": Button((left + 2 * (btn_w + gap), top, btn_w, btn_h), "Analyze Game", icon="🔍"),
            "export": Button((left + 48, top + btn_h + gap, btn_w, btn_h), "Export PGN", icon="📤"),
            "menu": Button((left + 48 + btn_w + gap, top + btn_h + gap, btn_w, btn_h), "Main Menu", icon="🏠"),
        }

    def _player_records(self):
        if self.summary.config.mode == "human":
            return list(self.summary.move_records)
        return [record for record in self.summary.move_records if record.color == self.summary.config.human]

    def _banner(self):
        if self.summary.config.mode == "human" and self.summary.winner in {"white", "black"}:
            return f"{self.summary.winner.upper()} WINS", SUCCESS if self.summary.winner == "white" else ACCENT
        if self.summary.result == "win":
            return "YOU WIN", SUCCESS
        if self.summary.result == "loss":
            return "YOU LOSE", DANGER
        return "DRAW", GOLD

    def _stats(self):
        records = self._player_records()
        total = max(1, len(records))
        best = sum(1 for record in records if record.quality == "good")
        inaccuracies = sum(1 for record in records if record.quality == "inaccuracy")
        mistakes = sum(1 for record in records if record.quality == "mistake")
        accuracy = best / total * 100.0 if records else 0.0
        acpl = sum(record.delta_cp for record in records) / total if records else 0.0
        full_moves = (len(self.summary.move_records) + 1) // 2
        secs = self.summary.total_time_ms // 1000
        mins, seconds = divmod(secs, 60)
        return accuracy, best, inaccuracies, mistakes, acpl, f"{full_moves} moves, {mins}:{seconds:02d}"

    def _rematch_config(self, switch_sides: bool) -> GameConfig:
        human = self.summary.config.human
        if switch_sides:
            human = "black" if human == "white" else "white"
        return replace(
            self.summary.config,
            human=human,
            username=self.app.profile.get("username", self.summary.config.username),
            settings=dict(self.app.settings),
        )

    def update(self, events: list[pygame.event.Event], dt_ms: int):
        _ = dt_ms
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    from gui.game import GameScreen

                    return GameScreen(self.screen, self.app, self._rematch_config(False))
                if event.key == pygame.K_ESCAPE:
                    from gui.screens.main_menu import MainMenuScreen

                    return MainMenuScreen(self.screen, self.app)

            for key, button in self.buttons.items():
                action = button.handle_event(event)
                if action != "clicked":
                    continue
                if key == "rematch":
                    from gui.game import GameScreen

                    return GameScreen(self.screen, self.app, self._rematch_config(False))
                if key == "switch":
                    from gui.game import GameScreen

                    return GameScreen(self.screen, self.app, self._rematch_config(True))
                if key == "analyze":
                    from gui.screens.analysis import AnalysisScreen

                    return AnalysisScreen(self.screen, self.app, self.summary, self)
                if key == "export":
                    try:
                        path = export_summary_pgn(self.summary)
                        self.summary.export_path = path
                        self.toast = f"Saved PGN to {path}"
                        self.toast_until = pygame.time.get_ticks() + 2600
                    except Exception as exc:
                        self.toast = f"Export failed: {exc}"
                        self.toast_until = pygame.time.get_ticks() + 2600
                if key == "menu":
                    from gui.screens.main_menu import MainMenuScreen

                    return MainMenuScreen(self.screen, self.app)
        return self

    def draw(self, surface: pygame.Surface) -> None:
        surface.blit(self.background, (0, 0))
        overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        overlay.fill((*BG_DARK, 168))
        surface.blit(overlay, (0, 0))

        banner_text, banner_color = self._banner()
        banner = self.title_font.render(banner_text, True, banner_color)
        surface.blit(banner, banner.get_rect(center=(WINDOW_W // 2, 110)))

        reason = self.subtitle_font.render(f"by {self.summary.reason}", True, WHITE_COL)
        surface.blit(reason, reason.get_rect(center=(WINDOW_W // 2, 150)))

        stats_card = pygame.Rect(WINDOW_W // 2 - 250, 200, 500, 280)
        pygame.draw.rect(surface, BG_CARD, stats_card, border_radius=24)
        pygame.draw.rect(surface, ACCENT, stats_card, width=2, border_radius=24)

        accuracy, best, inaccuracies, mistakes, acpl, length = self._stats()
        rows = [
            ("Accuracy %", f"{accuracy:.1f}%"),
            ("Best moves", str(best)),
            ("Inaccuracies", str(inaccuracies)),
            ("Mistakes", str(mistakes)),
            ("Average centipawn loss", f"{acpl:.1f}"),
            ("Game length", length),
        ]
        for index, (label, value) in enumerate(rows):
            y = stats_card.y + 34 + index * 36
            left = self.body_font.render(label, True, MUTED)
            right = self.body_font.render(value, True, WHITE_COL)
            surface.blit(left, (stats_card.x + 28, y))
            surface.blit(right, (stats_card.right - 28 - right.get_width(), y))

        for button in self.buttons.values():
            button.draw(surface)

        hotkeys = self.small_font.render("Hotkeys: R = Rematch, ESC = Main Menu", True, MUTED)
        surface.blit(hotkeys, hotkeys.get_rect(center=(WINDOW_W // 2, WINDOW_H - 42)))

        if self.toast and pygame.time.get_ticks() < self.toast_until:
            toast = self.small_font.render(self.toast, True, GOLD)
            surface.blit(toast, toast.get_rect(center=(WINDOW_W // 2, 520)))

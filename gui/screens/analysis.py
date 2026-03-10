"""Analysis screen with move navigation and evaluation graph."""

from __future__ import annotations

import pygame

from engine.board import Board, square_name
from gui.components import Button, EvalGraph, MoveList
from gui.constants import ACCENT, BG_CARD, BG_DARK, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE, GOLD, MUTED, SQUARE_SIZE, WHITE_COL
from gui import theme
from gui.ui_utils import draw_arrow, load_piece_images


class AnalysisScreen:
    """Move-by-move analysis viewer for completed games."""

    def __init__(self, screen: pygame.Surface, app_state, summary, return_screen) -> None:
        self.screen = screen
        self.app = app_state
        self.summary = summary
        self.return_screen = return_screen
        self.title_font = pygame.font.SysFont("segoeui", 24, bold=True)
        self.body_font = pygame.font.SysFont("segoeui", 16)
        self.back_button = Button((28, 28, 110, 38), "Back", icon="←")
        self.prev_button = Button((BOARD_OFFSET_X + 180, BOARD_OFFSET_Y + BOARD_SIZE + 24, 120, 38), "Prev")
        self.next_button = Button((BOARD_OFFSET_X + 340, BOARD_OFFSET_Y + BOARD_SIZE + 24, 120, 38), "Next")
        self.graph = EvalGraph((BOARD_OFFSET_X, BOARD_OFFSET_Y + BOARD_SIZE + 76, 640, 88))
        self.move_list = MoveList((890, 120, 330, 520))
        self.board_rect = pygame.Rect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE, BOARD_SIZE)
        self.piece_images = load_piece_images(SQUARE_SIZE)
        self.current_ply = 0
        self.boards = [Board()]

        if self.summary is not None:
            board = Board()
            entries = []
            for record in self.summary.move_records:
                board.push(record.move)
                self.boards.append(board.copy())
                entries.append(
                    {
                        "move_san": record.san,
                        "color": record.color,
                        "move_num": record.move_num,
                        "quality": record.quality,
                    }
                )
            self.move_list.set_entries(entries)
            self.graph.set_evals(self.summary.evals)

    def _board_flipped(self) -> bool:
        if self.summary is None:
            return False
        return self.summary.config.human == "black"

    def _square_rect(self, square: int) -> pygame.Rect:
        row, col = divmod(square, 8)
        if self._board_flipped():
            row, col = 7 - row, 7 - col
        return pygame.Rect(
            self.board_rect.x + col * SQUARE_SIZE,
            self.board_rect.y + row * SQUARE_SIZE,
            SQUARE_SIZE,
            SQUARE_SIZE,
        )

    def _set_ply(self, ply: int) -> None:
        self.current_ply = max(0, min(len(self.boards) - 1, ply))
        self.move_list.current = self.current_ply - 1

    def _parse_uci(self, uci: str):
        if len(uci) < 4:
            return None
        files = "abcdefgh"
        try:
            from_sq = (8 - int(uci[1])) * 8 + files.index(uci[0])
            to_sq = (8 - int(uci[3])) * 8 + files.index(uci[2])
        except (ValueError, IndexError):
            return None
        return from_sq, to_sq

    def update(self, events: list[pygame.event.Event], dt_ms: int):
        _ = dt_ms
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return self.return_screen
                if event.key == pygame.K_LEFT:
                    self._set_ply(self.current_ply - 1)
                if event.key == pygame.K_RIGHT:
                    self._set_ply(self.current_ply + 1)

            if self.back_button.handle_event(event) == "clicked":
                return self.return_screen
            if self.prev_button.handle_event(event) == "clicked":
                self._set_ply(self.current_ply - 1)
            if self.next_button.handle_event(event) == "clicked":
                self._set_ply(self.current_ply + 1)

            move_index = self.move_list.handle_event(event)
            if move_index is not None:
                self._set_ply(move_index + 1)

            graph_index = self.graph.handle_event(event)
            if graph_index is not None:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._set_ply(graph_index + 1)
                elif event.type == pygame.MOUSEMOTION:
                    self.move_list.current = graph_index
        return self

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DARK)
        self.back_button.draw(surface)

        pygame.draw.rect(surface, BG_CARD, (0, 0, 180, 800))
        pygame.draw.rect(surface, BG_CARD, (860, 0, 420, 800))

        title = self.title_font.render("Analysis Mode", True, WHITE_COL)
        surface.blit(title, (890, 40))
        if self.summary is not None:
            subtitle = self.body_font.render(self.summary.reason, True, MUTED)
            surface.blit(subtitle, (890, 72))

        skin = theme.get_skin()
        pygame.draw.rect(surface, BG_CARD, self.board_rect.inflate(16, 16), border_radius=18)
        for row in range(8):
            for col in range(8):
                color = skin["light"] if (row + col) % 2 == 0 else skin["dark"]
                display_row, display_col = (7 - row, 7 - col) if self._board_flipped() else (row, col)
                rect = pygame.Rect(
                    self.board_rect.x + display_col * SQUARE_SIZE,
                    self.board_rect.y + display_row * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                )
                pygame.draw.rect(surface, color, rect)

        board = self.boards[self.current_ply]
        if self.current_ply > 0 and self.summary is not None:
            move = self.summary.move_records[self.current_ply - 1].move
            for square in (move.from_sq, move.to_sq):
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill((*ACCENT, 80))
                surface.blit(overlay, self._square_rect(square).topleft)

        for row in range(8):
            for col in range(8):
                piece = int(board.squares[row, col])
                if piece == 0:
                    continue
                square = row * 8 + col
                rect = self._square_rect(square)
                image = self.piece_images[piece]
                surface.blit(image, image.get_rect(center=rect.center))

        files = "hgfedcba" if self._board_flipped() else "abcdefgh"
        ranks = [str(index + 1) for index in range(8)] if self._board_flipped() else [str(8 - index) for index in range(8)]
        for col in range(8):
            label = self.body_font.render(files[col], True, MUTED)
            surface.blit(label, (self.board_rect.x + col * SQUARE_SIZE + SQUARE_SIZE - 16, self.board_rect.bottom - 20))
        for row in range(8):
            label = self.body_font.render(ranks[row], True, MUTED)
            surface.blit(label, (self.board_rect.x + 6, self.board_rect.y + row * SQUARE_SIZE + 6))

        if self.summary is not None and self.current_ply < len(self.summary.move_records):
            best = self.summary.move_records[self.current_ply].best_move_uci
            parsed = self._parse_uci(best)
            if parsed is not None:
                start = self._square_rect(parsed[0]).center
                end = self._square_rect(parsed[1]).center
                draw_arrow(surface, start, end, MUTED, width=8, alpha=120)

        self.graph.draw(surface)
        self.prev_button.draw(surface)
        self.next_button.draw(surface)
        self.move_list.draw(surface)

        if self.summary is not None and self.current_ply > 0:
            record = self.summary.move_records[self.current_ply - 1]
            caption = self.body_font.render(
                f"{record.move_num}. {record.san}  ·  {record.quality or 'unrated'}",
                True,
                GOLD if record.quality == "good" else WHITE_COL,
            )
            surface.blit(caption, (BOARD_OFFSET_X, BOARD_OFFSET_Y + BOARD_SIZE + 6))

        left_panel = pygame.Rect(18, 98, 144, 230)
        pygame.draw.rect(surface, BG_CARD, left_panel, border_radius=18)
        pygame.draw.rect(surface, ACCENT, left_panel, width=1, border_radius=18)
        info = [
            "Current position",
            f"Ply: {self.current_ply}/{max(0, len(self.boards) - 1)}",
            f"Move: {square_name(self.summary.move_records[self.current_ply - 1].move.to_sq) if self.summary and self.current_ply else 'start'}",
            f"Completed: {self.summary.completed_at.split('T')[0] if self.summary else 'N/A'}",
        ]
        for idx, line in enumerate(info):
            txt = self.body_font.render(line, True, WHITE_COL if idx == 0 else MUTED)
            surface.blit(txt, (left_panel.x + 12, left_panel.y + 18 + idx * 34))

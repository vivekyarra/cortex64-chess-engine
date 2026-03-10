"""Main gameplay screen for the Cortex64 v2 interface."""

from __future__ import annotations

import threading
from dataclasses import replace

import pygame

from engine.board import BISHOP, BLACK, Board, KNIGHT, Move, QUEEN, ROOK, WHITE, piece_color, square_name
from engine.move_generator import generate_legal_moves, is_checkmate, is_in_check, is_stalemate, legal_moves_from
from gui import theme
from gui.animation import AnimationManager
from gui.components import Button, ChessClock, EvalBar, MoveList, PlayerCard
from gui.constants import (
    ACCENT,
    ANIM_FADE_MS,
    ANIM_PIECE_MS,
    ANIM_PULSE_MS,
    ANIM_SHAKE_MS,
    BG_CARD,
    BG_DARK,
    BG_SURFACE,
    BOARD_OFFSET_X,
    BOARD_OFFSET_Y,
    BOARD_SIZE,
    DANGER,
    GOLD,
    MUTED,
    SIDEBAR_LEFT_W,
    SQUARE_SIZE,
    SUCCESS,
    WARNING,
    WHITE_COL,
)
from gui.search import MaterialEvaluator, classify_quality, create_evaluator, evaluate_move_delta, search_best_move_with_budget
from gui.state import GameConfig, GameSummary, MoveRecord, save_profile
from gui.ui_utils import draw_arrow, fit_text, load_piece_images

PIECE_LETTERS = {1: "", 2: "N", 3: "B", 4: "R", 5: "Q", 6: "K"}


class GameScreen:
    """Interactive chess game view with sidebars, animations, and AI."""

    def __init__(self, screen: pygame.Surface, app_state, config: GameConfig) -> None:
        self.screen = screen
        self.app = app_state
        self.config = replace(config, settings=dict(config.settings or self.app.settings))
        self.board = Board()
        self.human_color = WHITE if self.config.human == "white" else BLACK
        self.ai_color = -self.human_color if self.config.has_ai else None
        self.animation = AnimationManager()
        self.piece_images = load_piece_images(SQUARE_SIZE)
        self.evaluator = create_evaluator(self.config.model or self.app.model_path)
        self.quality_evaluator = MaterialEvaluator(self.config.model or self.app.model_path or "")
        self.profile_name = self.config.username or "Soprano"

        self.move_history: list[Move] = []
        self.move_records: list[MoveRecord] = []
        self.eval_history: list[float] = []
        self.selected_sq: int | None = None
        self.selected_moves: list[Move] = []
        self.last_move: Move | None = None
        self.drag_sq: int | None = None
        self.drag_piece: int | None = None
        self.drag_origin: tuple[int, int] | None = None
        self.drag_pos: tuple[int, int] | None = None
        self.dragging = False
        self.pending_promotion: tuple[int, int, list[Move]] | None = None
        self.hint_move: Move | None = None
        self.hint_explanation = ""
        self.hint_until = 0
        self.hint_fade_end = 0
        self.hidden_anim_to_sq: int | None = None

        self.ai_thinking = False
        self.ai_result = None
        self.ai_lock = threading.Lock()
        self.ai_thread: threading.Thread | None = None
        self.ai_last_time = 0.0
        self.ai_last_nodes = 0
        self.ai_last_depth = 0
        self.ai_last_eval_cp = 0.0
        self.ai_stats_expanded = False

        self.total_elapsed_ms = 0
        self.status_text = ""
        self.banner_text = ""
        self.banner_until = 0
        self.confirm_resign = False
        self.results_summary: GameSummary | None = None
        self.results_pending = False
        self.results_ready = False

        self.font_scale = 1.15 if self.config.settings.get("font_size") == "large" else 1.0
        self.title_font = pygame.font.SysFont("segoeui", int(24 * self.font_scale), bold=True)
        self.body_font = pygame.font.SysFont("segoeui", int(16 * self.font_scale))
        self.small_font = pygame.font.SysFont("segoeui", int(14 * self.font_scale))

        self.board_rect = pygame.Rect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE, BOARD_SIZE)
        self.left_rect = pygame.Rect(0, 0, SIDEBAR_LEFT_W, 800)
        self.right_rect = pygame.Rect(860, 0, 420, 800)
        self.eval_bar = EvalBar((54, 150, 50, 310))
        self.move_list = MoveList((888, 128, 364, 400))
        self.opp_card = PlayerCard((18, 26, 144, 92), self._opponent_name(), self._opponent_rating(), self._opponent_color_name())
        self.player_card = PlayerCard((18, 626, 144, 92), self.profile_name if self.config.has_ai else self._player_name(), self._player_rating(), self._player_color_name())
        self.stats_rect = pygame.Rect(888, 542, 364, 92)
        self.undo_button = Button((888, 692, 112, 42), "Undo", icon="↩")
        self.hint_button = Button((1014, 692, 112, 42), "Hint", icon="💡")
        self.resign_button = Button((1140, 692, 112, 42), "Resign", icon="⚑", color=DANGER)
        self.confirm_yes = Button((0, 0, 120, 40), "Confirm", color=DANGER)
        self.confirm_no = Button((0, 0, 120, 40), "Cancel", color=BG_CARD)

        self.clock = None
        if self.config.time_control and self.config.time_control.enabled:
            tc = self.config.time_control
            self.clock = ChessClock(tc.initial_ms, tc.initial_ms, tc.increment_ms)
            self.clock.start("white")

        if self.app.sound_manager is not None:
            self.app.sound_manager.set_enabled(bool(self.config.settings.get("sound", True)))

        self._set_status()
        self._update_eval()

    def _opponent_name(self) -> str:
        if self.config.has_ai:
            return "Cortex64"
        return "Black" if self.config.human == "white" else "White"

    def _player_name(self) -> str:
        return "White" if self.config.human == "white" else "Black"

    def _opponent_color_name(self) -> str:
        return "black" if self.config.human == "white" else "white"

    def _player_color_name(self) -> str:
        return "white" if self.config.human == "white" else "black"

    def _opponent_rating(self) -> int:
        return 2140 if self.config.has_ai else 1200

    def _player_rating(self) -> int:
        wins = int(self.app.profile.get("wins", 0))
        losses = int(self.app.profile.get("losses", 0))
        return max(800, 1200 + wins * 8 - losses * 5)

    def _animation_duration(self, base: int) -> int:
        mode = self.config.settings.get("animations", "normal")
        if mode == "off":
            return 0
        if mode == "fast":
            return max(40, base // 2)
        return base

    def _board_flipped(self) -> bool:
        return self.config.human == "black"

    def _board_to_display(self, row: int, col: int) -> tuple[int, int]:
        return (7 - row, 7 - col) if self._board_flipped() else (row, col)

    def _display_to_board(self, row: int, col: int) -> tuple[int, int]:
        return (7 - row, 7 - col) if self._board_flipped() else (row, col)

    def _square_rect(self, square: int, with_shake: bool = False) -> pygame.Rect:
        row, col = divmod(square, 8)
        row, col = self._board_to_display(row, col)
        shake = self.animation.get_shake_offset_x() if with_shake else 0
        return pygame.Rect(
            self.board_rect.x + shake + col * SQUARE_SIZE,
            self.board_rect.y + row * SQUARE_SIZE,
            SQUARE_SIZE,
            SQUARE_SIZE,
        )

    def _square_center(self, square: int) -> tuple[int, int]:
        return self._square_rect(square).center

    def _square_from_mouse(self, pos: tuple[int, int]) -> int | None:
        shake = self.animation.get_shake_offset_x()
        rect = self.board_rect.move(shake, 0)
        if not rect.collidepoint(pos):
            return None
        display_col = (pos[0] - rect.x) // SQUARE_SIZE
        display_row = (pos[1] - rect.y) // SQUARE_SIZE
        row, col = self._display_to_board(display_row, display_col)
        return row * 8 + col

    def _captured_piece(self, board_before: Board, move: Move) -> int:
        if move.is_en_passant:
            direction = -8 if board_before.side_to_move == BLACK else 8
            return board_before.piece_at(move.to_sq + direction)
        return board_before.piece_at(move.to_sq)

    def _is_controlled_color(self, color: int) -> bool:
        return True if not self.config.has_ai else color == self.human_color

    def _is_human_turn(self) -> bool:
        return self._is_controlled_color(self.board.side_to_move)

    def _select_square(self, square: int | None) -> None:
        self.selected_sq = square
        self.selected_moves = legal_moves_from(self.board, square) if square is not None else []

    def _clear_selection(self) -> None:
        self.selected_sq = None
        self.selected_moves = []

    def _set_status(self) -> None:
        if is_checkmate(self.board):
            self.status_text = "Checkmate"
            return
        if is_stalemate(self.board):
            self.status_text = "Stalemate"
            return
        name = self.profile_name if self.board.side_to_move == self.human_color or not self.config.has_ai else "Cortex64"
        if not self.config.has_ai:
            name = "White" if self.board.side_to_move == WHITE else "Black"
        suffix = " · check" if is_in_check(self.board, self.board.side_to_move) else ""
        self.status_text = f"{name} to move{suffix}"

    def _update_eval(self) -> None:
        self.eval_bar.set_eval(float(self.quality_evaluator.evaluate(self.board)) * 100.0)

    def _play_sound(self, name: str) -> None:
        if self.app.sound_manager is not None:
            self.app.sound_manager.play(name)

    def _move_to_san(self, board_before: Board, move: Move) -> str:
        piece = abs(board_before.piece_at(move.from_sq))
        if move.is_castling:
            san = "O-O" if move.to_sq % 8 == 6 else "O-O-O"
        else:
            capture = self._captured_piece(board_before, move) != 0
            prefix = PIECE_LETTERS.get(piece, "")
            if piece == 1 and capture:
                prefix = square_name(move.from_sq)[0]

            disambiguation = ""
            if piece != 1:
                similar = []
                for candidate in generate_legal_moves(board_before):
                    if candidate == move or candidate.to_sq != move.to_sq:
                        continue
                    if abs(board_before.piece_at(candidate.from_sq)) == piece:
                        similar.append(candidate)
                if similar:
                    from_name = square_name(move.from_sq)
                    same_file = any(square_name(other.from_sq)[0] == from_name[0] for other in similar)
                    same_rank = any(square_name(other.from_sq)[1] == from_name[1] for other in similar)
                    if not same_file:
                        disambiguation = from_name[0]
                    elif not same_rank:
                        disambiguation = from_name[1]
                    else:
                        disambiguation = from_name

            san = f"{prefix}{disambiguation}{'x' if capture else ''}{square_name(move.to_sq)}"
            if move.promotion is not None:
                san += f"={PIECE_LETTERS.get(abs(move.promotion), 'Q')}"

        board_after = board_before.copy()
        board_after.push(move)
        if is_checkmate(board_after):
            san += "#"
        elif is_in_check(board_after, board_after.side_to_move):
            san += "+"
        return san

    def _build_hint_explanation(self, board_before: Board, move: Move) -> str:
        board_after = board_before.copy()
        board_after.push(move)
        captured = self._captured_piece(board_before, move)
        if is_checkmate(board_after):
            return "Mate threat — this move ends the game immediately."
        if captured:
            return f"Material gain — captures on {square_name(move.to_sq)}."
        if move.is_castling:
            return "King safety — tucks the king away and connects rooks."
        if is_in_check(board_after, board_after.side_to_move):
            return f"Threat — checks the king from {square_name(move.to_sq)}."
        return "Development — improves coordination and central control."

    def _mark_check(self) -> None:
        if is_in_check(self.board, self.board.side_to_move):
            king_sq = self.board.white_king_sq if self.board.side_to_move == WHITE else self.board.black_king_sq
            pulse_ms = self._animation_duration(ANIM_PULSE_MS)
            if pulse_ms:
                self.animation.add_check_pulse(self._square_rect(king_sq).topleft, pulse_ms)
            self.banner_text = "CHECK!"
            self.banner_until = pygame.time.get_ticks() + 1500
            self._play_sound("check")

    def _apply_move(self, move: Move, actor: str) -> None:
        board_before = self.board.copy()
        moving_piece = board_before.piece_at(move.from_sq)
        captured = self._captured_piece(board_before, move)
        san = self._move_to_san(board_before, move)
        delta_cp, best_uci = evaluate_move_delta(board_before, move, self.quality_evaluator, depth=1)
        quality = classify_quality(delta_cp)
        explanation = self._build_hint_explanation(board_before, move)

        duration = self._animation_duration(ANIM_PIECE_MS)
        if duration:
            piece_surface = self.piece_images[moving_piece]
            start_rect = self._square_rect(move.from_sq)
            end_rect = self._square_rect(move.to_sq)
            start_pos = (
                start_rect.x + (SQUARE_SIZE - piece_surface.get_width()) // 2,
                start_rect.y + (SQUARE_SIZE - piece_surface.get_height()) // 2,
            )
            end_pos = (
                end_rect.x + (SQUARE_SIZE - piece_surface.get_width()) // 2,
                end_rect.y + (SQUARE_SIZE - piece_surface.get_height()) // 2,
            )
            self.animation.add_piece_move(
                piece_surface,
                start_pos,
                end_pos,
                duration,
            )
            self.hidden_anim_to_sq = move.to_sq

        self.board.push(move)
        self.move_history.append(move)
        self.last_move = move
        record = MoveRecord(
            move=move,
            san=san,
            color="white" if piece_color(moving_piece) == WHITE else "black",
            move_num=(len(self.move_history) + 1) // 2,
            quality=quality,
            eval_cp=float(self.quality_evaluator.evaluate(self.board)) * 100.0,
            delta_cp=delta_cp,
            best_move_uci=best_uci,
            explanation=explanation,
        )
        self.move_records.append(record)
        self.eval_history.append(record.eval_cp)
        self.move_list.add_move(san, record.color, record.move_num, quality)
        self.hint_move = None
        self.hint_explanation = ""
        self._clear_selection()
        self.pending_promotion = None

        if self.clock is not None:
            self.clock.press("white" if piece_color(moving_piece) == WHITE else "black")
        self._play_sound("capture" if captured else "move")
        self._update_eval()
        self._mark_check()
        self._set_status()

        if actor == "human":
            self.banner_text = {"good": "Good move", "inaccuracy": "Inaccuracy", "mistake": "Mistake"}[quality]
            self.banner_until = pygame.time.get_ticks() + 1200

        if is_checkmate(self.board):
            winner = "white" if self.board.side_to_move == BLACK else "black"
            self._finish_game("Checkmate", winner)
        elif is_stalemate(self.board):
            self._finish_game("Stalemate", None)

    def _finish_game(self, reason: str, winner: str | None) -> None:
        if self.results_summary is not None:
            return
        if self.clock is not None:
            self.clock.pause()
        self.ai_thinking = False
        result = "draw"
        if winner is not None:
            result = "win" if winner == self.config.human else "loss"

        self.app.profile["games_played"] = int(self.app.profile.get("games_played", 0)) + 1
        if result == "win":
            self.app.profile["wins"] = int(self.app.profile.get("wins", 0)) + 1
        elif result == "loss":
            self.app.profile["losses"] = int(self.app.profile.get("losses", 0)) + 1
        else:
            self.app.profile["draws"] = int(self.app.profile.get("draws", 0)) + 1
        self.app.profile["username"] = self.profile_name
        self.app.profile["preferred_side"] = self.config.human
        save_profile(self.app.profile)

        self.results_summary = GameSummary(
            config=replace(self.config, settings=dict(self.app.settings)),
            result=result,
            reason=reason,
            winner=winner,
            move_records=list(self.move_records),
            evals=list(self.eval_history),
            total_time_ms=self.total_elapsed_ms,
        )
        self.results_pending = True
        self._play_sound("game_end")

    def _maybe_timeout(self, flagged: str) -> None:
        winner = "black" if flagged == "white" else "white"
        self._finish_game("Timeout", winner)

    def _start_ai_search(self) -> None:
        if not self.config.has_ai or self.ai_thinking or self.results_summary is not None:
            return
        if self.board.side_to_move != self.ai_color or self.pending_promotion is not None:
            return

        self.ai_thinking = True
        board_snapshot = self.board.copy()

        def worker() -> None:
            payload = search_best_move_with_budget(board_snapshot, self.config.depth, self.config.move_time, self.evaluator)
            with self.ai_lock:
                self.ai_result = payload

        self.ai_thread = threading.Thread(target=worker, daemon=True)
        self.ai_thread.start()

    def _poll_ai(self) -> None:
        if not self.ai_thinking:
            return
        with self.ai_lock:
            payload = self.ai_result
            self.ai_result = None
        if payload is None:
            return
        move, score, nodes, depth_used, elapsed = payload
        self.ai_thinking = False
        self.ai_last_time = elapsed
        self.ai_last_nodes = int(nodes)
        self.ai_last_depth = int(depth_used)
        self.ai_last_eval_cp = float(score) * (-100.0 if self.ai_color == BLACK else 100.0)
        if move is not None and self.results_summary is None and self.board.side_to_move == self.ai_color:
            self._apply_move(move, "ai")

    def _undo(self) -> None:
        if self.ai_thinking or not self.move_history or self.results_summary is not None:
            return
        plies = 1 if not self.config.has_ai else min(2, len(self.move_history))
        for _ in range(plies):
            self.board.pop()
            self.move_history.pop()
            self.move_records.pop()
            if self.eval_history:
                self.eval_history.pop()
            if self.move_list.entries:
                self.move_list.entries.pop()
        self.move_list.current = len(self.move_list.entries) - 1
        self.last_move = self.move_history[-1] if self.move_history else None
        self.confirm_resign = False
        self.hint_move = None
        self._clear_selection()
        self._set_status()
        self._update_eval()

    def _show_hint(self) -> None:
        if self.ai_thinking or self.results_summary is not None or not self._is_human_turn():
            return
        move, _score, _nodes, _depth_used, _elapsed = search_best_move_with_budget(
            self.board.copy(),
            min(self.config.depth, 8),
            min(1.2, self.config.move_time),
            self.evaluator,
        )
        if move is None:
            return
        self.hint_move = move
        self.hint_explanation = self._build_hint_explanation(self.board.copy(), move)
        self.hint_until = pygame.time.get_ticks() + 3000
        self.hint_fade_end = self.hint_until + ANIM_FADE_MS

    def _attempt_move(self, from_sq: int, to_sq: int) -> bool:
        if self.results_summary is not None or not self._is_human_turn():
            return False
        candidates = [move for move in legal_moves_from(self.board, from_sq) if move.to_sq == to_sq]
        if not candidates:
            return False
        promos = [move for move in candidates if move.promotion is not None]
        if promos:
            self.pending_promotion = (from_sq, to_sq, promos)
            self.banner_text = "Promotion: press Q, R, B, or N"
            self.banner_until = pygame.time.get_ticks() + 1600
            return True
        self._apply_move(candidates[0], "human")
        return True

    def _apply_promotion_key(self, key: int) -> None:
        if self.pending_promotion is None:
            return
        wanted = {pygame.K_q: QUEEN, pygame.K_r: ROOK, pygame.K_b: BISHOP, pygame.K_n: KNIGHT}.get(key)
        if wanted is None:
            return
        for move in self.pending_promotion[2]:
            if abs(move.promotion or 0) == wanted:
                self._apply_move(move, "human")
                return

    def _illegal_feedback(self) -> None:
        duration = self._animation_duration(ANIM_SHAKE_MS)
        if duration:
            self.animation.add_shake(duration)
        self.banner_text = "Illegal move"
        self.banner_until = pygame.time.get_ticks() + 900
        self._play_sound("illegal")

    def _handle_board_down(self, pos: tuple[int, int]) -> None:
        square = self._square_from_mouse(pos)
        if square is None or not self._is_human_turn():
            return
        piece = self.board.piece_at(square)
        if piece and piece_color(piece) == self.board.side_to_move:
            self.drag_sq = square
            self.drag_piece = piece
            self.drag_origin = pos
            self.drag_pos = pos
            self.dragging = False
            self._select_square(square)
        elif self.selected_sq is not None:
            if not self._attempt_move(self.selected_sq, square):
                self._clear_selection()
                self._illegal_feedback()

    def _handle_board_up(self, pos: tuple[int, int]) -> None:
        if self.drag_sq is None:
            return
        target = self._square_from_mouse(pos)
        from_sq = self.drag_sq
        was_dragging = self.dragging
        self.drag_sq = None
        self.drag_piece = None
        self.drag_origin = None
        self.drag_pos = None
        self.dragging = False
        if target is None:
            return
        if was_dragging or target != from_sq:
            success = self._attempt_move(from_sq, target)
            if not success and target != from_sq:
                if self.board.piece_at(target) and piece_color(self.board.piece_at(target)) == self.board.side_to_move:
                    self._select_square(target)
                else:
                    self._illegal_feedback()

    def _handle_buttons(self, event: pygame.event.Event) -> None:
        for button, action in (
            (self.undo_button, self._undo),
            (self.hint_button, self._show_hint),
            (self.resign_button, lambda: setattr(self, "confirm_resign", True)),
        ):
            if button.handle_event(event) == "clicked" and not button.disabled:
                action()

    def update(self, events: list[pygame.event.Event], dt_ms: int):
        if self.results_ready and self.results_summary is not None:
            self.results_summary.background = self.screen.copy()
            self.app.last_summary = self.results_summary
            from gui.screens.results import ResultsScreen

            return ResultsScreen(self.screen, self.app, self.results_summary)

        self.total_elapsed_ms += dt_ms
        self.animation.update(dt_ms)
        self.eval_bar.update(dt_ms)
        if self.hidden_anim_to_sq is not None and not self.animation.is_animating_move():
            self.hidden_anim_to_sq = None
        if self.clock is not None and self.results_summary is None:
            flagged = self.clock.update(dt_ms)
            if flagged is not None:
                self._maybe_timeout(flagged)

        self.undo_button.disabled = self.ai_thinking or not self.move_history
        self.hint_button.disabled = self.ai_thinking or not self._is_human_turn() or self.results_summary is not None
        self.resign_button.disabled = self.ai_thinking or self.results_summary is not None

        self._poll_ai()
        if self.results_summary is None and not self.ai_thinking:
            self._start_ai_search()

        for event in events:
            if self.confirm_resign:
                dialog = pygame.Rect(0, 0, 320, 160)
                dialog.center = (640, 400)
                self.confirm_yes.rect = pygame.Rect(dialog.x + 38, dialog.bottom - 56, 110, 40)
                self.confirm_no.rect = pygame.Rect(dialog.right - 148, dialog.bottom - 56, 110, 40)
                if self.confirm_yes.handle_event(event) == "clicked":
                    self.confirm_resign = False
                    winner = "black" if self.config.human == "white" else "white"
                    self._finish_game("Resignation", winner)
                if self.confirm_no.handle_event(event) == "clicked":
                    self.confirm_resign = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.confirm_resign = False
                continue

            if self.move_list.handle_event(event) is not None:
                continue
            self._handle_buttons(event)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._clear_selection()
                    self.pending_promotion = None
                if event.key in (pygame.K_q, pygame.K_r, pygame.K_b, pygame.K_n):
                    self._apply_promotion_key(event.key)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.stats_rect.collidepoint(event.pos):
                    self.ai_stats_expanded = not self.ai_stats_expanded
                elif self.board_rect.inflate(18, 18).move(self.animation.get_shake_offset_x(), 0).collidepoint(event.pos):
                    self._handle_board_down(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._handle_board_up(event.pos)
            elif event.type == pygame.MOUSEMOTION and self.drag_sq is not None:
                self.drag_pos = event.pos
                if self.drag_origin and (abs(event.pos[0] - self.drag_origin[0]) > 4 or abs(event.pos[1] - self.drag_origin[1]) > 4):
                    self.dragging = True

        return self

    def _draw_board(self, surface: pygame.Surface) -> None:
        skin = theme.get_skin()
        shake = self.animation.get_shake_offset_x()
        container = self.board_rect.inflate(18, 18).move(shake, 0)
        pygame.draw.rect(surface, BG_SURFACE, container, border_radius=18)
        pygame.draw.rect(surface, (0, 0, 0), container.inflate(-4, -4), width=2, border_radius=16)

        for row in range(8):
            for col in range(8):
                display_row, display_col = self._board_to_display(row, col)
                rect = pygame.Rect(
                    self.board_rect.x + shake + display_col * SQUARE_SIZE,
                    self.board_rect.y + display_row * SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                )
                color = skin["light"] if (row + col) % 2 == 0 else skin["dark"]
                pygame.draw.rect(surface, color, rect)

        if self.last_move is not None:
            for square in (self.last_move.from_sq, self.last_move.to_sq):
                overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                overlay.fill((*ACCENT, 80))
                surface.blit(overlay, self._square_rect(square, True).topleft)

        if self.selected_sq is not None:
            rect = self._square_rect(self.selected_sq, True)
            overlay = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            overlay.fill((*ACCENT, 140))
            surface.blit(overlay, rect.topleft)
            pygame.draw.rect(surface, ACCENT, rect, width=3, border_radius=8)

        if self.config.settings.get("show_legal_moves", True):
            for move in self.selected_moves:
                rect = self._square_rect(move.to_sq, True)
                target = self.board.piece_at(move.to_sq)
                if target or move.is_en_passant:
                    pygame.draw.circle(surface, GOLD, rect.center, 18, width=4)
                else:
                    dot = pygame.Surface((20, 20), pygame.SRCALPHA)
                    pygame.draw.circle(dot, (255, 255, 255, 80), (10, 10), 10)
                    surface.blit(dot, dot.get_rect(center=rect.center))

        self.animation.draw_check_pulse(surface, SQUARE_SIZE)
        self.animation.draw_moving_pieces(surface)

        for row in range(8):
            for col in range(8):
                piece = int(self.board.squares[row, col])
                if piece == 0:
                    continue
                square = row * 8 + col
                if self.hidden_anim_to_sq == square:
                    continue
                if self.dragging and self.drag_sq == square:
                    continue
                image = self.piece_images[piece]
                surface.blit(image, image.get_rect(center=self._square_rect(square, True).center))

        if self.dragging and self.drag_piece is not None and self.drag_pos is not None:
            image = self.piece_images[self.drag_piece]
            surface.blit(image, image.get_rect(center=self.drag_pos))

        files = "hgfedcba" if self._board_flipped() else "abcdefgh"
        ranks = [str(index + 1) for index in range(8)] if self._board_flipped() else [str(8 - index) for index in range(8)]
        for col in range(8):
            label = self.small_font.render(files[col], True, MUTED)
            surface.blit(label, (self.board_rect.x + shake + col * SQUARE_SIZE + SQUARE_SIZE - 16, self.board_rect.bottom - 20))
        for row in range(8):
            label = self.small_font.render(ranks[row], True, MUTED)
            surface.blit(label, (self.board_rect.x + shake + 6, self.board_rect.y + row * SQUARE_SIZE + 6))

    def _draw_sidebars(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, BG_CARD, self.left_rect)
        pygame.draw.rect(surface, BG_CARD, self.right_rect)

        opp_turn = (self.board.side_to_move == (BLACK if self.config.human == "white" else WHITE)) if self.config.has_ai else (self.board.side_to_move != self.human_color)
        self.opp_card.active = opp_turn
        self.player_card.active = not opp_turn
        self.opp_card.draw(surface)
        self.player_card.draw(surface)

        if self.config.settings.get("show_eval_bar", True):
            self.eval_bar.draw(surface)
        if self.clock is not None:
            top_clock = pygame.Rect(18, 486, 144, 54)
            bottom_clock = pygame.Rect(18, 554, 144, 54)
            self.clock.draw(surface, top_clock, "white" if self.config.human == "black" else "black")
            self.clock.draw(surface, bottom_clock, self.config.human)

        score = self.title_font.render("CORTEX64", True, ACCENT)
        surface.blit(score, (888, 32))
        surface.blit(self.body_font.render("⬡ Engine HUD", True, MUTED), (890, 66))
        profile_line = self.small_font.render(
            f"W/L/D {self.app.profile.get('wins', 0)}/{self.app.profile.get('losses', 0)}/{self.app.profile.get('draws', 0)}",
            True,
            MUTED,
        )
        surface.blit(profile_line, (890, 92))
        surface.blit(self.title_font.render("Moves", True, WHITE_COL), (888, 104))
        self.move_list.draw(surface)

        pygame.draw.rect(surface, BG_SURFACE, self.stats_rect, border_radius=14)
        pygame.draw.rect(surface, ACCENT if self.ai_stats_expanded else MUTED, self.stats_rect, width=1, border_radius=14)
        collapsed = f"AI: depth {self.ai_last_depth} · {self.ai_last_eval_cp/100:+.1f} · {self.ai_last_time:.1f}s"
        if not self.config.has_ai:
            collapsed = f"Local: {len(self.move_history)} plies · {self.status_text}"
        surface.blit(self.body_font.render(collapsed, True, WHITE_COL), (self.stats_rect.x + 12, self.stats_rect.y + 12))
        if self.ai_stats_expanded:
            items = [
                ("Depth", str(self.ai_last_depth)),
                ("Nodes", str(self.ai_last_nodes)),
                ("Eval", f"{self.ai_last_eval_cp/100:+.2f}"),
                ("Time", f"{self.ai_last_time:.2f}s"),
            ]
            for idx, (label, value) in enumerate(items):
                x = self.stats_rect.x + 12 + (idx % 2) * 170
                y = self.stats_rect.y + 42 + (idx // 2) * 24
                surface.blit(self.small_font.render(f"{label}: {value}", True, WHITE_COL if idx % 2 == 0 else MUTED), (x, y))

        toast_rect = pygame.Rect(888, 646, 364, 32)
        pygame.draw.rect(surface, BG_SURFACE, toast_rect, border_radius=10)
        text = self.hint_explanation if self.hint_move and pygame.time.get_ticks() < self.hint_fade_end else self.status_text
        color = GOLD if self.hint_move and pygame.time.get_ticks() < self.hint_fade_end else WHITE_COL
        surface.blit(self.small_font.render(fit_text(self.small_font, text, 344), True, color), (toast_rect.x + 10, toast_rect.y + 8))

        self.undo_button.draw(surface)
        self.hint_button.draw(surface)
        self.resign_button.draw(surface)

    def _draw_hint(self, surface: pygame.Surface) -> None:
        if self.hint_move is None:
            return
        now = pygame.time.get_ticks()
        if now >= self.hint_fade_end:
            self.hint_move = None
            self.hint_explanation = ""
            return
        alpha = 220 if now <= self.hint_until else int(220 * (self.hint_fade_end - now) / ANIM_FADE_MS)
        draw_arrow(surface, self._square_center(self.hint_move.from_sq), self._square_center(self.hint_move.to_sq), GOLD, width=12, alpha=alpha)

    def _draw_banner(self, surface: pygame.Surface) -> None:
        if not self.banner_text or pygame.time.get_ticks() >= self.banner_until:
            return
        banner = pygame.Rect(0, 0, 240, 40)
        banner.midtop = (640, 18)
        pygame.draw.rect(surface, DANGER if self.banner_text == "CHECK!" else BG_CARD, banner, border_radius=20)
        pygame.draw.rect(surface, ACCENT if self.banner_text != "CHECK!" else DANGER, banner, width=1, border_radius=20)
        txt = self.body_font.render(self.banner_text, True, WHITE_COL)
        surface.blit(txt, txt.get_rect(center=banner.center))

    def _draw_confirm(self, surface: pygame.Surface) -> None:
        if not self.confirm_resign:
            return
        overlay = pygame.Surface((1280, 800), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        surface.blit(overlay, (0, 0))
        dialog = pygame.Rect(0, 0, 320, 160)
        dialog.center = (640, 400)
        pygame.draw.rect(surface, BG_CARD, dialog, border_radius=20)
        pygame.draw.rect(surface, DANGER, dialog, width=2, border_radius=20)
        title = self.title_font.render("Confirm resignation?", True, WHITE_COL)
        body = self.body_font.render("This will immediately end the game.", True, MUTED)
        surface.blit(title, title.get_rect(center=(dialog.centerx, dialog.y + 46)))
        surface.blit(body, body.get_rect(center=(dialog.centerx, dialog.y + 82)))
        self.confirm_yes.draw(surface)
        self.confirm_no.draw(surface)

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG_DARK)
        self._draw_sidebars(surface)
        self._draw_board(surface)
        self._draw_hint(surface)
        self._draw_banner(surface)
        self._draw_confirm(surface)
        if self.results_pending:
            self.results_pending = False
            self.results_ready = True

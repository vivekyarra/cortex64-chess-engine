from __future__ import annotations

import json
import math
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Tuple

import pygame

from ai.evaluate import NeuralEvaluator
from engine.board import (
    BISHOP,
    BLACK,
    EMPTY,
    KNIGHT,
    Move,
    QUEEN,
    ROOK,
    WHITE,
    Board,
    piece_color,
)
from engine.minimax import find_best_move
from engine.move_generator import generate_legal_moves, is_checkmate, is_in_check, is_stalemate, legal_moves_from


BOARD_SIZE = 520
SQUARE_SIZE = 65

LEFT_MARGIN = 24
TOP_MARGIN = 70
RIGHT_PANEL_GAP = 20
RIGHT_PANEL_WIDTH = 340
BOTTOM_MARGIN = 90

WINDOW_WIDTH = LEFT_MARGIN + BOARD_SIZE + RIGHT_PANEL_GAP + RIGHT_PANEL_WIDTH + 36
WINDOW_HEIGHT = TOP_MARGIN + BOARD_SIZE + BOTTOM_MARGIN

BOARD_LEFT = LEFT_MARGIN
BOARD_TOP = TOP_MARGIN

LIGHT_SQUARE = (206, 222, 236)
DARK_SQUARE = (88, 145, 188)
BOARD_BORDER = (22, 60, 78)

SELECTED_BORDER = (45, 190, 255)
LAST_MOVE_FROM = (236, 242, 112, 88)
LAST_MOVE_TO = (236, 242, 112, 136)
LEGAL_MOVE_DOT = (30, 38, 46)
LEGAL_CAPTURE_DOT = (196, 80, 72)
HINT_ARROW_COLOR = (169, 225, 80, 180)

PANEL_BG = (8, 36, 54, 210)
CARD_BG = (12, 49, 73, 220)
TEXT_PRIMARY = (238, 246, 250)
TEXT_MUTED = (168, 196, 210)
ACCENT = (68, 199, 255)
BUTTON_BG = (21, 67, 89)
BUTTON_BG_HOVER = (31, 89, 118)
BUTTON_BG_DISABLED = (37, 52, 63)

PIECE_IMAGE_FILES = {
    1: "wp.png",
    2: "wn.png",
    3: "wb.png",
    4: "wr.png",
    5: "wq.png",
    6: "wk.png",
    -1: "bp.png",
    -2: "bn.png",
    -3: "bb.png",
    -4: "br.png",
    -5: "bq.png",
    -6: "bk.png",
}

FALLBACK_PIECE_LABELS = {
    1: "P",
    2: "N",
    3: "B",
    4: "R",
    5: "Q",
    6: "K",
    -1: "p",
    -2: "n",
    -3: "b",
    -4: "r",
    -5: "q",
    -6: "k",
}

MOVE_QUALITY_THRESHOLDS = {"best": 0.08, "normal": 0.25}
AI_MOVE_TIME_LIMIT_SEC = 1.8
POSTGAME_ANALYSIS_BUDGET_SEC = 1.8
MOVE_FEEDBACK_DURATION_SEC = 2.2
PROFILE_FILENAME = "profile.json"
EXPORTS_DIRNAME = "exports"
MATERIAL_VALUES = {
    1: 1.0,   # pawn
    2: 3.2,   # knight
    3: 3.35,  # bishop
    4: 5.1,   # rook
    5: 9.4,   # queen
    6: 0.0,   # king (safety handled implicitly by checkmate in search)
}


class MaterialEvaluator:
    """
    Fast deterministic evaluator used when no trained CNN model is available.
    """

    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        self.loaded = False

    def evaluate(self, board: Board) -> float:
        score = 0.0
        for row in range(8):
            for col in range(8):
                piece = int(board.squares[row, col])
                if piece == EMPTY:
                    continue

                piece_type = abs(piece)
                base = MATERIAL_VALUES.get(piece_type, 0.0)
                sign = 1.0 if piece > 0 else -1.0

                # Light positional shaping for stronger play without neural weights.
                center_distance = abs(3.5 - row) + abs(3.5 - col)
                if piece_type == 1:  # pawn
                    advance = (6 - row) if piece > 0 else (row - 1)
                    base += 0.06 * advance
                elif piece_type in (2, 3):  # knight / bishop
                    base += max(0.0, 0.22 - 0.04 * center_distance)
                elif piece_type == 4:  # rook on open-ish files
                    base += 0.04 * max(0, 3 - abs(3.5 - col))
                elif piece_type == 5:  # queen centralization late helps
                    base += max(0.0, 0.12 - 0.025 * center_distance)

                score += sign * base
        return score


def _create_search_evaluator(model_path: str):
    neural = NeuralEvaluator(model_path=model_path)
    if neural.loaded:
        return neural
    return MaterialEvaluator(model_path=model_path)


def _timed_search_worker(
    board: Board,
    max_depth: int,
    model_path: str,
    use_neural_model: bool,
    out_queue: mp.Queue,
) -> None:
    """
    Run iterative deepening in a child process.
    The parent enforces wall-time by terminating this process if needed.
    """
    evaluator = NeuralEvaluator(model_path=model_path) if use_neural_model else MaterialEvaluator(model_path=model_path)
    if use_neural_model and not evaluator.loaded:
        evaluator = MaterialEvaluator(model_path=model_path)
    for depth in range(1, max(1, max_depth) + 1):
        result = find_best_move(board.copy(), depth=depth, evaluator=evaluator)
        out_queue.put((depth, result.move, float(result.score), int(result.nodes)))


@dataclass
class PromotionChoice:
    from_sq: int
    to_sq: int
    options: List[Move]


@dataclass
class UIButton:
    key: str
    label: str
    rect: pygame.Rect


class Game:
    def __init__(
        self,
        ai_depth: int = 12,
        ai_move_time_limit: float = AI_MOVE_TIME_LIMIT_SEC,
        human_color: int = WHITE,
        model_path: str = "ai/models/chess_cnn.pt",
        window_size: int = BOARD_SIZE,
        fps: int = 60,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Cortex64")

        self.window_size = BOARD_SIZE
        self.board_px = BOARD_SIZE
        self.square_size = SQUARE_SIZE
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.board = Board()
        self.human_color = human_color
        self.ai_color = -human_color
        self.ai_depth = max(1, ai_depth)
        self.evaluator = _create_search_evaluator(model_path=model_path)
        self.use_neural_model = bool(getattr(self.evaluator, "loaded", False))
        self.eval_label = "CNN" if self.use_neural_model else "Material"

        self.profile_path = Path(__file__).resolve().parent / "data" / PROFILE_FILENAME
        self.exports_dir = Path(__file__).resolve().parent / EXPORTS_DIRNAME
        self.total_games = 0
        self.total_wins = 0
        self.total_losses = 0
        self.total_draws = 0

        self.user_name = "Soprano"
        self.ai_name = "Cortex64"
        self.editing_name = False
        self.awaiting_start_choice = True

        self.selected_sq: Optional[int] = None
        self.selected_moves: List[Move] = []
        self.promotion: Optional[PromotionChoice] = None
        self.last_move: Optional[Move] = None
        self.hint_move: Optional[Move] = None
        self.move_history: List[Move] = []

        self.game_state = "playing"
        self.status_text = ""
        self.game_over = False

        self.end_title = ""
        self.end_reason = ""
        self.popup_visible = False
        self.popup_anim = 0.0
        self.popup_close_rect: Optional[pygame.Rect] = None
        self.popup_rematch_rect: Optional[pygame.Rect] = None
        self.popup_continue_rect: Optional[pygame.Rect] = None
        self.popup_switch_rect: Optional[pygame.Rect] = None
        self.popup_analyze_rect: Optional[pygame.Rect] = None
        self.popup_export_rect: Optional[pygame.Rect] = None
        self.popup_analysis_prev_rect: Optional[pygame.Rect] = None
        self.popup_analysis_next_rect: Optional[pygame.Rect] = None
        self.popup_analysis_back_rect: Optional[pygame.Rect] = None
        self.rematch_choice_visible = False
        self.analysis_mode_active = False
        self.analysis_view_index = 0
        self.analysis_view_board: Optional[Board] = None
        self.analysis_records: List[Dict[str, object]] = []
        self.popup_notice = ""
        self.popup_notice_until = 0.0
        self.postgame_stats = self._empty_postgame_stats()
        self.last_result = ""

        # Session scoreboard resets naturally every app run.
        self.session_games = 0
        self.session_user_wins = 0
        self.session_ai_wins = 0
        self.session_draws = 0

        self.ai_thinking = False
        self.ai_start_ts = 0.0
        self.ai_last_time = 0.0
        self.ai_last_depth = 0
        self.ai_last_nodes = 0
        self.ai_last_eval_white = 0.0
        self.ai_task_id = 0
        self.ai_thread: Optional[threading.Thread] = None
        self.ai_result_payload: Optional[Tuple[int, Optional[Move], float, int, float, int]] = None
        self.ai_result_lock = threading.Lock()
        self.ai_move_time_limit = max(0.4, float(ai_move_time_limit))

        self.analysis_in_progress = False
        self.analysis_task_id = 0
        self.analysis_thread: Optional[threading.Thread] = None
        self.analysis_result_payload: Optional[Tuple[int, Dict[str, Dict[str, int]], List[Dict[str, object]]]] = None
        self.analysis_lock = threading.Lock()

        self.hint_explanation = ""
        self.move_feedback_text = ""
        self.move_feedback_until = 0.0
        self.move_feedback_color = TEXT_MUTED

        self.font_title = pygame.font.SysFont("segoeui", 34, bold=True)
        self.font_panel_title = pygame.font.SysFont("segoeui", 28, bold=True)
        self.font_name = pygame.font.SysFont("segoeui", 24, bold=True)
        self.font_player = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_body = pygame.font.SysFont("segoeui", 22)
        self.font_small = pygame.font.SysFont("segoeui", 18)
        self.font_button = pygame.font.SysFont("segoeui", 22, bold=True)
        self.font_board = pygame.font.SysFont("consolas", 16, bold=True)

        self.board_rect = pygame.Rect(BOARD_LEFT, BOARD_TOP, self.board_px, self.board_px)
        self.ai_panel_rect = pygame.Rect(BOARD_LEFT, max(6, BOARD_TOP - 62), self.board_px, 58)
        self.user_panel_rect = pygame.Rect(BOARD_LEFT, BOARD_TOP + self.board_px + 16, self.board_px, 58)
        self.right_panel_rect = pygame.Rect(BOARD_LEFT + self.board_px + RIGHT_PANEL_GAP, 10, RIGHT_PANEL_WIDTH, WINDOW_HEIGHT - 20)
        self.user_name_input_rect = pygame.Rect(self.user_panel_rect.x + 90, self.user_panel_rect.y + 12, 400, 34)

        button_w = 104
        button_h = 56
        button_gap = 14
        total_w = 3 * button_w + 2 * button_gap
        start_x = self.right_panel_rect.x + (self.right_panel_rect.width - total_w) // 2
        start_y = self.right_panel_rect.bottom - button_h - 26
        self.buttons: Dict[str, UIButton] = {
            "undo": UIButton("undo", "Undo", pygame.Rect(start_x, start_y, button_w, button_h)),
            "hint": UIButton("hint", "Hint", pygame.Rect(start_x + button_w + button_gap, start_y, button_w, button_h)),
            "resign": UIButton("resign", "Resign", pygame.Rect(start_x + 2 * (button_w + button_gap), start_y, button_w, button_h)),
        }
        self.move_list_row_height = 22
        self.move_list_scroll = 0
        self.move_list_auto_follow = True
        self.move_list_rect = self._recalculate_move_list_rect()

        self.background = self._build_background_surface()
        self.piece_images = self._load_piece_images()
        self.cortex_avatar = self._load_avatar("cortex.png", 40, "C", (82, 164, 132))

        setup_w, setup_h = 500, 280
        self.setup_panel_rect = pygame.Rect((WINDOW_WIDTH - setup_w) // 2, (WINDOW_HEIGHT - setup_h) // 2, setup_w, setup_h)
        self.setup_name_input_rect = pygame.Rect(self.setup_panel_rect.x + 90, self.setup_panel_rect.y + 78, 300, 42)
        self.setup_white_rect = pygame.Rect(self.setup_panel_rect.x + 68, self.setup_panel_rect.y + 194, 150, 52)
        self.setup_black_rect = pygame.Rect(self.setup_panel_rect.x + 282, self.setup_panel_rect.y + 194, 150, 52)

        _ = window_size
        self._load_profile()
        self.analysis_view_board = self.board.copy()
        self.status_text = "Choose White or Black to start."

    def _empty_postgame_stats(self) -> Dict[str, Dict[str, int]]:
        return {"user": {"best": 0, "normal": 0, "worst": 0}, "ai": {"best": 0, "normal": 0, "worst": 0}}

    def _default_profile(self) -> Dict[str, object]:
        return {
            "username": "Soprano",
            "games_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "preferred_side": "white",
        }

    def _load_profile(self) -> None:
        data = self._default_profile()
        if self.profile_path.exists():
            try:
                with self.profile_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    data.update(loaded)
            except Exception:
                pass

        self.user_name = str(data.get("username", "Soprano")).strip() or "Soprano"
        self.total_games = int(data.get("games_played", 0))
        self.total_wins = int(data.get("wins", 0))
        self.total_losses = int(data.get("losses", 0))
        self.total_draws = int(data.get("draws", 0))
        side = str(data.get("preferred_side", "white")).lower()
        self.human_color = WHITE if side != "black" else BLACK
        self.ai_color = -self.human_color

    def _save_profile(self) -> None:
        side = "white" if self.human_color == WHITE else "black"
        payload = {
            "username": self.user_name.strip() or "Soprano",
            "games_played": int(self.total_games),
            "wins": int(self.total_wins),
            "losses": int(self.total_losses),
            "draws": int(self.total_draws),
            "preferred_side": side,
        }
        try:
            self.profile_path.parent.mkdir(parents=True, exist_ok=True)
            with self.profile_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def _set_popup_notice(self, text: str, duration: float = 2.8) -> None:
        self.popup_notice = text
        self.popup_notice_until = time.time() + max(0.8, duration)

    def _game_result_code(self) -> str:
        if self.last_result == "user":
            return "1-0" if self.human_color == WHITE else "0-1"
        if self.last_result == "ai":
            return "0-1" if self.human_color == WHITE else "1-0"
        return "1/2-1/2"

    def _export_game_pgn(self) -> Optional[Path]:
        white_name = self.user_name if self.human_color == WHITE else self.ai_name
        black_name = self.ai_name if self.human_color == WHITE else self.user_name
        result = self._game_result_code()
        date_hdr = datetime.now().strftime("%Y.%m.%d")

        move_tokens: List[str] = []
        for i in range(0, len(self.move_history), 2):
            move_tokens.append(f"{i // 2 + 1}.")
            move_tokens.append(self.move_history[i].uci())
            if i + 1 < len(self.move_history):
                move_tokens.append(self.move_history[i + 1].uci())
        move_tokens.append(result)

        pgn = "\n".join(
            [
                '[Event "Cortex64 Single Player"]',
                '[Site "Local"]',
                f'[Date "{date_hdr}"]',
                f'[White "{white_name}"]',
                f'[Black "{black_name}"]',
                f'[Result "{result}"]',
                "",
                " ".join(move_tokens),
                "",
            ]
        )

        try:
            self.exports_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self.exports_dir / f"cortex64_{stamp}.pgn"
            out.write_text(pgn, encoding="utf-8")
            return out
        except Exception:
            return None

    def _render_board(self) -> Board:
        if self.popup_visible and self.analysis_mode_active and self.analysis_view_board is not None:
            return self.analysis_view_board
        return self.board

    def _build_board_at_ply(self, ply_index: int) -> Board:
        idx = max(0, min(len(self.move_history), ply_index))
        board = Board()
        for i in range(idx):
            board.push(self.move_history[i])
        return board

    def _set_analysis_index(self, new_index: int) -> None:
        self.analysis_view_index = max(0, min(len(self.move_history), int(new_index)))
        self.analysis_view_board = self._build_board_at_ply(self.analysis_view_index)

    def _enter_analysis_mode(self) -> None:
        if not self.popup_visible:
            return
        self.analysis_mode_active = True
        self.rematch_choice_visible = False
        self._set_analysis_index(len(self.move_history))

    def _leave_analysis_mode(self) -> None:
        self.analysis_mode_active = False
        self.popup_analysis_prev_rect = None
        self.popup_analysis_next_rect = None
        self.popup_analysis_back_rect = None
        self.analysis_view_board = self.board.copy()

    def _captured_piece_for_move(self, board: Board, move: Move) -> int:
        if move.is_en_passant:
            direction = -8 if board.side_to_move == BLACK else 8
            return board.piece_at(move.to_sq + direction)
        return board.piece_at(move.to_sq)

    def _count_checking_moves(self, board: Board, attacker_color: int, defender_color: int) -> int:
        probe = board.copy()
        probe.side_to_move = attacker_color
        count = 0
        for candidate in generate_legal_moves(probe):
            probe.push(candidate)
            if is_in_check(probe, defender_color):
                count += 1
            probe.pop()
        return count

    def _build_hint_explanation(self, board_before: Board, move: Move) -> str:
        mover = board_before.side_to_move
        board_after = board_before.copy()
        board_after.push(move)

        if is_checkmate(board_after):
            return "Check or mate threat: this move is immediate checkmate."
        if is_in_check(board_before, mover) and not is_in_check(board_after, mover):
            return "Threat prevention: this move gets your king out of danger."

        captured = self._captured_piece_for_move(board_before, move)
        captured_value = MATERIAL_VALUES.get(abs(captured), 0.0)
        if captured != EMPTY and captured_value > 0.0:
            return f"Material gain: wins about {captured_value:.1f} points."

        if move.is_castling or (abs(board_before.piece_at(move.from_sq)) == 6 and not is_in_check(board_after, mover)):
            return "King safety improvement: king position becomes safer."

        if is_in_check(board_after, -mover):
            return "Check threat: puts the opponent king in check."

        checks_before = self._count_checking_moves(board_before, -mover, mover)
        checks_after = self._count_checking_moves(board_after, -mover, mover)
        if checks_after < checks_before:
            return "Threat prevention: reduces opponent checking chances."

        eval_before = float(mover) * float(self.evaluator.evaluate(board_before))
        eval_after = float(mover) * float(self.evaluator.evaluate(board_after))
        delta = eval_after - eval_before
        if delta >= 0.12:
            return f"Positional gain: improves evaluation by {delta:+.2f}."
        return "Solid move: keeps the position stable."

    def _evaluate_move_delta(self, board_before: Board, move: Move, depth: int = 1) -> float:
        analysis_depth = max(1, depth)
        best_result = find_best_move(board_before.copy(), depth=analysis_depth, evaluator=self.evaluator)
        best_score = float(best_result.score) if best_result.move is not None else 0.0

        board_after = board_before.copy()
        board_after.push(move)
        if is_checkmate(board_after):
            played_score = 1000.0
        else:
            reply = find_best_move(board_after.copy(), depth=analysis_depth, evaluator=self.evaluator)
            played_score = 0.0 if reply.move is None else -float(reply.score)
        return max(0.0, best_score - played_score)

    def _set_move_feedback_from_delta(self, delta: float) -> None:
        quality = self._classify_move_delta(delta)
        if quality == "best":
            self.move_feedback_text = "Good move"
            self.move_feedback_color = (126, 214, 151)
        elif quality == "normal":
            self.move_feedback_text = "Inaccuracy"
            self.move_feedback_color = (244, 200, 108)
        else:
            self.move_feedback_text = "Mistake"
            self.move_feedback_color = (236, 123, 110)
        self.move_feedback_until = time.time() + MOVE_FEEDBACK_DURATION_SEC

    def _asset_path(self, relative: str) -> Optional[Path]:
        base = Path(__file__).resolve().parent
        for root in ("assets", "assests"):
            candidate = base / root / relative
            if candidate.exists():
                return candidate
        return None

    def _build_background_surface(self) -> pygame.Surface:
        surface = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        h = WINDOW_HEIGHT - 1 if WINDOW_HEIGHT > 1 else 1
        for y in range(WINDOW_HEIGHT):
            t = y / h
            r = int(6 + 10 * t)
            g = int(40 + 28 * t)
            b = int(56 + 44 * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (WINDOW_WIDTH, y))

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        for i in range(-2, 10):
            x = i * 130
            pygame.draw.polygon(overlay, (32, 124, 148, 18), [(x, 0), (x + 56, 0), (x + 360, WINDOW_HEIGHT), (x + 304, WINDOW_HEIGHT)])
        pygame.draw.rect(overlay, (18, 80, 100, 32), self.right_panel_rect.inflate(30, 10), border_radius=22)
        surface.blit(overlay, (0, 0))
        return surface

    def _recalculate_move_list_rect(self) -> pygame.Rect:
        top = self.right_panel_rect.y + 368
        button_top = min(button.rect.top for button in self.buttons.values())
        bottom = button_top - 14
        height = max(170, bottom - top)
        return pygame.Rect(self.right_panel_rect.x + 18, top, self.right_panel_rect.width - 36, height)

    def _is_board_flipped(self) -> bool:
        return self.human_color == BLACK

    def _board_to_display(self, row: int, col: int) -> Tuple[int, int]:
        if self._is_board_flipped():
            return 7 - row, 7 - col
        return row, col

    def _display_to_board(self, row: int, col: int) -> Tuple[int, int]:
        if self._is_board_flipped():
            return 7 - row, 7 - col
        return row, col

    def _create_piece_fallback(self, piece: int) -> pygame.Surface:
        surf = pygame.Surface((self.square_size - 8, self.square_size - 8), pygame.SRCALPHA)
        center = (surf.get_width() // 2, surf.get_height() // 2)
        radius = min(center) - 2
        fill = (244, 246, 248, 245) if piece > 0 else (22, 31, 41, 245)
        stroke = (35, 57, 80) if piece > 0 else (220, 227, 234)
        pygame.draw.circle(surf, fill, center, radius)
        pygame.draw.circle(surf, stroke, center, radius, width=2)
        label = FALLBACK_PIECE_LABELS.get(piece, "?")
        txt = self.font_name.render(label, True, stroke if piece < 0 else (22, 35, 48))
        surf.blit(txt, txt.get_rect(center=center))
        return surf

    def _load_piece_images(self) -> Dict[int, pygame.Surface]:
        images: Dict[int, pygame.Surface] = {}
        piece_px = self.square_size - 8
        for piece, filename in PIECE_IMAGE_FILES.items():
            img_path = self._asset_path(f"pieces/{filename}")
            if img_path is None:
                images[piece] = self._create_piece_fallback(piece)
                continue
            image = pygame.image.load(str(img_path)).convert_alpha()
            images[piece] = pygame.transform.smoothscale(image, (piece_px, piece_px))
        return images

    def _make_avatar_fallback(self, size: int, label: str, fill_color: Tuple[int, int, int]) -> pygame.Surface:
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        center = (size // 2, size // 2)
        pygame.draw.circle(surf, (12, 35, 47, 230), center, size // 2)
        pygame.draw.circle(surf, fill_color, center, size // 2 - 2)
        txt = self.font_name.render(label[:1].upper(), True, (245, 250, 252))
        surf.blit(txt, txt.get_rect(center=center))
        return surf

    def _load_avatar(self, filename: str, size: int, fallback_label: str, fill_color: Tuple[int, int, int]) -> pygame.Surface:
        path = self._asset_path(filename)
        if path is None:
            return self._make_avatar_fallback(size, fallback_label, fill_color)
        image = pygame.image.load(str(path)).convert_alpha()
        return pygame.transform.smoothscale(image, (size, size))

    def _get_user_avatar(self, size: int) -> pygame.Surface:
        initial = (self.user_name.strip()[:1] or "S").upper()
        # Derive a stable but varied color from first letter.
        hue_seed = ord(initial[0]) if initial else 83
        r = 58 + (hue_seed * 17) % 60
        g = 112 + (hue_seed * 11) % 70
        b = 146 + (hue_seed * 7) % 70
        return self._make_avatar_fallback(size, initial, (r, g, b))

    def _square_from_mouse(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        if not self.board_rect.collidepoint(x, y):
            return None
        display_col = (x - self.board_rect.x) // self.square_size
        display_row = (y - self.board_rect.y) // self.square_size
        row, col = self._display_to_board(display_row, display_col)
        return row * 8 + col

    def _square_rect(self, square: int) -> pygame.Rect:
        row, col = divmod(square, 8)
        display_row, display_col = self._board_to_display(row, col)
        return pygame.Rect(
            self.board_rect.x + display_col * self.square_size,
            self.board_rect.y + display_row * self.square_size,
            self.square_size,
            self.square_size,
        )

    def _square_center(self, square: int) -> Tuple[float, float]:
        rect = self._square_rect(square)
        return float(rect.centerx), float(rect.centery)

    def _draw_square_overlay(self, square: int, color: Tuple[int, int, int, int]) -> None:
        rect = self._square_rect(square)
        overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        overlay.fill(color)
        self.screen.blit(overlay, rect.topleft)

    def _clear_selection(self) -> None:
        self.selected_sq = None
        self.selected_moves = []

    def _is_human_turn(self) -> bool:
        return self.board.side_to_move == self.human_color

    def _push_move(self, move: Move) -> None:
        self.board.push(move)
        self.move_history.append(move)
        self.last_move = move
        self.hint_move = None
        self.hint_explanation = ""
        self.move_list_auto_follow = True
        self._clear_selection()
        self._update_turn_status()

    def _classify_move_delta(self, delta: float) -> str:
        if delta <= MOVE_QUALITY_THRESHOLDS["best"]:
            return "best"
        if delta <= MOVE_QUALITY_THRESHOLDS["normal"]:
            return "normal"
        return "worst"

    def _record_session_result(self) -> None:
        self.session_games += 1
        self.total_games += 1
        if self.last_result == "user":
            self.session_user_wins += 1
            self.total_wins += 1
        elif self.last_result == "ai":
            self.session_ai_wins += 1
            self.total_losses += 1
        else:
            self.session_draws += 1
            self.total_draws += 1
        self._save_profile()

    def _search_best_move_with_budget(
        self,
        board_snapshot: Board,
        max_depth: int,
        budget_sec: float,
    ) -> Tuple[Optional[Move], float, int, int, float]:
        """
        Return best move found within strict wall-time budget.
        """
        start = time.time()
        budget = max(0.4, float(budget_sec))
        deadline = start + budget

        search_queue: mp.Queue = mp.Queue()
        proc = mp.Process(
            target=_timed_search_worker,
            args=(board_snapshot, max_depth, self.evaluator.model_path, self.use_neural_model, search_queue),
            daemon=True,
        )
        proc.start()

        best_depth = 0
        best_move: Optional[Move] = None
        best_score = 0.0
        best_nodes = 0

        while time.time() < deadline:
            timeout = min(0.05, max(0.0, deadline - time.time()))
            if timeout <= 0:
                break
            try:
                depth, move, score, nodes = search_queue.get(timeout=timeout)
            except Empty:
                if not proc.is_alive():
                    break
                continue

            if move is not None:
                best_depth = int(depth)
                best_move = move
                best_score = float(score)
                best_nodes = int(nodes)

        # Drain already-produced results quickly.
        while True:
            try:
                depth, move, score, nodes = search_queue.get_nowait()
            except Empty:
                break
            if move is not None:
                best_depth = int(depth)
                best_move = move
                best_score = float(score)
                best_nodes = int(nodes)

        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=0.2)
        search_queue.close()
        search_queue.cancel_join_thread()

        if best_move is None:
            fallback = find_best_move(board_snapshot.copy(), depth=1, evaluator=self.evaluator)
            best_move = fallback.move
            best_score = float(fallback.score)
            best_nodes = int(fallback.nodes)
            best_depth = 1 if fallback.move is not None else 0

        elapsed = time.time() - start
        return best_move, best_score, best_nodes, best_depth, elapsed

    def _build_postgame_analysis(
        self,
        move_history: Optional[List[Move]] = None,
        evaluator: Optional[object] = None,
        max_duration_sec: float = POSTGAME_ANALYSIS_BUDGET_SEC,
    ) -> Tuple[Dict[str, Dict[str, int]], List[Dict[str, object]]]:
        # Uses engine evaluation deltas per move to classify quality for both sides.
        stats_by_color: Dict[int, Dict[str, int]] = {
            WHITE: {"best": 0, "normal": 0, "worst": 0},
            BLACK: {"best": 0, "normal": 0, "worst": 0},
        }
        history = list(self.move_history) if move_history is None else list(move_history)
        if not history:
            return self._empty_postgame_stats(), []

        eval_model = self.evaluator if evaluator is None else evaluator
        analysis_depth = 1
        deadline = time.time() + max(0.6, float(max_duration_sec))
        analysis_board = Board()
        analysis_records: List[Dict[str, object]] = []
        for ply_index, played_move in enumerate(history, start=1):
            mover = analysis_board.side_to_move
            before_eval_white = float(eval_model.evaluate(analysis_board))
            best_move_uci = "-"

            if time.time() <= deadline:
                best_result = find_best_move(analysis_board.copy(), depth=analysis_depth, evaluator=eval_model)
                best_score = float(best_result.score) if best_result.move is not None else 0.0
                if best_result.move is not None:
                    best_move_uci = best_result.move.uci()

                analysis_board.push(played_move)
                if is_checkmate(analysis_board):
                    played_score = 1000.0
                else:
                    reply_result = find_best_move(analysis_board.copy(), depth=analysis_depth, evaluator=eval_model)
                    played_score = 0.0 if reply_result.move is None else -float(reply_result.score)
            else:
                # Budget fallback: classify from static evaluation drop so stats remain complete.
                best_score = float(mover) * float(eval_model.evaluate(analysis_board))
                analysis_board.push(played_move)
                played_score = float(mover) * float(eval_model.evaluate(analysis_board))

            after_eval_white = float(eval_model.evaluate(analysis_board))
            delta = max(0.0, best_score - played_score)
            quality = self._classify_move_delta(delta)
            stats_by_color[mover][quality] += 1
            analysis_records.append(
                {
                    "ply": ply_index,
                    "side": "White" if mover == WHITE else "Black",
                    "played": played_move.uci(),
                    "best": best_move_uci,
                    "eval_swing_white": round(after_eval_white - before_eval_white, 3),
                    "delta": round(delta, 3),
                    "quality": quality,
                }
            )

        return {"user": stats_by_color[self.human_color], "ai": stats_by_color[self.ai_color]}, analysis_records

    def _start_postgame_analysis(self) -> None:
        self.analysis_task_id += 1
        task_id = self.analysis_task_id
        history_snapshot = list(self.move_history)
        model_path = self.evaluator.model_path

        self.analysis_in_progress = True
        with self.analysis_lock:
            self.analysis_result_payload = None

        def worker() -> None:
            local_eval = MaterialEvaluator(model_path=model_path)
            stats, records = self._build_postgame_analysis(
                move_history=history_snapshot,
                evaluator=local_eval,
                max_duration_sec=POSTGAME_ANALYSIS_BUDGET_SEC,
            )
            with self.analysis_lock:
                self.analysis_result_payload = (task_id, stats, records)

        self.analysis_thread = threading.Thread(target=worker, daemon=True)
        self.analysis_thread.start()

    def _cancel_postgame_analysis(self) -> None:
        self.analysis_task_id += 1
        self.analysis_in_progress = False
        self.analysis_thread = None
        with self.analysis_lock:
            self.analysis_result_payload = None

    def _poll_postgame_analysis(self) -> None:
        if not self.analysis_in_progress:
            return

        with self.analysis_lock:
            payload = self.analysis_result_payload
        if payload is None:
            return

        task_id, stats, records = payload
        if task_id != self.analysis_task_id:
            with self.analysis_lock:
                self.analysis_result_payload = None
            return

        self.analysis_in_progress = False
        self.analysis_thread = None
        self.postgame_stats = stats
        self.analysis_records = records
        with self.analysis_lock:
            self.analysis_result_payload = None

    def _finish_game(self, state: str) -> None:
        if self.game_state != "playing":
            return
        if self.last_result not in {"user", "ai", "draw"}:
            self.last_result = "draw"
        self.game_state = state
        self.game_over = True
        self._record_session_result()
        self.editing_name = False
        self.popup_visible = True
        self.popup_anim = 0.0
        self.rematch_choice_visible = False
        self._leave_analysis_mode()
        self.analysis_records = []
        self.popup_notice = ""
        self.popup_notice_until = 0.0
        self.popup_continue_rect = None
        self.popup_switch_rect = None
        self._cancel_ai_search()
        self._cancel_postgame_analysis()
        self.postgame_stats = self._empty_postgame_stats()
        self._start_postgame_analysis()

    def _reset_end_state(self) -> None:
        self.game_state = "playing"
        self.game_over = False
        self.end_title = ""
        self.end_reason = ""
        self.popup_visible = False
        self.popup_anim = 0.0
        self.popup_close_rect = None
        self.popup_rematch_rect = None
        self.popup_continue_rect = None
        self.popup_switch_rect = None
        self.popup_analyze_rect = None
        self.popup_export_rect = None
        self.popup_analysis_prev_rect = None
        self.popup_analysis_next_rect = None
        self.popup_analysis_back_rect = None
        self.rematch_choice_visible = False
        self.analysis_mode_active = False
        self.analysis_records = []
        self.postgame_stats = self._empty_postgame_stats()
        self.last_result = ""
        self.analysis_view_board = self.board.copy()
        self._cancel_postgame_analysis()

    def _update_turn_status(self) -> None:
        if self.game_state != "playing":
            return

        if is_checkmate(self.board):
            winner_color = -self.board.side_to_move
            winner_name = self.user_name if winner_color == self.human_color else self.ai_name
            self.status_text = f"Checkmate! {winner_name} wins."
            self.end_title = "You Won" if winner_color == self.human_color else "You Lost"
            self.end_reason = "Checkmate"
            self.last_result = "user" if winner_color == self.human_color else "ai"
            self._finish_game("ended")
            return

        if is_stalemate(self.board):
            self.status_text = "Stalemate."
            self.end_title = "Draw"
            self.end_reason = "Stalemate"
            self.last_result = "draw"
            self._finish_game("ended")
            return

        side_name = self.user_name if self.board.side_to_move == self.human_color else self.ai_name
        check_suffix = " (check)" if is_in_check(self.board, self.board.side_to_move) else ""
        self.status_text = f"{side_name} to move{check_suffix}."

    def _resign(self) -> None:
        if self.game_state != "playing" or self.ai_thinking:
            return
        self.status_text = f"{self.user_name} resigned. {self.ai_name} wins."
        self.end_title = "You Lost"
        self.end_reason = "Resigned"
        self.last_result = "ai"
        self._finish_game("resigned")

    def _start_rematch(self) -> None:
        self._cancel_ai_search()
        self._cancel_postgame_analysis()

        self.board = Board()
        self.move_history = []
        self.selected_sq = None
        self.selected_moves = []
        self.promotion = None
        self.last_move = None
        self.hint_move = None

        self.game_state = "playing"
        self.game_over = False
        self.end_title = ""
        self.end_reason = ""
        self.last_result = ""
        self.popup_visible = False
        self.popup_anim = 0.0
        self.popup_close_rect = None
        self.popup_rematch_rect = None
        self.popup_continue_rect = None
        self.popup_switch_rect = None
        self.popup_analyze_rect = None
        self.popup_export_rect = None
        self.popup_analysis_prev_rect = None
        self.popup_analysis_next_rect = None
        self.popup_analysis_back_rect = None
        self.rematch_choice_visible = False
        self.analysis_mode_active = False
        self.analysis_records = []
        self.postgame_stats = self._empty_postgame_stats()
        self.move_list_scroll = 0
        self.move_list_auto_follow = True
        self.ai_last_depth = 0
        self.ai_last_nodes = 0
        self.ai_last_eval_white = 0.0
        self.ai_last_time = 0.0
        self.editing_name = False
        self.move_feedback_text = ""
        self.move_feedback_until = 0.0
        self.popup_notice = ""
        self.popup_notice_until = 0.0
        self.hint_explanation = ""
        self.analysis_view_board = self.board.copy()
        self._save_profile()

        self._update_turn_status()

    def _begin_game_as(self, human_color: int) -> None:
        self.human_color = human_color
        self.ai_color = -human_color
        self.awaiting_start_choice = False
        self.editing_name = False
        self._save_profile()
        self._update_turn_status()

    def _pop_one_move(self) -> bool:
        if not self.move_history:
            return False
        self.board.pop()
        self.move_history.pop()
        self.last_move = self.move_history[-1] if self.move_history else None
        return True

    def _undo_last_turn(self) -> None:
        if self.ai_thinking:
            return

        undone = self._pop_one_move()
        if self.move_history and self.board.side_to_move != self.human_color:
            undone = self._pop_one_move() or undone

        if not undone:
            self.status_text = "No moves to undo."
            return

        self._cancel_ai_search()
        self._clear_selection()
        self.promotion = None
        self.hint_move = None
        self.move_list_auto_follow = True
        self._reset_end_state()
        self._update_turn_status()

    def _show_hint(self) -> None:
        if self.game_state != "playing":
            return
        if self.promotion is not None or not self._is_human_turn() or self.ai_thinking:
            return

        board_before = self.board.copy()
        move, _score, _nodes, depth_used, elapsed = self._search_best_move_with_budget(
            board_before,
            max_depth=self.ai_depth,
            budget_sec=min(1.2, self.ai_move_time_limit),
        )
        self.hint_move = move
        if self.hint_move is None:
            self.status_text = "No hint available."
            self.hint_explanation = ""
        else:
            self.status_text = f"Hint: {self.hint_move.uci()}  (d{depth_used}, {elapsed:.2f}s)"
            self.hint_explanation = self._build_hint_explanation(board_before, self.hint_move)

    def _apply_promotion(self, key: int) -> None:
        if self.promotion is None:
            return

        wanted = None
        if key == pygame.K_q:
            wanted = QUEEN
        elif key == pygame.K_r:
            wanted = ROOK
        elif key == pygame.K_b:
            wanted = BISHOP
        elif key == pygame.K_n:
            wanted = KNIGHT
        else:
            return

        for move in self.promotion.options:
            if abs(move.promotion or 0) == wanted:
                before = self.board.copy()
                self.promotion = None
                self._push_move(move)
                delta = self._evaluate_move_delta(before, move, depth=1)
                self._set_move_feedback_from_delta(delta)
                return

    def _try_make_human_move(self, from_sq: int, to_sq: int) -> None:
        if not self._is_human_turn() or self.game_state != "playing":
            return

        moves = legal_moves_from(self.board, from_sq)
        candidates = [m for m in moves if m.to_sq == to_sq]
        if not candidates:
            return

        promo_moves = [m for m in candidates if m.promotion is not None]
        if promo_moves:
            self.promotion = PromotionChoice(from_sq, to_sq, promo_moves)
            self.status_text = "Promotion: press Q / R / B / N."
            return

        selected_move = candidates[0]
        before = self.board.copy()
        self._push_move(selected_move)
        delta = self._evaluate_move_delta(before, selected_move, depth=1)
        self._set_move_feedback_from_delta(delta)

    def _start_ai_search(self) -> None:
        if self.game_state != "playing":
            return
        if self.awaiting_start_choice:
            return
        if self.promotion is not None:
            return
        if self.board.side_to_move != self.ai_color:
            return
        if self.ai_thinking:
            return

        self.ai_task_id += 1
        task_id = self.ai_task_id
        board_snapshot = self.board.copy()
        evaluator = self.evaluator
        depth = self.ai_depth

        self.ai_thinking = True
        self.ai_start_ts = time.time()
        with self.ai_result_lock:
            self.ai_result_payload = None

        def worker() -> None:
            move, score, nodes, best_depth, elapsed = self._search_best_move_with_budget(
                board_snapshot=board_snapshot,
                max_depth=depth,
                budget_sec=self.ai_move_time_limit,
            )
            with self.ai_result_lock:
                self.ai_result_payload = (task_id, move, score, nodes, elapsed, best_depth)

        self.ai_thread = threading.Thread(target=worker, daemon=True)
        self.ai_thread.start()

    def _cancel_ai_search(self) -> None:
        self.ai_task_id += 1
        self.ai_thinking = False
        self.ai_thread = None
        with self.ai_result_lock:
            self.ai_result_payload = None

    def _poll_ai_result(self) -> None:
        if not self.ai_thinking:
            return

        with self.ai_result_lock:
            payload = self.ai_result_payload
        if payload is None:
            return

        task_id, move, score, nodes, elapsed, depth_used = payload
        if task_id != self.ai_task_id:
            with self.ai_result_lock:
                self.ai_result_payload = None
            return

        self.ai_thinking = False
        self.ai_thread = None
        self.ai_last_time = elapsed
        self.ai_last_depth = depth_used
        self.ai_last_nodes = int(nodes)
        self.ai_last_eval_white = float(score) * float(self.ai_color)
        with self.ai_result_lock:
            self.ai_result_payload = None

        if self.game_state != "playing":
            return
        if self.board.side_to_move != self.ai_color:
            return
        if move is None:
            self._update_turn_status()
            return

        self._push_move(move)

    def _handle_name_edit_key(self, key: int, unicode_char: str) -> None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            cleaned = self.user_name.strip()
            self.user_name = cleaned if cleaned else "Soprano"
            self.editing_name = False
            self._save_profile()
            self._update_turn_status()
            return
        if key == pygame.K_ESCAPE:
            if not self.user_name.strip():
                self.user_name = "Soprano"
            self.editing_name = False
            return
        if key == pygame.K_BACKSPACE:
            self.user_name = self.user_name[:-1]
            return
        if len(self.user_name) >= 16:
            return
        if unicode_char and unicode_char.isprintable():
            self.user_name += unicode_char

    def _is_button_enabled(self, key: str) -> bool:
        if key == "undo":
            return bool(self.move_history) and not self.ai_thinking
        if key == "hint":
            return self.game_state == "playing" and self._is_human_turn() and self.promotion is None and not self.ai_thinking
        if key == "resign":
            return self.game_state == "playing" and not self.ai_thinking
        return True

    def _open_rematch_choice(self) -> None:
        if not self.popup_visible or self.analysis_mode_active:
            return
        self.rematch_choice_visible = True
        self.popup_continue_rect = None
        self.popup_switch_rect = None

    def _start_rematch_with_side_choice(self, switch_side: bool) -> None:
        if switch_side:
            self.human_color = -self.human_color
            self.ai_color = -self.human_color
        self._start_rematch()

    def _handle_move_list_scroll(self, wheel_y: int) -> bool:
        if wheel_y == 0:
            return False
        if not self.move_list_rect.collidepoint(pygame.mouse.get_pos()):
            return False

        total_rows = (len(self.move_history) + 1) // 2
        content_height = max(1, self.move_list_rect.height - 50)
        visible_rows = max(1, content_height // self.move_list_row_height)
        max_scroll = max(0, total_rows - visible_rows)

        self.move_list_scroll = max(0, min(max_scroll, self.move_list_scroll - int(wheel_y)))
        self.move_list_auto_follow = self.move_list_scroll >= max_scroll
        return True

    def _handle_panel_click(self, pos: Tuple[int, int]) -> bool:
        if self.user_name_input_rect.collidepoint(pos):
            self.editing_name = True
            return True

        for key, button in self.buttons.items():
            if not button.rect.collidepoint(pos):
                continue
            if not self._is_button_enabled(key):
                return True
            if key == "undo":
                self._undo_last_turn()
            elif key == "hint":
                self._show_hint()
            elif key == "resign":
                self._resign()
            return True
        return False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            raise SystemExit

        if event.type == pygame.MOUSEWHEEL:
            if self._handle_move_list_scroll(event.y):
                return

        if event.type == pygame.KEYDOWN:
            if self.editing_name:
                self._handle_name_edit_key(event.key, event.unicode)
                return
            if self.awaiting_start_choice:
                if event.key == pygame.K_w:
                    self._begin_game_as(WHITE)
                    return
                if event.key == pygame.K_b:
                    self._begin_game_as(BLACK)
                    return
                return
            if event.key == pygame.K_ESCAPE:
                if self.popup_visible:
                    if self.analysis_mode_active:
                        self._leave_analysis_mode()
                    else:
                        self.popup_visible = False
                        self.rematch_choice_visible = False
                        self.popup_continue_rect = None
                        self.popup_switch_rect = None
                self.promotion = None
                self.hint_move = None
                self._clear_selection()
                return
            if self.popup_visible and self.analysis_mode_active:
                if event.key == pygame.K_LEFT:
                    self._set_analysis_index(self.analysis_view_index - 1)
                    return
                if event.key == pygame.K_RIGHT:
                    self._set_analysis_index(self.analysis_view_index + 1)
                    return
                if event.key == pygame.K_r:
                    return
                if event.key in (pygame.K_a, pygame.K_TAB):
                    self._leave_analysis_mode()
                    return
            if self.popup_visible and event.key == pygame.K_r:
                if self.rematch_choice_visible:
                    self._start_rematch_with_side_choice(False)
                else:
                    self._open_rematch_choice()
                return
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.popup_visible:
                if self.rematch_choice_visible:
                    self._start_rematch_with_side_choice(False)
                else:
                    self.popup_visible = False
                    self.rematch_choice_visible = False
                    self.popup_continue_rect = None
                    self.popup_switch_rect = None
                return
            self._apply_promotion(event.key)
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (4, 5):
            wheel_y = 1 if event.button == 4 else -1
            if self._handle_move_list_scroll(wheel_y):
                return

        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return

        pos = event.pos
        if self.awaiting_start_choice:
            if self.setup_name_input_rect.collidepoint(pos):
                self.editing_name = True
                return
            self.editing_name = False
            if self.setup_white_rect.collidepoint(pos):
                self._begin_game_as(WHITE)
                return
            if self.setup_black_rect.collidepoint(pos):
                self._begin_game_as(BLACK)
                return
            return

        if self.popup_visible:
            if self.analysis_mode_active:
                if self.popup_analysis_prev_rect is not None and self.popup_analysis_prev_rect.collidepoint(pos):
                    self._set_analysis_index(self.analysis_view_index - 1)
                    return
                if self.popup_analysis_next_rect is not None and self.popup_analysis_next_rect.collidepoint(pos):
                    self._set_analysis_index(self.analysis_view_index + 1)
                    return
                if self.popup_analysis_back_rect is not None and self.popup_analysis_back_rect.collidepoint(pos):
                    self._leave_analysis_mode()
                    return
            else:
                if self.rematch_choice_visible:
                    if self.popup_continue_rect is not None and self.popup_continue_rect.collidepoint(pos):
                        self._start_rematch_with_side_choice(False)
                        return
                    if self.popup_switch_rect is not None and self.popup_switch_rect.collidepoint(pos):
                        self._start_rematch_with_side_choice(True)
                        return
                else:
                    if self.popup_analyze_rect is not None and self.popup_analyze_rect.collidepoint(pos):
                        self._enter_analysis_mode()
                        return
                    if self.popup_export_rect is not None and self.popup_export_rect.collidepoint(pos):
                        out_path = self._export_game_pgn()
                        if out_path is None:
                            self._set_popup_notice("PGN export failed.")
                        else:
                            self._set_popup_notice(f"Exported: {out_path.name}")
                        return
                    if self.popup_rematch_rect is not None and self.popup_rematch_rect.collidepoint(pos):
                        self._open_rematch_choice()
                        return
            if self.popup_close_rect is not None and self.popup_close_rect.collidepoint(pos):
                self.popup_visible = False
                self.rematch_choice_visible = False
                self.popup_continue_rect = None
                self.popup_switch_rect = None
                self._leave_analysis_mode()
                return
            return
        if self._handle_panel_click(pos):
            return

        if self.editing_name:
            self.editing_name = False
        if self.game_state != "playing":
            return
        if self.promotion is not None or not self._is_human_turn() or self.ai_thinking:
            return

        sq = self._square_from_mouse(pos)
        if sq is None:
            return

        if self.selected_sq is None:
            piece = self.board.piece_at(sq)
            if piece != EMPTY and piece_color(piece) == self.human_color:
                self.selected_sq = sq
                self.selected_moves = legal_moves_from(self.board, sq)
            return

        if sq == self.selected_sq:
            self._clear_selection()
            return

        self._try_make_human_move(self.selected_sq, sq)
        if self.board.side_to_move == self.human_color and self.game_state == "playing":
            piece = self.board.piece_at(sq)
            if piece != EMPTY and piece_color(piece) == self.human_color:
                self.selected_sq = sq
                self.selected_moves = legal_moves_from(self.board, sq)

    def _draw_board(self) -> None:
        pygame.draw.rect(self.screen, BOARD_BORDER, self.board_rect.inflate(10, 10), border_radius=8)
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                rect = pygame.Rect(self.board_rect.x + col * self.square_size, self.board_rect.y + row * self.square_size, self.square_size, self.square_size)
                pygame.draw.rect(self.screen, color, rect)

    def _draw_last_move(self) -> None:
        move_to_draw: Optional[Move] = self.last_move
        if self.popup_visible and self.analysis_mode_active:
            if self.analysis_view_index <= 0:
                move_to_draw = None
            elif self.analysis_view_index - 1 < len(self.move_history):
                move_to_draw = self.move_history[self.analysis_view_index - 1]
        if move_to_draw is None:
            return
        self._draw_square_overlay(move_to_draw.from_sq, LAST_MOVE_FROM)
        self._draw_square_overlay(move_to_draw.to_sq, LAST_MOVE_TO)

    def _draw_pieces(self) -> None:
        board_view = self._render_board()
        for row in range(8):
            for col in range(8):
                piece = int(board_view.squares[row, col])
                if piece == EMPTY:
                    continue
                image = self.piece_images.get(piece)
                if image is None:
                    continue
                square = row * 8 + col
                self.screen.blit(image, image.get_rect(center=self._square_rect(square).center))

    def _draw_selected_square(self) -> None:
        if self.selected_sq is None or (self.popup_visible and self.analysis_mode_active):
            return
        pygame.draw.rect(self.screen, SELECTED_BORDER, self._square_rect(self.selected_sq), width=4)

    def _draw_legal_move_hints(self) -> None:
        if self.popup_visible and self.analysis_mode_active:
            return
        board_view = self._render_board()
        for move in self.selected_moves:
            rect = self._square_rect(move.to_sq)
            target = board_view.piece_at(move.to_sq)
            color = LEGAL_CAPTURE_DOT if (target != EMPTY or move.is_en_passant) else LEGAL_MOVE_DOT
            pygame.draw.circle(self.screen, color, rect.center, self.square_size // 10)

    def _draw_hint_arrow(self) -> None:
        if self.hint_move is None or self.game_state != "playing" or (self.popup_visible and self.analysis_mode_active):
            return

        sx, sy = self._square_center(self.hint_move.from_sq)
        ex, ey = self._square_center(self.hint_move.to_sq)
        dx = ex - sx
        dy = ey - sy
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return

        ux = dx / dist
        uy = dy / dist
        start_pad = self.square_size * 0.18
        end_pad = self.square_size * 0.22
        line_start = (sx + ux * start_pad, sy + uy * start_pad)
        line_end = (ex - ux * end_pad, ey - uy * end_pad)

        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(overlay, HINT_ARROW_COLOR, (int(line_start[0]), int(line_start[1])), (int(line_end[0]), int(line_end[1])), 12)

        head_len = 24
        head_w = 16
        tip = (line_end[0], line_end[1])
        left = (tip[0] - ux * head_len + uy * head_w * 0.5, tip[1] - uy * head_len - ux * head_w * 0.5)
        right = (tip[0] - ux * head_len - uy * head_w * 0.5, tip[1] - uy * head_len + ux * head_w * 0.5)
        pygame.draw.polygon(overlay, HINT_ARROW_COLOR, [(int(tip[0]), int(tip[1])), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))])
        self.screen.blit(overlay, (0, 0))

    def _draw_coordinates(self) -> None:
        if self._is_board_flipped():
            files = "hgfedcba"
            ranks = [str(row + 1) for row in range(8)]
        else:
            files = "abcdefgh"
            ranks = [str(8 - row) for row in range(8)]

        for col in range(8):
            label = self.font_board.render(files[col], True, (226, 238, 245))
            self.screen.blit(label, (self.board_rect.x + col * self.square_size + self.square_size - 14, self.board_rect.bottom - 18))

        for row in range(8):
            label = self.font_board.render(ranks[row], True, (226, 238, 245))
            self.screen.blit(label, (self.board_rect.x + 6, self.board_rect.y + row * self.square_size + 6))

    def _draw_player_cards(self) -> None:
        pygame.draw.rect(self.screen, CARD_BG, self.ai_panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, CARD_BG, self.user_panel_rect, border_radius=10)

        ai_avatar_y = self.ai_panel_rect.y + (self.ai_panel_rect.height - self.cortex_avatar.get_height()) // 2
        self.screen.blit(self.cortex_avatar, (self.ai_panel_rect.x + 8, ai_avatar_y))

        ai_label = self.font_player.render(self.ai_name, True, TEXT_PRIMARY)
        ai_label_y = self.ai_panel_rect.y + (self.ai_panel_rect.height - ai_label.get_height()) // 2
        self.screen.blit(ai_label, (self.ai_panel_rect.x + 56, ai_label_y))

        if self.ai_thinking and self.game_state == "playing":
            dots = "." * ((int(time.time() * 3) % 3) + 1)
            thinking = self.font_small.render(f"Thinking{dots}", True, ACCENT)
            self.screen.blit(
                thinking,
                (
                    self.ai_panel_rect.right - thinking.get_width() - 14,
                    self.ai_panel_rect.y + (self.ai_panel_rect.height - thinking.get_height()) // 2,
                ),
            )
        else:
            ready = self.font_small.render("Ready", True, TEXT_MUTED)
            self.screen.blit(
                ready,
                (
                    self.ai_panel_rect.right - ready.get_width() - 14,
                    self.ai_panel_rect.y + (self.ai_panel_rect.height - ready.get_height()) // 2,
                ),
            )

        user_avatar = self._get_user_avatar(40)
        user_avatar_y = self.user_panel_rect.y + (self.user_panel_rect.height - user_avatar.get_height()) // 2
        self.screen.blit(user_avatar, (self.user_panel_rect.x + 8, user_avatar_y))
        pygame.draw.rect(self.screen, (14, 58, 82), self.user_name_input_rect, border_radius=7)
        pygame.draw.rect(self.screen, ACCENT if self.editing_name else (64, 115, 141), self.user_name_input_rect, width=2, border_radius=7)

        display_name = self.user_name if self.user_name else "Soprano"
        caret = "|" if self.editing_name and int(time.time() * 2) % 2 == 0 else ""
        label = self.font_player.render(display_name + (caret if self.editing_name else ""), True, TEXT_PRIMARY)
        self.screen.blit(
            label,
            (
                self.user_name_input_rect.x + 8,
                self.user_name_input_rect.y + (self.user_name_input_rect.height - label.get_height()) // 2,
            ),
        )

    def _draw_button(self, button: UIButton, enabled: bool) -> None:
        mouse = pygame.mouse.get_pos()
        hover = button.rect.collidepoint(mouse)
        if not enabled:
            bg = BUTTON_BG_DISABLED
            fg = (142, 161, 172)
        elif hover:
            bg = BUTTON_BG_HOVER
            fg = (248, 252, 255)
        else:
            bg = BUTTON_BG
            fg = (236, 245, 250)

        pygame.draw.rect(self.screen, bg, button.rect, border_radius=12)
        pygame.draw.rect(self.screen, (31, 96, 124), button.rect, width=2, border_radius=12)
        text = self.font_button.render(button.label, True, fg)
        self.screen.blit(text, text.get_rect(center=button.rect.center))

    def _draw_move_list(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, (8, 42, 60, 170), rect, border_radius=10)
        pygame.draw.rect(self.screen, (36, 88, 112), rect, width=1, border_radius=10)

        self.screen.blit(self.font_body.render("Move List", True, TEXT_PRIMARY), (rect.x + 12, rect.y + 10))

        rows = []
        for i in range(0, len(self.move_history), 2):
            move_no = i // 2 + 1
            white_move = self.move_history[i].uci()
            black_move = self.move_history[i + 1].uci() if i + 1 < len(self.move_history) else ""
            rows.append((move_no, white_move, black_move))

        content_rect = pygame.Rect(rect.x + 10, rect.y + 42, rect.width - 20, rect.height - 50)
        visible_rows = max(1, content_rect.height // self.move_list_row_height)
        max_scroll = max(0, len(rows) - visible_rows)
        if self.move_list_auto_follow:
            self.move_list_scroll = max_scroll
        else:
            self.move_list_scroll = max(0, min(self.move_list_scroll, max_scroll))

        start = self.move_list_scroll
        end = min(len(rows), start + visible_rows)
        clip_prev = self.screen.get_clip()
        self.screen.set_clip(content_rect)
        y = content_rect.y + 2
        for row_index in range(start, end):
            move_no, white_move, black_move = rows[row_index]
            text = self.font_small.render(f"{move_no:>2}. {white_move:<7} {black_move:<7}", True, TEXT_MUTED)
            self.screen.blit(text, (content_rect.x + 2, y))
            y += self.move_list_row_height
        self.screen.set_clip(clip_prev)

    def _draw_sidebar(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, self.right_panel_rect, border_radius=14)
        pygame.draw.rect(self.screen, (30, 92, 120), self.right_panel_rect, width=2, border_radius=14)

        self.screen.blit(self.font_panel_title.render("Cortex64", True, TEXT_PRIMARY), (self.right_panel_rect.x + 20, self.right_panel_rect.y + 16))
        subtitle = self.font_small.render(f"Single Player vs AI  |  Eval: {self.eval_label}", True, TEXT_MUTED)
        self.screen.blit(subtitle, (self.right_panel_rect.x + 22, self.right_panel_rect.y + 52))

        score_rect = pygame.Rect(self.right_panel_rect.x + 20, self.right_panel_rect.y + 84, self.right_panel_rect.width - 40, 76)
        pygame.draw.rect(self.screen, (10, 50, 70, 185), score_rect, border_radius=10)
        score_title = self.font_small.render(f"Games this run: {self.session_games}", True, TEXT_PRIMARY)
        score_line = self.font_small.render(
            f"{self.user_name}: {self.session_user_wins}  {self.ai_name}: {self.session_ai_wins}  Draw: {self.session_draws}",
            True,
            TEXT_MUTED,
        )
        total_line = self.font_small.render(
            f"Total W/L/D: {self.total_wins}/{self.total_losses}/{self.total_draws}",
            True,
            TEXT_MUTED,
        )
        self.screen.blit(score_title, (score_rect.x + 10, score_rect.y + 8))
        self.screen.blit(score_line, (score_rect.x + 10, score_rect.y + 30))
        self.screen.blit(total_line, (score_rect.x + 10, score_rect.y + 50))

        status_rect = pygame.Rect(self.right_panel_rect.x + 20, self.right_panel_rect.y + 168, self.right_panel_rect.width - 40, 52)
        pygame.draw.rect(self.screen, (10, 50, 70, 185), status_rect, border_radius=10)
        status = self.font_body.render(self.status_text, True, TEXT_PRIMARY)
        self.screen.blit(status, (status_rect.x + 10, status_rect.y + 14))

        if self.ai_thinking and self.game_state == "playing":
            info = self.font_small.render(f"{self.ai_name} search: {time.time() - self.ai_start_ts:.1f}s", True, ACCENT)
        else:
            info = self.font_small.render(f"Last AI move: {self.ai_last_time:.2f}s  d{self.ai_last_depth}", True, TEXT_MUTED)
        self.screen.blit(info, (self.right_panel_rect.x + 22, self.right_panel_rect.y + 228))

        stats_rect = pygame.Rect(self.right_panel_rect.x + 20, self.right_panel_rect.y + 252, self.right_panel_rect.width - 40, 74)
        pygame.draw.rect(self.screen, (10, 50, 70, 185), stats_rect, border_radius=10)
        pygame.draw.rect(self.screen, (36, 88, 112), stats_rect, width=1, border_radius=10)
        depth_txt = self.font_small.render(f"Depth: {self.ai_last_depth}    Nodes: {self.ai_last_nodes}", True, TEXT_PRIMARY)
        time_txt = self.font_small.render(f"Time: {self.ai_last_time:.2f}s", True, TEXT_MUTED)
        eval_txt = self.font_small.render(f"Eval (White): {self.ai_last_eval_white:+.2f}", True, TEXT_MUTED)
        self.screen.blit(depth_txt, (stats_rect.x + 10, stats_rect.y + 8))
        self.screen.blit(time_txt, (stats_rect.x + 10, stats_rect.y + 30))
        self.screen.blit(eval_txt, (stats_rect.x + 130, stats_rect.y + 30))

        bar_rect = pygame.Rect(stats_rect.x + 10, stats_rect.y + 52, stats_rect.width - 20, 10)
        pygame.draw.rect(self.screen, (27, 68, 90), bar_rect, border_radius=5)
        center_x = bar_rect.x + bar_rect.width // 2
        pygame.draw.line(self.screen, (96, 142, 166), (center_x, bar_rect.y), (center_x, bar_rect.bottom), 1)
        clamped_eval = max(-3.0, min(3.0, self.ai_last_eval_white))
        fill = int((clamped_eval / 3.0) * (bar_rect.width // 2))
        if fill >= 0:
            fill_rect = pygame.Rect(center_x, bar_rect.y, fill, bar_rect.height)
            fill_color = (120, 204, 132)
        else:
            fill_rect = pygame.Rect(center_x + fill, bar_rect.y, -fill, bar_rect.height)
            fill_color = (214, 124, 110)
        pygame.draw.rect(self.screen, fill_color, fill_rect, border_radius=5)

        coach_rect = pygame.Rect(self.right_panel_rect.x + 20, self.right_panel_rect.y + 332, self.right_panel_rect.width - 40, 30)
        pygame.draw.rect(self.screen, (10, 45, 64, 175), coach_rect, border_radius=8)
        coach_text = ""
        coach_color = TEXT_MUTED
        if time.time() < self.move_feedback_until:
            coach_text = self.move_feedback_text
            coach_color = self.move_feedback_color
        elif self.hint_explanation:
            coach_text = self.hint_explanation
            coach_color = ACCENT
        if coach_text:
            trimmed = coach_text
            while self.font_small.size(trimmed)[0] > coach_rect.width - 14 and len(trimmed) > 4:
                trimmed = trimmed[:-4] + "..."
            coach_surf = self.font_small.render(trimmed, True, coach_color)
            self.screen.blit(coach_surf, (coach_rect.x + 8, coach_rect.y + 6))

        self.move_list_rect = self._recalculate_move_list_rect()
        self._draw_move_list(self.move_list_rect)
        for key, button in self.buttons.items():
            self._draw_button(button, self._is_button_enabled(key))

    def _draw_postgame_popup(self) -> None:
        if not self.popup_visible:
            self.popup_close_rect = None
            self.popup_rematch_rect = None
            self.popup_continue_rect = None
            self.popup_switch_rect = None
            self.popup_analyze_rect = None
            self.popup_export_rect = None
            self.popup_analysis_prev_rect = None
            self.popup_analysis_next_rect = None
            self.popup_analysis_back_rect = None
            return

        self.popup_anim = min(1.0, self.popup_anim + 0.08)
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 130))
        self.screen.blit(overlay, (0, 0))

        scale = 0.92 + 0.08 * self.popup_anim
        popup_rect = pygame.Rect(0, 0, int(540 * scale), int(392 * scale))
        popup_rect.center = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
        popup_notice_y = popup_rect.bottom - 84

        panel = pygame.Surface((popup_rect.width, popup_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, (10, 40, 56, 236), panel.get_rect(), border_radius=14)
        pygame.draw.rect(panel, (65, 140, 172, 220), panel.get_rect(), width=2, border_radius=14)
        self.screen.blit(panel, popup_rect.topleft)

        if self.analysis_mode_active:
            self.popup_rematch_rect = None
            self.popup_continue_rect = None
            self.popup_switch_rect = None
            self.popup_analyze_rect = None
            self.popup_export_rect = None

            self.screen.blit(self.font_title.render("Game Analysis", True, TEXT_PRIMARY), (popup_rect.x + 24, popup_rect.y + 20))
            self.screen.blit(
                self.font_small.render("Use Left/Right keys or buttons to step through moves.", True, TEXT_MUTED),
                (popup_rect.x + 26, popup_rect.y + 68),
            )

            total = len(self.move_history)
            idx = self.analysis_view_index
            self.screen.blit(
                self.font_body.render(f"Ply: {idx}/{total}", True, TEXT_PRIMARY),
                (popup_rect.x + 26, popup_rect.y + 102),
            )

            if self.analysis_in_progress and not self.analysis_records:
                self.screen.blit(
                    self.font_body.render("Preparing analysis...", True, ACCENT),
                    (popup_rect.x + 26, popup_rect.y + 144),
                )
                self.screen.blit(
                    self.font_small.render("Records will appear automatically.", True, TEXT_MUTED),
                    (popup_rect.x + 26, popup_rect.y + 176),
                )
            else:
                record: Optional[Dict[str, object]] = None
                if idx > 0 and idx - 1 < len(self.analysis_records):
                    record = self.analysis_records[idx - 1]
                if record is None:
                    self.screen.blit(
                        self.font_small.render("At game start position.", True, TEXT_MUTED),
                        (popup_rect.x + 26, popup_rect.y + 144),
                    )
                else:
                    played = str(record.get("played", "-"))
                    best = str(record.get("best", "-"))
                    side = str(record.get("side", ""))
                    swing = float(record.get("eval_swing_white", 0.0))
                    quality = str(record.get("quality", "normal")).capitalize()
                    self.screen.blit(self.font_body.render(f"{side} played: {played}", True, TEXT_PRIMARY), (popup_rect.x + 26, popup_rect.y + 140))
                    self.screen.blit(self.font_body.render(f"Best alternative: {best}", True, TEXT_PRIMARY), (popup_rect.x + 26, popup_rect.y + 174))
                    self.screen.blit(self.font_body.render(f"Eval swing (White): {swing:+.2f}", True, TEXT_PRIMARY), (popup_rect.x + 26, popup_rect.y + 208))
                    self.screen.blit(self.font_small.render(f"Move quality: {quality}", True, TEXT_MUTED), (popup_rect.x + 26, popup_rect.y + 242))

            button_y = popup_rect.bottom - 22
            self.popup_analysis_prev_rect = pygame.Rect(0, 0, 128, 42)
            self.popup_analysis_prev_rect.midbottom = (popup_rect.centerx - 132, button_y)
            pygame.draw.rect(self.screen, BUTTON_BG, self.popup_analysis_prev_rect, border_radius=10)
            pygame.draw.rect(self.screen, (54, 134, 170), self.popup_analysis_prev_rect, width=2, border_radius=10)
            self.screen.blit(self.font_button.render("Prev", True, TEXT_PRIMARY), self.font_button.render("Prev", True, TEXT_PRIMARY).get_rect(center=self.popup_analysis_prev_rect.center))

            self.popup_analysis_next_rect = pygame.Rect(0, 0, 128, 42)
            self.popup_analysis_next_rect.midbottom = (popup_rect.centerx, button_y)
            pygame.draw.rect(self.screen, BUTTON_BG_HOVER, self.popup_analysis_next_rect, border_radius=10)
            pygame.draw.rect(self.screen, (54, 164, 170), self.popup_analysis_next_rect, width=2, border_radius=10)
            self.screen.blit(self.font_button.render("Next", True, TEXT_PRIMARY), self.font_button.render("Next", True, TEXT_PRIMARY).get_rect(center=self.popup_analysis_next_rect.center))

            self.popup_analysis_back_rect = pygame.Rect(0, 0, 128, 42)
            self.popup_analysis_back_rect.midbottom = (popup_rect.centerx + 132, button_y)
            pygame.draw.rect(self.screen, BUTTON_BG, self.popup_analysis_back_rect, border_radius=10)
            pygame.draw.rect(self.screen, (54, 134, 170), self.popup_analysis_back_rect, width=2, border_radius=10)
            self.screen.blit(self.font_button.render("Back", True, TEXT_PRIMARY), self.font_button.render("Back", True, TEXT_PRIMARY).get_rect(center=self.popup_analysis_back_rect.center))

            self.popup_close_rect = pygame.Rect(0, 0, 94, 34)
            self.popup_close_rect.topright = (popup_rect.right - 16, popup_rect.y + 16)
            pygame.draw.rect(self.screen, BUTTON_BG, self.popup_close_rect, border_radius=8)
            pygame.draw.rect(self.screen, (54, 134, 170), self.popup_close_rect, width=2, border_radius=8)
            close_txt = self.font_small.render("Close", True, TEXT_PRIMARY)
            self.screen.blit(close_txt, close_txt.get_rect(center=self.popup_close_rect.center))
        else:
            self.popup_analysis_prev_rect = None
            self.popup_analysis_next_rect = None
            self.popup_analysis_back_rect = None

            self.screen.blit(self.font_title.render(self.end_title or "Game Ended", True, TEXT_PRIMARY), (popup_rect.x + 24, popup_rect.y + 20))
            self.screen.blit(self.font_body.render(f"Reason: {self.end_reason}", True, TEXT_MUTED), (popup_rect.x + 26, popup_rect.y + 70))
            self.screen.blit(self.font_small.render(f"Total half-moves: {len(self.move_history)}", True, TEXT_MUTED), (popup_rect.x + 26, popup_rect.y + 102))
            session_line = (
                f"Session Score  {self.user_name}: {self.session_user_wins}  "
                f"{self.ai_name}: {self.session_ai_wins}  Draw: {self.session_draws}"
            )
            self.screen.blit(self.font_small.render(session_line, True, TEXT_MUTED), (popup_rect.x + 26, popup_rect.y + 124))

            if self.analysis_in_progress:
                self.screen.blit(self.font_body.render("Analyzing move quality...", True, ACCENT), (popup_rect.x + 26, popup_rect.y + 156))
                self.screen.blit(self.font_small.render("The game is finished. Stats are loading in background.", True, TEXT_MUTED), (popup_rect.x + 26, popup_rect.y + 194))
            else:
                user = self.postgame_stats["user"]
                ai = self.postgame_stats["ai"]
                user_line = f"{self.user_name}  Best: {user['best']}  Normal: {user['normal']}  Worst: {user['worst']}"
                ai_line = f"{self.ai_name}  Best: {ai['best']}  Normal: {ai['normal']}  Worst: {ai['worst']}"
                self.screen.blit(self.font_body.render(user_line, True, TEXT_PRIMARY), (popup_rect.x + 26, popup_rect.y + 156))
                self.screen.blit(self.font_body.render(ai_line, True, TEXT_PRIMARY), (popup_rect.x + 26, popup_rect.y + 192))
                self.screen.blit(self.font_small.render("Move quality uses AI evaluation deltas.", True, TEXT_MUTED), (popup_rect.x + 26, popup_rect.y + 228))

            if self.rematch_choice_visible:
                self.popup_analyze_rect = None
                self.popup_export_rect = None
                self.popup_rematch_rect = None
                button_y = popup_rect.bottom - 22
                prompt = self.font_body.render("Rematch: Continue or Switch side?", True, TEXT_PRIMARY)
                prompt_y = button_y - 92
                self.screen.blit(prompt, (popup_rect.x + 26, prompt_y))

                self.popup_continue_rect = pygame.Rect(0, 0, 170, 42)
                self.popup_continue_rect.midbottom = (popup_rect.centerx - 96, button_y)
                pygame.draw.rect(self.screen, BUTTON_BG_HOVER, self.popup_continue_rect, border_radius=10)
                pygame.draw.rect(self.screen, (54, 164, 170), self.popup_continue_rect, width=2, border_radius=10)
                continue_txt = self.font_button.render("Continue", True, TEXT_PRIMARY)
                self.screen.blit(continue_txt, continue_txt.get_rect(center=self.popup_continue_rect.center))

                self.popup_switch_rect = pygame.Rect(0, 0, 170, 42)
                self.popup_switch_rect.midbottom = (popup_rect.centerx + 96, button_y)
                pygame.draw.rect(self.screen, BUTTON_BG, self.popup_switch_rect, border_radius=10)
                pygame.draw.rect(self.screen, (54, 134, 170), self.popup_switch_rect, width=2, border_radius=10)
                switch_txt = self.font_button.render("Switch Side", True, TEXT_PRIMARY)
                self.screen.blit(switch_txt, switch_txt.get_rect(center=self.popup_switch_rect.center))

                self.popup_close_rect = pygame.Rect(0, 0, 94, 34)
                self.popup_close_rect.topright = (popup_rect.right - 16, popup_rect.y + 16)
                pygame.draw.rect(self.screen, BUTTON_BG, self.popup_close_rect, border_radius=8)
                pygame.draw.rect(self.screen, (54, 134, 170), self.popup_close_rect, width=2, border_radius=8)
                close_txt = self.font_small.render("Close", True, TEXT_PRIMARY)
                self.screen.blit(close_txt, close_txt.get_rect(center=self.popup_close_rect.center))
            else:
                self.popup_continue_rect = None
                self.popup_switch_rect = None

                button_y = popup_rect.bottom - 22
                self.popup_rematch_rect = pygame.Rect(0, 0, 148, 42)
                self.popup_rematch_rect.midbottom = (popup_rect.centerx - 86, button_y)
                pygame.draw.rect(self.screen, BUTTON_BG_HOVER, self.popup_rematch_rect, border_radius=10)
                pygame.draw.rect(self.screen, (54, 164, 170), self.popup_rematch_rect, width=2, border_radius=10)
                rematch_txt = self.font_button.render("Rematch", True, TEXT_PRIMARY)
                self.screen.blit(rematch_txt, rematch_txt.get_rect(center=self.popup_rematch_rect.center))

                self.popup_close_rect = pygame.Rect(0, 0, 118, 42)
                self.popup_close_rect.midbottom = (popup_rect.centerx + 86, button_y)
                pygame.draw.rect(self.screen, BUTTON_BG, self.popup_close_rect, border_radius=10)
                pygame.draw.rect(self.screen, (54, 134, 170), self.popup_close_rect, width=2, border_radius=10)
                close_txt = self.font_button.render("Close", True, TEXT_PRIMARY)
                self.screen.blit(close_txt, close_txt.get_rect(center=self.popup_close_rect.center))

                second_row_y = button_y - 56
                self.popup_analyze_rect = pygame.Rect(0, 0, 148, 40)
                self.popup_analyze_rect.midbottom = (popup_rect.centerx - 86, second_row_y)
                pygame.draw.rect(self.screen, BUTTON_BG, self.popup_analyze_rect, border_radius=10)
                pygame.draw.rect(self.screen, (54, 134, 170), self.popup_analyze_rect, width=2, border_radius=10)
                analyze_txt = self.font_small.render("Analyze Game", True, TEXT_PRIMARY)
                self.screen.blit(analyze_txt, analyze_txt.get_rect(center=self.popup_analyze_rect.center))

                self.popup_export_rect = pygame.Rect(0, 0, 148, 40)
                self.popup_export_rect.midbottom = (popup_rect.centerx + 86, second_row_y)
                pygame.draw.rect(self.screen, BUTTON_BG, self.popup_export_rect, border_radius=10)
                pygame.draw.rect(self.screen, (54, 134, 170), self.popup_export_rect, width=2, border_radius=10)
                export_txt = self.font_small.render("Export PGN", True, TEXT_PRIMARY)
                self.screen.blit(export_txt, export_txt.get_rect(center=self.popup_export_rect.center))

                hotkey_txt = self.font_small.render("Tip: press R for rematch", True, TEXT_MUTED)
                tip_y = second_row_y - 72
                self.screen.blit(hotkey_txt, (popup_rect.x + 26, tip_y))
                popup_notice_y = second_row_y - 52

        if self.popup_notice and time.time() < self.popup_notice_until:
            note = self.font_small.render(self.popup_notice, True, ACCENT)
            self.screen.blit(note, (popup_rect.x + 26, popup_notice_y))

    def _draw_start_setup_screen(self) -> None:
        # Dimmed backdrop with centered setup card.
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        panel = pygame.Surface((self.setup_panel_rect.width, self.setup_panel_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, (10, 42, 62, 238), panel.get_rect(), border_radius=14)
        pygame.draw.rect(panel, (60, 134, 166, 220), panel.get_rect(), width=2, border_radius=14)
        self.screen.blit(panel, self.setup_panel_rect.topleft)

        title = self.font_panel_title.render("Choose Your Side", True, TEXT_PRIMARY)
        self.screen.blit(title, (self.setup_panel_rect.x + 18, self.setup_panel_rect.y + 16))

        opp = self.font_small.render("Opponent: Cortex64", True, TEXT_MUTED)
        self.screen.blit(opp, (self.setup_panel_rect.right - opp.get_width() - 18, self.setup_panel_rect.y + 24))

        user_avatar = self._get_user_avatar(46)
        self.screen.blit(user_avatar, (self.setup_panel_rect.x + 26, self.setup_panel_rect.y + 76))

        pygame.draw.rect(self.screen, (14, 58, 82), self.setup_name_input_rect, border_radius=8)
        pygame.draw.rect(
            self.screen,
            ACCENT if self.editing_name else (64, 115, 141),
            self.setup_name_input_rect,
            width=2,
            border_radius=8,
        )
        name_text = self.user_name if self.user_name else "Soprano"
        caret = "|" if self.editing_name and int(time.time() * 2) % 2 == 0 else ""
        name_label = self.font_player.render(name_text + (caret if self.editing_name else ""), True, TEXT_PRIMARY)
        self.screen.blit(
            name_label,
            (
                self.setup_name_input_rect.x + 8,
                self.setup_name_input_rect.y + (self.setup_name_input_rect.height - name_label.get_height()) // 2,
            ),
        )

        instruction = self.font_small.render("Pick color: White moves first, Black gives AI first move.", True, TEXT_MUTED)
        self.screen.blit(instruction, (self.setup_panel_rect.x + 26, self.setup_panel_rect.y + 142))
        preferred = "White" if self.human_color == WHITE else "Black"
        pref_txt = self.font_small.render(f"Preferred side: {preferred}", True, TEXT_MUTED)
        self.screen.blit(pref_txt, (self.setup_panel_rect.x + 26, self.setup_panel_rect.y + 162))

        mouse = pygame.mouse.get_pos()
        for rect, label, color in (
            (self.setup_white_rect, "Play White", (235, 239, 244)),
            (self.setup_black_rect, "Play Black", (236, 245, 250)),
        ):
            hover = rect.collidepoint(mouse)
            base = BUTTON_BG_HOVER if hover else BUTTON_BG
            pygame.draw.rect(self.screen, base, rect, border_radius=10)
            pygame.draw.rect(self.screen, (54, 134, 170), rect, width=2, border_radius=10)
            txt = self.font_button.render(label, True, color)
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        hotkeys = self.font_small.render("Hotkeys: W = White, B = Black", True, TEXT_MUTED)
        self.screen.blit(hotkeys, (self.setup_panel_rect.x + 26, self.setup_panel_rect.bottom - 26))

    def draw(self) -> None:
        self.screen.blit(self.background, (0, 0))
        if self.awaiting_start_choice:
            self._draw_start_setup_screen()
            return

        self._draw_player_cards()
        self._draw_board()
        self._draw_last_move()
        self._draw_pieces()
        self._draw_selected_square()
        self._draw_legal_move_hints()
        self._draw_hint_arrow()
        self._draw_coordinates()
        self._draw_sidebar()
        self._draw_postgame_popup()

    def run(self) -> None:
        while True:
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                self.handle_event(event)
            self._poll_ai_result()
            self._poll_postgame_analysis()
            self._start_ai_search()
            self.draw()
            pygame.display.flip()


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()

"""Bounded AI search helpers reused by the game and analysis screens."""

from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty
from typing import Optional

from ai.evaluate import NeuralEvaluator
from engine.board import Board, EMPTY, Move
from engine.minimax import find_best_move
from engine.move_generator import is_checkmate

MATERIAL_VALUES = {
    1: 1.0,
    2: 3.2,
    3: 3.35,
    4: 5.1,
    5: 9.4,
    6: 0.0,
}


class MaterialEvaluator:
    """Deterministic material-plus-shape evaluator used as the safe fallback."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        self.loaded = False

    def evaluate(self, board: Board) -> float:
        """Return an evaluation from White's perspective in pawns."""
        score = 0.0
        for row in range(8):
            for col in range(8):
                piece = int(board.squares[row, col])
                if piece == EMPTY:
                    continue
                piece_type = abs(piece)
                base = MATERIAL_VALUES.get(piece_type, 0.0)
                sign = 1.0 if piece > 0 else -1.0
                center_distance = abs(3.5 - row) + abs(3.5 - col)
                if piece_type == 1:
                    advance = (6 - row) if piece > 0 else (row - 1)
                    base += 0.06 * advance
                elif piece_type in (2, 3):
                    base += max(0.0, 0.22 - 0.04 * center_distance)
                elif piece_type == 4:
                    base += 0.04 * max(0, 3 - abs(3.5 - col))
                elif piece_type == 5:
                    base += max(0.0, 0.12 - 0.025 * center_distance)
                score += sign * base
        return score


def create_evaluator(model_path: str) -> MaterialEvaluator | NeuralEvaluator:
    """Load the neural evaluator if available, otherwise use the fallback evaluator."""
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
    evaluator = NeuralEvaluator(model_path=model_path) if use_neural_model else MaterialEvaluator(model_path=model_path)
    if use_neural_model and not evaluator.loaded:
        evaluator = MaterialEvaluator(model_path=model_path)
    for depth in range(1, max(1, max_depth) + 1):
        result = find_best_move(board.copy(), depth=depth, evaluator=evaluator)
        out_queue.put((depth, result.move, float(result.score), int(result.nodes)))


def search_best_move_with_budget(
    board_snapshot: Board,
    max_depth: int,
    budget_sec: float,
    evaluator: MaterialEvaluator | NeuralEvaluator,
) -> tuple[Optional[Move], float, int, int, float]:
    """Return the best move found within a strict wall-time budget."""
    start = time.time()
    budget = max(0.2, float(budget_sec))
    deadline = start + budget

    search_queue: mp.Queue = mp.Queue()
    proc = mp.Process(
        target=_timed_search_worker,
        args=(board_snapshot, max_depth, evaluator.model_path, bool(getattr(evaluator, "loaded", False)), search_queue),
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
        fallback = find_best_move(board_snapshot.copy(), depth=1, evaluator=evaluator)
        best_move = fallback.move
        best_score = float(fallback.score)
        best_nodes = int(fallback.nodes)
        best_depth = 1 if fallback.move is not None else 0

    elapsed = time.time() - start
    return best_move, best_score, best_nodes, best_depth, elapsed


def evaluate_move_delta(board_before: Board, move: Move, evaluator: MaterialEvaluator | NeuralEvaluator, depth: int = 1) -> tuple[float, str]:
    """Return a centipawn loss style delta and best move uci for a played move."""
    analysis_depth = max(1, depth)
    best_result = find_best_move(board_before.copy(), depth=analysis_depth, evaluator=evaluator)
    best_score = float(best_result.score) if best_result.move is not None else 0.0
    best_move_uci = best_result.move.uci() if best_result.move is not None else ""

    board_after = board_before.copy()
    board_after.push(move)
    if is_checkmate(board_after):
        played_score = 1000.0
    else:
        reply = find_best_move(board_after.copy(), depth=analysis_depth, evaluator=evaluator)
        played_score = 0.0 if reply.move is None else -float(reply.score)
    return max(0.0, best_score - played_score) * 100.0, best_move_uci


def classify_quality(delta_cp: float) -> str:
    """Map a centipawn loss style delta to one of the UI quality badges."""
    if delta_cp <= 8:
        return "good"
    if delta_cp <= 25:
        return "inaccuracy"
    return "mistake"

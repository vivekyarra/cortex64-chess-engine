from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

# Piece encoding:
#   0  : empty
#   1-6: white pawn/knight/bishop/rook/queen/king
#  -1..-6: same for black
EMPTY = 0

PAWN = 1
KNIGHT = 2
BISHOP = 3
ROOK = 4
QUEEN = 5
KING = 6

WHITE = 1
BLACK = -1

# Castling rights bitmask.
W_KINGSIDE = 1 << 0
W_QUEENSIDE = 1 << 1
B_KINGSIDE = 1 << 2
B_QUEENSIDE = 1 << 3

PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 0,
}


def piece_color(piece: int) -> int:
    if piece > 0:
        return WHITE
    if piece < 0:
        return BLACK
    return 0


def piece_type(piece: int) -> int:
    return abs(piece)


def square_to_coord(square: int) -> Tuple[int, int]:
    return square // 8, square % 8


def coord_to_square(row: int, col: int) -> int:
    return row * 8 + col


def square_name(square: int) -> str:
    row, col = square_to_coord(square)
    file_char = "abcdefgh"[col]
    rank_char = str(8 - row)
    return f"{file_char}{rank_char}"


@dataclass(frozen=True)
class Move:
    """A single chess move.

    Squares are 0..63 with 0 = a8 and 63 = h1.
    """

    from_sq: int
    to_sq: int
    promotion: Optional[int] = None  # piece code (signed), e.g. WHITE*QUEEN
    is_en_passant: bool = False
    is_castling: bool = False

    def uci(self) -> str:
        s = f"{square_name(self.from_sq)}{square_name(self.to_sq)}"
        if self.promotion is not None:
            promo = piece_type(self.promotion)
            promo_char = {QUEEN: "q", ROOK: "r", BISHOP: "b", KNIGHT: "n"}.get(promo, "q")
            s += promo_char
        return s


@dataclass
class UndoState:
    move: Move
    moved_piece: int
    captured_piece: int
    captured_sq: int
    castling_rights: int
    en_passant: int
    halfmove_clock: int
    fullmove_number: int
    white_king_sq: int
    black_king_sq: int


class Board:
    """Chess board state with push/pop for search."""

    def __init__(
        self,
        squares: Optional[np.ndarray] = None,
        side_to_move: int = WHITE,
        castling_rights: int = W_KINGSIDE | W_QUEENSIDE | B_KINGSIDE | B_QUEENSIDE,
        en_passant: int = -1,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
    ) -> None:
        if squares is None:
            self.squares = self._starting_position()
        else:
            arr = np.asarray(squares, dtype=np.int8)
            if arr.shape != (8, 8):
                raise ValueError("squares must have shape (8, 8)")
            self.squares = arr.copy()

        self.side_to_move = side_to_move
        self.castling_rights = castling_rights
        self.en_passant = en_passant
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number

        self.white_king_sq, self.black_king_sq = self._find_kings()
        self._undo_stack: list[UndoState] = []

    @staticmethod
    def _starting_position() -> np.ndarray:
        b = np.zeros((8, 8), dtype=np.int8)
        b[0] = np.array([-ROOK, -KNIGHT, -BISHOP, -QUEEN, -KING, -BISHOP, -KNIGHT, -ROOK], dtype=np.int8)
        b[1] = -PAWN
        b[6] = PAWN
        b[7] = np.array([ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK], dtype=np.int8)
        return b

    def _find_kings(self) -> Tuple[int, int]:
        white_sq = -1
        black_sq = -1
        for r in range(8):
            for c in range(8):
                p = int(self.squares[r, c])
                if p == WHITE * KING:
                    white_sq = coord_to_square(r, c)
                elif p == BLACK * KING:
                    black_sq = coord_to_square(r, c)
        if white_sq == -1 or black_sq == -1:
            raise ValueError("Both kings must be present on the board.")
        return white_sq, black_sq

    def copy(self) -> "Board":
        return Board(
            squares=self.squares,
            side_to_move=self.side_to_move,
            castling_rights=self.castling_rights,
            en_passant=self.en_passant,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )

    def piece_at(self, square: int) -> int:
        r, c = square_to_coord(square)
        return int(self.squares[r, c])

    def _set_piece(self, square: int, piece: int) -> None:
        r, c = square_to_coord(square)
        self.squares[r, c] = np.int8(piece)

    def _update_castling_rights_for_move(self, moved_piece: int, from_sq: int) -> None:
        if piece_type(moved_piece) == KING:
            if piece_color(moved_piece) == WHITE:
                self.castling_rights &= ~(W_KINGSIDE | W_QUEENSIDE)
            else:
                self.castling_rights &= ~(B_KINGSIDE | B_QUEENSIDE)
            return

        if piece_type(moved_piece) != ROOK:
            return

        # Rook moved from its starting square -> lose that side's right.
        if moved_piece == WHITE * ROOK:
            if from_sq == coord_to_square(7, 0):
                self.castling_rights &= ~W_QUEENSIDE
            elif from_sq == coord_to_square(7, 7):
                self.castling_rights &= ~W_KINGSIDE
        elif moved_piece == BLACK * ROOK:
            if from_sq == coord_to_square(0, 0):
                self.castling_rights &= ~B_QUEENSIDE
            elif from_sq == coord_to_square(0, 7):
                self.castling_rights &= ~B_KINGSIDE

    def _update_castling_rights_for_capture(self, captured_piece: int, captured_sq: int) -> None:
        if piece_type(captured_piece) != ROOK:
            return

        # Captured a rook on its starting square -> lose that side's right.
        if captured_piece == WHITE * ROOK:
            if captured_sq == coord_to_square(7, 0):
                self.castling_rights &= ~W_QUEENSIDE
            elif captured_sq == coord_to_square(7, 7):
                self.castling_rights &= ~W_KINGSIDE
        elif captured_piece == BLACK * ROOK:
            if captured_sq == coord_to_square(0, 0):
                self.castling_rights &= ~B_QUEENSIDE
            elif captured_sq == coord_to_square(0, 7):
                self.castling_rights &= ~B_KINGSIDE

    def push(self, move: Move) -> None:
        """Apply move to board, saving undo state for pop()."""

        moved_piece = self.piece_at(move.from_sq)
        if moved_piece == EMPTY:
            raise ValueError("No piece on from_sq.")
        if piece_color(moved_piece) != self.side_to_move:
            raise ValueError("Tried to move opponent piece.")

        prev_castling = self.castling_rights
        prev_ep = self.en_passant
        prev_halfmove = self.halfmove_clock
        prev_fullmove = self.fullmove_number
        prev_wk = self.white_king_sq
        prev_bk = self.black_king_sq

        captured_piece = EMPTY
        captured_sq = move.to_sq

        if move.is_en_passant:
            # Captured pawn is behind the destination square.
            direction = -8 if self.side_to_move == BLACK else 8
            captured_sq = move.to_sq + direction
            captured_piece = self.piece_at(captured_sq)
        else:
            captured_piece = self.piece_at(move.to_sq)

        undo = UndoState(
            move=move,
            moved_piece=moved_piece,
            captured_piece=captured_piece,
            captured_sq=captured_sq,
            castling_rights=prev_castling,
            en_passant=prev_ep,
            halfmove_clock=prev_halfmove,
            fullmove_number=prev_fullmove,
            white_king_sq=prev_wk,
            black_king_sq=prev_bk,
        )

        # Reset en passant unless set by a double pawn push.
        self.en_passant = -1

        # Halfmove clock (for completeness; not used by search).
        if piece_type(moved_piece) == PAWN or captured_piece != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # Update castling rights due to moved piece and captures.
        self._update_castling_rights_for_move(moved_piece, move.from_sq)
        if captured_piece != EMPTY:
            self._update_castling_rights_for_capture(captured_piece, captured_sq)

        # Move piece.
        self._set_piece(move.from_sq, EMPTY)

        placed_piece = moved_piece
        if move.promotion is not None:
            placed_piece = move.promotion
        else:
            # If a promotion move was constructed without promotion info, default to queen.
            if piece_type(moved_piece) == PAWN:
                to_row, _ = square_to_coord(move.to_sq)
                if (self.side_to_move == WHITE and to_row == 0) or (self.side_to_move == BLACK and to_row == 7):
                    placed_piece = self.side_to_move * QUEEN

        self._set_piece(move.to_sq, placed_piece)

        # Handle captures.
        if captured_piece != EMPTY:
            if move.is_en_passant:
                self._set_piece(captured_sq, EMPTY)

        # Handle pawn double push -> set en passant square.
        if piece_type(moved_piece) == PAWN:
            from_row, _ = square_to_coord(move.from_sq)
            to_row, _ = square_to_coord(move.to_sq)
            if abs(to_row - from_row) == 2:
                ep_row = (from_row + to_row) // 2
                _, file_col = square_to_coord(move.from_sq)
                self.en_passant = coord_to_square(ep_row, file_col)

        # Handle castling rook move.
        if move.is_castling:
            row, to_col = square_to_coord(move.to_sq)
            if to_col == 6:  # kingside
                rook_from = coord_to_square(row, 7)
                rook_to = coord_to_square(row, 5)
            elif to_col == 2:  # queenside
                rook_from = coord_to_square(row, 0)
                rook_to = coord_to_square(row, 3)
            else:
                raise ValueError("Invalid castling destination square.")
            rook_piece = self.piece_at(rook_from)
            self._set_piece(rook_from, EMPTY)
            self._set_piece(rook_to, rook_piece)

        # Update king square tracking.
        if piece_type(moved_piece) == KING:
            if self.side_to_move == WHITE:
                self.white_king_sq = move.to_sq
            else:
                self.black_king_sq = move.to_sq

        # Switch side.
        self.side_to_move *= -1
        if self.side_to_move == WHITE:
            self.fullmove_number += 1

        self._undo_stack.append(undo)

    def pop(self) -> None:
        """Undo last push()."""

        if not self._undo_stack:
            raise IndexError("No moves to pop.")

        undo = self._undo_stack.pop()
        move = undo.move

        # Switch side back first (important for some move-dependent logic).
        self.side_to_move *= -1

        # Restore counters/state.
        self.castling_rights = undo.castling_rights
        self.en_passant = undo.en_passant
        self.halfmove_clock = undo.halfmove_clock
        self.fullmove_number = undo.fullmove_number
        self.white_king_sq = undo.white_king_sq
        self.black_king_sq = undo.black_king_sq

        # Undo castling rook move.
        if move.is_castling:
            row, to_col = square_to_coord(move.to_sq)
            if to_col == 6:  # kingside
                rook_from = coord_to_square(row, 7)
                rook_to = coord_to_square(row, 5)
            else:  # queenside
                rook_from = coord_to_square(row, 0)
                rook_to = coord_to_square(row, 3)
            rook_piece = self.piece_at(rook_to)
            self._set_piece(rook_to, EMPTY)
            self._set_piece(rook_from, rook_piece)

        # Move piece back.
        self._set_piece(move.to_sq, EMPTY)
        self._set_piece(move.from_sq, undo.moved_piece)

        # Restore captured piece if any.
        if undo.captured_piece != EMPTY:
            self._set_piece(undo.captured_sq, undo.captured_piece)

    def legal_tensor(self) -> np.ndarray:
        """Return (C, 8, 8) float32 planes suitable for a CNN."""

        planes = np.zeros((18, 8, 8), dtype=np.float32)

        # Piece planes.
        for r in range(8):
            for c in range(8):
                p = int(self.squares[r, c])
                if p == EMPTY:
                    continue
                pt = piece_type(p)
                is_white = piece_color(p) == WHITE
                base = 0 if is_white else 6
                planes[base + (pt - 1), r, c] = 1.0

        # Side to move plane: 1.0 for white-to-move, 0.0 for black-to-move.
        planes[12, :, :] = 1.0 if self.side_to_move == WHITE else 0.0

        # Castling rights planes.
        planes[13, :, :] = 1.0 if (self.castling_rights & W_KINGSIDE) else 0.0
        planes[14, :, :] = 1.0 if (self.castling_rights & W_QUEENSIDE) else 0.0
        planes[15, :, :] = 1.0 if (self.castling_rights & B_KINGSIDE) else 0.0
        planes[16, :, :] = 1.0 if (self.castling_rights & B_QUEENSIDE) else 0.0

        # En passant plane: a single 1.0 at the en-passant target square (if any).
        if self.en_passant != -1:
            r, c = square_to_coord(self.en_passant)
            planes[17, r, c] = 1.0

        return planes

    def __str__(self) -> str:
        piece_chars = {
            WHITE * PAWN: "P",
            WHITE * KNIGHT: "N",
            WHITE * BISHOP: "B",
            WHITE * ROOK: "R",
            WHITE * QUEEN: "Q",
            WHITE * KING: "K",
            BLACK * PAWN: "p",
            BLACK * KNIGHT: "n",
            BLACK * BISHOP: "b",
            BLACK * ROOK: "r",
            BLACK * QUEEN: "q",
            BLACK * KING: "k",
            EMPTY: ".",
        }

        lines = []
        for r in range(8):
            row = []
            for c in range(8):
                row.append(piece_chars[int(self.squares[r, c])])
            lines.append(" ".join(row))
        turn = "White" if self.side_to_move == WHITE else "Black"
        return "\n".join(lines) + f"\nTurn: {turn}"

    def iter_pieces(self, color: int) -> Iterable[Tuple[int, int]]:
        """Yield (square, piece_code) for all pieces of a color."""
        for r in range(8):
            for c in range(8):
                p = int(self.squares[r, c])
                if p != EMPTY and piece_color(p) == color:
                    yield coord_to_square(r, c), p


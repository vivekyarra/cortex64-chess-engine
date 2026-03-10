"""Shared GUI constants for the Cortex64 v2 interface."""

# Window
WINDOW_W, WINDOW_H = 1280, 800
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_OFFSET_X = 200
BOARD_OFFSET_Y = 80
SIDEBAR_LEFT_W = 180
SIDEBAR_RIGHT_W = 420
FPS = 60

# Colors
BG_DARK = (10, 10, 15)
BG_SURFACE = (18, 18, 26)
BG_CARD = (26, 26, 46)
ACCENT = (108, 99, 255)
GOLD = (255, 215, 0)
WHITE_COL = (240, 240, 255)
MUTED = (136, 136, 170)
SUCCESS = (0, 229, 160)
DANGER = (255, 76, 106)
WARNING = (255, 179, 71)
BLACK_COL = (10, 10, 15)

# Board skins
SKIN_OBSIDIAN = {
    "light": (200, 169, 110),
    "dark": (74, 55, 40),
    "name": "Obsidian Classic",
}
SKIN_WALNUT = {
    "light": (222, 196, 145),
    "dark": (96, 60, 20),
    "name": "Classic Walnut",
}
SKIN_ICE = {
    "light": (200, 220, 245),
    "dark": (80, 120, 180),
    "name": "Ice Blue",
}
SKIN_NEON = {
    "light": (30, 40, 60),
    "dark": (10, 15, 30),
    "name": "Neon Dark",
}
ALL_SKINS = [SKIN_OBSIDIAN, SKIN_WALNUT, SKIN_ICE, SKIN_NEON]

# Animation durations (ms)
ANIM_PIECE_MS = 120
ANIM_SHAKE_MS = 200
ANIM_PULSE_MS = 600
ANIM_FADE_MS = 300
ANIM_MODAL_MS = 200

# Fonts
FONT_UI_SIZE = 16
FONT_TITLE_SIZE = 32
FONT_CLOCK_SIZE = 28
FONT_EVAL_SIZE = 24
FONT_COORD_SIZE = 14

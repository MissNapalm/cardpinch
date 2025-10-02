import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
from collections import deque
import subprocess
import time
import os
import random
import datetime

# ==============================
# Carousel config
# ==============================
CARD_COUNT = 7
CARD_WIDTH = 280
CARD_HEIGHT = 280
CARD_SPACING = 50
ROW_BASE_SPACING = CARD_HEIGHT + 80

CAROUSEL_CATEGORIES = [
    ["Mail", "Music", "Browser", "Messages", "Calendar", "Maps", "Camera"],
    ["Photos", "Notes", "Reminders", "Clock", "Weather", "Stocks", "News"],
    ["YouTube", "Netflix", "Twitch", "Spotify", "Podcasts", "Books", "Games"]
]
NUM_CATEGORIES = len(CAROUSEL_CATEGORIES)

APP_COLORS = {
    "Mail": (74, 144, 226), "Music": (252, 61, 86), "Safari": (35, 142, 250),
    "Messages": (76, 217, 100), "Calendar": (252, 61, 57), "Maps": (89, 199, 249),
    "Camera": (138, 138, 142), "Photos": (252, 203, 47), "Notes": (255, 214, 10),
    "Reminders": (255, 69, 58), "Clock": (30, 30, 30), "Weather": (99, 204, 250),
    "Stocks": (30, 30, 30), "News": (252, 61, 86), "YouTube": (255, 0, 0),
    "Netflix": (229, 9, 20), "Twitch": (145, 70, 255), "Spotify": (30, 215, 96),
    "Podcasts": (146, 72, 223), "Books": (255, 124, 45), "Games": (255, 45, 85),
    "Browser": (35, 142, 250)
}

mp_hands = mp.solutions.hands


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ==============================
# Carousel classes/util
# ==============================
class FingerSmoother:
    def __init__(self, window_size=5):
        self.thumb_history = deque(maxlen=window_size)
        self.index_history = deque(maxlen=window_size)

    def update(self, thumb_pos, index_pos):
        self.thumb_history.append(thumb_pos)
        self.index_history.append(index_pos)
        tx = sum(p[0] for p in self.thumb_history) / len(self.thumb_history)
        ty = sum(p[1] for p in self.thumb_history) / len(self.thumb_history)
        ix = sum(p[0] for p in self.index_history) / len(self.index_history)
        iy = sum(p[1] for p in self.index_history) / len(self.index_history)
        return (tx, ty), (ix, iy)

    def reset(self):
        self.thumb_history.clear()
        self.index_history.clear()


class HandState:
    def __init__(self):
        self.card_offset = 0.0
        self.category_offset = 0.0
        self.smooth_card_offset = 0.0
        self.smooth_category_offset = 0.0
        self.scroll_smoothing = 0.25
        self.scroll_gain = 5.0  # more scroll with less movement

        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_start_pos = None
        self.movement_threshold = 10

        self.selected_card = None
        self.selected_category = None
        self.zoom_progress = 0.0
        self.zoom_target = 0.0
        self.finger_smoother = FingerSmoother(window_size=5)

        # zoom wheel
        self.wheel_active = False
        self.wheel_angle = math.pi
        self.last_finger_angle = None
        self.wheel_center_x = 0
        self.wheel_center_y = 0
        self.wheel_radius = 110
        self.gui_scale = 1.00
        self.gui_scale_min = 0.60
        self.gui_scale_max = 1.80
        self.gui_scale_sensitivity = 0.32

        # pinch timing
        self.pinch_threshold = 0.08
        self.pinch_prev = False
        self.last_pinch_time = 0
        self.double_pinch_window = 0.4  # seconds

        # misc
        self.browser_process = None
        self.email_process = None
        self.current_fps = 0.0

        # A-OK gesture to reset zoom
        self.ok_prev = False
        self.ok_touch_threshold = 0.035


def get_pinch_distance(landmarks):
    if not landmarks:
        return None
    a = landmarks[4]
    b = landmarks[8]
    return math.hypot(a.x - b.x, a.y - b.y)


def is_pinching(landmarks, thresh):
    d = get_pinch_distance(landmarks)
    return (d is not None) and (d < thresh)


def get_pinch_position(landmarks):
    if not landmarks:
        return None
    a = landmarks[4]
    b = landmarks[8]
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)


def is_finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def detect_three_finger_gesture(landmarks):
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    thumb_ext = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
    index_ext = is_finger_extended(landmarks, 8, 6)
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_fold = landmarks[16].y > landmarks[14].y - 0.02
    pinky_fold = landmarks[20].y > landmarks[18].y - 0.02
    return thumb_ext and index_ext and middle_ext and ring_fold and pinky_fold


def detect_ok_gesture(landmarks, touch_thresh=0.035):
    if not landmarks:
        return False
    a = landmarks[4]
    b = landmarks[8]
    touching = math.hypot(a.x - b.x, a.y - b.y) < touch_thresh
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_ext = is_finger_extended(landmarks, 16, 14)
    pinky_ext = is_finger_extended(landmarks, 20, 18)
    return touching and middle_ext and ring_ext and pinky_ext


def get_hand_center(landmarks):
    return landmarks[9]


def calculate_finger_angle(landmarks):
    c = get_hand_center(landmarks)
    idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)


# ==============================
# Launch helpers (Browser / Email)
# ==============================
def _search_paths_for(basename_or_path):
    """
    Return an absolute path to the first matching file:
    - If absolute path provided and exists, return it.
    - Else search in script directory and current working directory.
    """
    if not basename_or_path:
        return None
    p = os.path.expanduser(basename_or_path)
    if os.path.isabs(p) and os.path.isfile(p):
        return p

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), p),
        os.path.join(script_dir, p)
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def try_spawn(candidates_or_paths):
    """
    Try a list of basenames or absolute paths.
    Returns a Popen if successful, or raises the last error.
    """
    last_err = None

    # Env override first (used for email)
    env_app = os.environ.get("GESTURE_EMAIL_APP")
    if env_app:
        resolved = _search_paths_for(env_app)
        if resolved:
            try:
                return subprocess.Popen([sys.executable, resolved])
            except Exception as e:
                last_err = e

    # Then provided candidates
    for name in candidates_or_paths:
        resolved = _search_paths_for(name)
        if not resolved:
            continue
        try:
            return subprocess.Popen([sys.executable, resolved])
        except Exception as e:
            last_err = e

    if last_err:
        raise last_err
    raise FileNotFoundError("No candidate script found")


def launch_browser_window(state):
    if state.browser_process is None or state.browser_process.poll() is not None:
        try:
            state.browser_process = try_spawn(["gesture_webview.py", "browser_window.py"])
            print("✓ BROWSER WINDOW LAUNCHED!")
        except Exception as e:
            print(f"Error launching browser: {e}")
            print("Make sure 'gesture_webview.py' or 'browser_window.py' exists in the same directory!")


def launch_email_window(state):
    """Try external email scripts. If none found, launch embedded fallback in a separate process."""
    if state.email_process is not None and state.email_process.poll() is None:
        return

    # 1) Try user/env + common names
    try:
        state.email_process = try_spawn(["my_email.py", "email.py", "gesture_mail_inbox.py", "email_inbox.py"])
        print("✉️  EMAIL WINDOW LAUNCHED!")
        return
    except Exception as e:
        print(f"Email script not found / failed to start external: {e}")

    # 2) Fallback: spawn this file with --email-fallback
    try:
        state.email_process = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--email-fallback"])
        print("✉️  EMAIL FALLBACK LAUNCHED (embedded)!")
    except Exception as e:
        print(f"Error launching embedded email fallback: {e}")
        print("If you prefer your own file, create one named 'my_email.py' or set GESTURE_EMAIL_APP=/path/to/your.py")


# ==============================
# Carousel drawing
# ==============================
def draw_app_icon(surface, app_name, x, y, base_w, base_h, is_selected=False, zoom_scale=1.0, gui_scale=1.0):
    width = int(base_w * gui_scale)
    height = int(base_h * gui_scale)
    if is_selected:
        width = int(width * zoom_scale)
        height = int(height * zoom_scale)
    br = max(12, int(50 * gui_scale))
    rect = pygame.Rect(x - width // 2, y - height // 2, width, height)
    color = tuple(min(255, int(APP_COLORS.get(app_name, (100, 100, 100))[i] * 1.2)) for i in range(3))
    pygame.draw.rect(surface, color, rect, border_radius=br)
    if is_selected:
        sel = pygame.Rect(rect.x - int(6 * gui_scale), rect.y - int(6 * gui_scale),
                          rect.width + int(12 * gui_scale), rect.height + int(12 * gui_scale))
        pygame.draw.rect(surface, (255, 255, 255), sel, width=max(2, int(8 * gui_scale)), border_radius=br)
    icon_font = pygame.font.Font(None, max(24, int(120 * (width / max(1, int(base_w * gui_scale))))))
    icon_img = icon_font.render(app_name[0], True, (255, 255, 255, 180))
    surface.blit(icon_img, icon_img.get_rect(center=(x, y - int(20 * gui_scale))))
    text_size = max(12, int(36 * (width / max(1, int(base_w * gui_scale)))))
    font = pygame.font.Font(None, text_size)
    text_img = font.render(app_name, True, (255, 255, 255))
    surface.blit(text_img, text_img.get_rect(center=(x, y + int(60 * gui_scale))))
    return rect


def draw_cards(surface, center_x, center_y, card_offset, category_idx,
               selected_card=None, selected_category=None, zoom_progress=0.0,
               window_width=1280, gui_scale=1.0, base_w=280, base_h=280, base_spacing=50):
    app_names = CAROUSEL_CATEGORIES[category_idx]
    card_rects = []
    sw = int(base_w * gui_scale)
    ss = int(base_spacing * gui_scale)
    stride = sw + ss
    first_vis = int((-card_offset - window_width // 2) / stride) - 1
    last_vis = int((-card_offset + window_width // 2) / stride) + 2
    first_vis = max(0, first_vis)
    last_vis = min(CARD_COUNT, last_vis)
    # non-selected behind
    for i in range(first_vis, last_vis):
        x = int(center_x + (i * stride) + card_offset)
        y = int(center_y)
        sel = (selected_card == i and selected_category == category_idx)
        if not sel:
            rect = draw_app_icon(surface, app_names[i], x, y, base_w, base_h, False, 1.0, gui_scale)
            card_rects.append((rect, i, category_idx))
    # selected on top
    for i in range(first_vis, last_vis):
        x = int(center_x + (i * stride) + card_offset)
        y = int(center_y)
        sel = (selected_card == i and selected_category == category_idx)
        if sel:
            rect = draw_app_icon(surface, app_names[i], x, y, base_w, base_h, True, 1.0 + (zoom_progress * 0.3), gui_scale)
            card_rects.append((rect, i, category_idx))
    return card_rects


def draw_wheel(surface, state, window_width, window_height):
    if not state.wheel_active:
        return
    scale = state.gui_scale
    cx = state.wheel_center_x
    cy = state.wheel_center_y
    r = int(state.wheel_radius * scale)
    white = (255, 255, 255)
    for i in range(5):
        rr = r + int(15 * scale) + i * int(10 * scale)
        op = int(100 - i * 20)
        s = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        pygame.draw.circle(s, (*white, op), (cx, cy), rr, max(1, int(2 * scale)))
        surface.blit(s, (0, 0))
    pygame.draw.circle(surface, white, (cx, cy), r, max(1, int(4 * scale)))
    pygame.draw.circle(surface, white, (cx, cy), r - int(20 * scale), max(1, int(2 * scale)))
    segs = 48
    prog = int((state.wheel_angle / (2 * math.pi)) * segs) % segs
    ir = r - int(10 * scale)
    for i in range(prog):
        sa = math.radians(i * 360 / segs) - math.pi / 2
        ea = math.radians((i + 1) * 360 / segs) - math.pi / 2
        sx = cx + int(ir * math.cos(sa))
        sy = cy + int(ir * math.sin(sa))
        ex = cx + int(ir * math.cos(ea))
        ey = cy + int(ir * math.sin(ea))
        pygame.draw.line(surface, white, (sx, sy), (ex, ey), max(1, int(6 * scale)))
    pl = r - int(30 * scale)
    px = cx + int(pl * math.cos(state.wheel_angle))
    py = cy + int(pl * math.sin(state.wheel_angle))
    pygame.draw.line(surface, white, (cx, cy), (px, py), max(1, int(3 * scale)))
    pygame.draw.circle(surface, white, (px, py), max(2, int(6 * scale)))
    pygame.draw.circle(surface, white, (cx, cy), max(2, int(8 * scale)))
    font = pygame.font.Font(None, max(18, int(40 * scale)))
    t = font.render(f"GUI {state.gui_scale:.2f}x", True, white)
    tr = t.get_rect(center=(cx, cy + r + int(44 * scale)))
    bg = pygame.Rect(tr.x - int(10 * scale), tr.y - int(5 * scale), tr.width + int(20 * scale), tr.height + int(10 * scale))
    pygame.draw.rect(surface, (20, 20, 20), bg)
    pygame.draw.rect(surface, white, bg, max(1, int(2 * scale)))
    surface.blit(t, tr)


def lm_to_screen(lm, W, H):
    return (lm.x * W, lm.y * H)


# ==============================
# Carousel main
# ==============================
def carousel_main():
    pygame.init()
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Gesture Carousel • ⌘ Double-pinch Mail or Browser to launch")
    clock = pygame.time.Clock()

    print("=" * 50)
    print("GESTURE CAROUSEL STARTED")
    print("Pinch to scroll rows/columns • Tap to select • Double-pinch directly on a card to launch")
    print("If Mail script not found, embedded inbox fallback will open.")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

    state = HandState()
    tap_to_check = None
    double_pinch_to_check = None  # (x,y) of the double-pinch location
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        fps = clock.get_fps()
        if fps > 0:
            state.current_fps = (0.9 * state.current_fps + 0.1 * fps) if state.current_fps > 0 else fps

        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                if hd.classification[0].label == "Right":
                    right_hand = hl.landmark

        if right_hand is None:
            state.finger_smoother.reset()
            state.wheel_active = False
            state.last_finger_angle = None

        # A-OK resets GUI zoom to default
        ok_now = detect_ok_gesture(right_hand, state.ok_touch_threshold) if right_hand else False
        if ok_now and not state.ok_prev:
            state.gui_scale = 1.0
            state.wheel_active = False
            state.last_finger_angle = None
            print("A-OK detected — GUI scale reset to 1.00x")
        state.ok_prev = ok_now

        pinch_now = is_pinching(right_hand, state.pinch_threshold) if right_hand else False

        # Three-finger wheel gesture (zoom)
        if right_hand and detect_three_finger_gesture(right_hand):
            if not state.wheel_active:
                hc = get_hand_center(right_hand)
                state.wheel_active = True
                state.wheel_center_x = int(hc.x * WINDOW_WIDTH)
                state.wheel_center_y = int(hc.y * WINDOW_HEIGHT)
                state.last_finger_angle = None
            ang = calculate_finger_angle(right_hand)
            if state.last_finger_angle is not None:
                diff = ang - state.last_finger_angle
                if diff > math.pi:
                    diff -= 2 * math.pi
                elif diff < -math.pi:
                    diff += 2 * math.pi
                state.wheel_angle = (state.wheel_angle + diff * 2) % (2 * math.pi)
                state.gui_scale = clamp(
                    state.gui_scale + diff * state.gui_scale_sensitivity,
                    state.gui_scale_min, state.gui_scale_max
                )
            state.last_finger_angle = ang
        else:
            state.wheel_active = False
            state.last_finger_angle = None

        # Pinch scroll + tap/double-pinch detection (single, unified handler)
        if right_hand and not state.wheel_active:
            pos = get_pinch_position(right_hand)
            if pinch_now and not state.pinch_prev:
                if pos:
                    px = pos[0] * WINDOW_WIDTH
                    py = pos[1] * WINDOW_HEIGHT
                    state.pinch_start_pos = (px, py)
                    state.last_pinch_x = px
                    state.last_pinch_y = py
                    state.is_pinching = True

            elif pinch_now and state.pinch_prev and pos:
                px = pos[0] * WINDOW_WIDTH
                py = pos[1] * WINDOW_HEIGHT
                if state.last_pinch_x is not None:
                    dx = px - state.last_pinch_x
                    dy = py - state.last_pinch_y
                    # only scroll if movement is significant
                    if state.pinch_start_pos:
                        total_dx = px - state.pinch_start_pos[0]
                        total_dy = py - state.pinch_start_pos[1]
                        if math.hypot(total_dx, total_dy) > state.movement_threshold:
                            state.card_offset += dx * state.scroll_gain
                            state.category_offset += dy * state.scroll_gain
                            stride_x = int((CARD_WIDTH + CARD_SPACING) * state.gui_scale)
                            min_x = -(CARD_COUNT - 1) * stride_x
                            state.card_offset = clamp(state.card_offset, min_x, 0)
                            row_stride = int(ROW_BASE_SPACING * state.gui_scale)
                            min_y = -(NUM_CATEGORIES - 1) * row_stride
                            state.category_offset = clamp(state.category_offset, min_y, 0)
                state.last_pinch_x = px
                state.last_pinch_y = py

            elif not pinch_now and state.pinch_prev:
                # pinch released — decide single-tap vs double-pinch
                if state.pinch_start_pos and state.last_pinch_x is not None:
                    total_dx = state.last_pinch_x - state.pinch_start_pos[0]
                    total_dy = state.last_pinch_y - state.pinch_start_pos[1]
                    total_move = math.hypot(total_dx, total_dy)

                    # evaluate time window
                    current_time = time.time()
                    dt = current_time - state.last_pinch_time

                    # if minimal movement, queue a tap hit-test
                    if total_move <= state.movement_threshold:
                        tap_to_check = (state.last_pinch_x, state.last_pinch_y)

                    # if within double-pinch window, queue a double-pinch hit-test
                    if 0.05 < dt < state.double_pinch_window:
                        double_pinch_to_check = (state.last_pinch_x, state.last_pinch_y)
                        print("✓ Double pinch detected — will resolve on card under finger")

                    state.last_pinch_time = current_time

                state.is_pinching = False
                state.last_pinch_x = None
                state.last_pinch_y = None
                state.pinch_start_pos = None
        else:
            state.is_pinching = False
            state.last_pinch_x = None
            state.last_pinch_y = None
            state.pinch_start_pos = None

        state.pinch_prev = pinch_now

        # Smooth offsets
        s = state.scroll_smoothing
        state.smooth_card_offset += (state.card_offset - state.smooth_card_offset) * s
        state.smooth_category_offset += (state.category_offset - state.smooth_category_offset) * s

        # Draw carousel
        screen.fill((20, 20, 30))
        cx = WINDOW_WIDTH // 2
        cy = WINDOW_HEIGHT // 2
        all_rects = []
        row_stride = int(ROW_BASE_SPACING * state.gui_scale)
        first_cat = max(0, int(-state.smooth_category_offset / row_stride) - 1)
        last_cat = min(NUM_CATEGORIES, int((-state.smooth_category_offset + WINDOW_HEIGHT) / row_stride) + 2)
        for cat_idx in range(first_cat, last_cat):
            y = cy + (cat_idx * row_stride) + state.smooth_category_offset
            all_rects += draw_cards(
                screen, cx, int(y), state.smooth_card_offset, cat_idx,
                state.selected_card, state.selected_category, state.zoom_progress,
                WINDOW_WIDTH, state.gui_scale, CARD_WIDTH, CARD_HEIGHT, CARD_SPACING
            )

        # Resolve queued taps (selection)
        if tap_to_check:
            tx, ty = tap_to_check
            for rect, ci, ca in all_rects:
                if rect.collidepoint(tx, ty):
                    state.selected_card = ci
                    state.selected_category = ca
                    state.zoom_target = 1.0
                    print(f"Selected: {CAROUSEL_CATEGORIES[ca][ci]} (card {ci}, category {ca})")
                    break
            tap_to_check = None

        # Resolve queued double-pinch (launch under finger)
        if double_pinch_to_check:
            dx, dy = double_pinch_to_check
            launched = False
            for rect, ci, ca in all_rects:
                if rect.collidepoint(dx, dy):
                    app_name = CAROUSEL_CATEGORIES[ca][ci]
                    if app_name == "Mail":
                        launch_email_window(state)
                        print("✓✓✓ DOUBLE PINCH ON MAIL — LAUNCHING EMAIL! ✓✓✓")
                        launched = True
                    elif app_name == "Browser":
                        launch_browser_window(state)
                        print("✓✓✓ DOUBLE PINCH ON BROWSER — LAUNCHING WINDOW! ✓✓✓")
                        launched = True
                    else:
                        print(f"Double pinch on {app_name} — no action bound")
                    break
            if not launched:
                print("Double pinch location didn't hit a card")
            double_pinch_to_check = None

        # Wheel overlay
        draw_wheel(screen, state, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Camera preview (top-right)
        frame_surface = pygame.surfarray.make_surface(cv2.transpose(rgb))
        frame_surface = pygame.transform.scale(frame_surface, (320, 240))
        screen.blit(frame_surface, (WINDOW_WIDTH - 330, 10))

        # Hand HUD
        if right_hand:
            tt = right_hand[4]
            it = right_hand[8]
            (tx, ty), (ix, iy) = state.finger_smoother.update(
                lm_to_screen(tt, WINDOW_WIDTH, WINDOW_HEIGHT),
                lm_to_screen(it, WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            if not state.wheel_active and pinch_now:
                pygame.draw.line(screen, (255, 255, 255), (int(tx), int(ty)), (int(ix), int(iy)), 2)
            pygame.draw.circle(screen, (255, 255, 255), (int(tx), int(ty)), 8)
            pygame.draw.circle(screen, (255, 255, 255), (int(ix), int(iy)), 8)
        else:
            state.finger_smoother.reset()

        # Status text
        font = pygame.font.Font(None, 48)
        if state.wheel_active:
            status = f"WHEEL • GUI {state.gui_scale:.2f}x"
        elif state.is_pinching:
            status = "PINCHED"
        else:
            status = "Ready • ⌘ Double-pinch Mail/Browser to launch"
        screen.blit(font.render(status, True, (255, 255, 255)), (30, 30))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()


# ==============================
# Embedded Email Fallback (runs in separate process)
# ==============================
def _make_fake_emails(n=60):
    SENDERS = [
        "Lena from Summation", "Nadia (Infra)", "Core Team Updates",
        "Billing Bot", "Product Announce", "Priya @ ML", "Marketing Ops",
        "Ahmed • Security", "Bruno (Design)", "Support", "CI Logs", "The Ops Room"
    ]
    SUBJECTS = [
        "Standup notes & priorities", "Your invoice is ready",
        "Quarterly planning: draft agenda", "Welcome to the beta",
        "Alert: action required on deployment", "Design review — nav cleanup",
        "Incident postmortem draft", "Metrics weekly recap",
        "Invitation: user research sessions", "Release train R42 checklist",
        "Reminder: security training", "Infra cost dashboard — July"
    ]
    SNIPPETS = [
        "Sharing quick notes from the sync today. We agreed to cut scope…",
        "Hi there! Your invoice for September is now available. You can…",
        "Here's a rough draft for next quarter's planning doc. Please…",
        "You're in! Start by exploring the quick start guide and sample…",
        "We detected a failing canary in us-west-2. Rollback is prepared…",
        "Attaching comps with the nav collapsed and with tabs. Feedback…",
        "Root cause was a misconfigured retry policy. We'll add guards…",
        "Top-line usage is up 12%. Retention is flat; activation dipped…",
        "We'd love to schedule 30 minutes to discuss your current tooling…",
        "This week: 16 merges, 2 hotfixes, 0 regressions. Please read…",
        "A friendly nudge that your security review is due Friday. It…",
        "Spend is trending 8% down week over week due to improved cache…"
    ]
    LABELS = ["Work", "Docs", "Billing", "Follow-up", "Personal", "Newsletters", "Release"]
    now = datetime.datetime.now()
    emails = []
    for _ in range(n):
        sender = random.choice(SENDERS)
        subject = random.choice(SUBJECTS)
        snippet = random.choice(SNIPPETS)
        unread = random.random() < 0.45
        starred = random.random() < 0.20
        lbls = random.sample(LABELS, k=random.randint(0, 2))
        dt = now - datetime.timedelta(minutes=random.randint(5, 60 * 24 * 14))
        if (now - dt).days == 0:
            time_label = dt.strftime("%I:%M %p").lstrip("0")
        else:
            time_label = dt.strftime("%b %d")
        emails.append({
            "sender": sender,
            "subject": subject,
            "snippet": snippet,
            "time": time_label,
            "unread": unread,
            "starred": starred,
            "labels": lbls
        })
    return emails


def _draw_inbox_list(surface, x, y, w, h, emails, scroll_y):
    pygame.draw.rect(surface, (255, 255, 255), (x, y, w, h))
    row_h = 72
    start_i = max(0, int((-scroll_y) // row_h) - 2)
    end_i = min(len(emails), start_i + h // row_h + 4)
    for i in range(start_i, end_i):
        ry = y + int(scroll_y) + i * row_h
        rect = pygame.Rect(x, ry, w, row_h)
        if emails[i]["unread"]:
            pygame.draw.rect(surface, (244, 248, 255), rect)
        else:
            pygame.draw.rect(surface, (255, 255, 255), rect)
        pygame.draw.line(surface, (233, 237, 241), (rect.x, rect.bottom), (rect.right, rect.bottom), 1)
        # sender
        f_sender = pygame.font.SysFont('arial', 16, bold=emails[i]["unread"])
        surface.blit(f_sender.render(emails[i]["sender"], True, (33, 37, 41)), (rect.x + 70, rect.y + 10))
        # subject + snippet
        subj_font = pygame.font.SysFont('arial', 16, bold=emails[i]["unread"])
        snip_font = pygame.font.SysFont('arial', 16)
        subj = emails[i]["subject"]
        snip = emails[i]["snippet"]
        subj_img = subj_font.render(subj, True, (33, 37, 41))
        sep_img = snip_font.render(" — ", True, (107, 114, 128))
        snp_img = snip_font.render(snip, True, (107, 114, 128))
        mx = rect.x + 70 + 200 + 10
        my = rect.y + 10
        # truncate to fit
        maxw = w - (mx - x) - 90
        sj = subj
        sj_img = subj_img
        while sj_img.get_width() > maxw and len(sj) > 1:
            sj = sj[:-2] + "…"
            sj_img = subj_font.render(sj, True, (33, 37, 41))
        surface.blit(sj_img, (mx, my))
        sx = mx + sj_img.get_width()
        if sx + sep_img.get_width() < x + w - 90:
            surface.blit(sep_img, (sx, my))
            sx += sep_img.get_width()
        remain = (x + w - 90) - sx
        if remain > 0:
            sn = snip
            sn_img = snip_font.render(sn, True, (107, 114, 128))
            while sn_img.get_width() > remain and len(sn) > 1:
                sn = sn[:-2] + "…"
                sn_img = snip_font.render(sn, True, (107, 114, 128))
            surface.blit(sn_img, (sx, my))
        # time
        timg = pygame.font.SysFont('arial', 14).render(emails[i]["time"], True, (107, 114, 128))
        surface.blit(timg, (x + w - 12 - timg.get_width(), my + 2))
    page_bottom = y + scroll_y + len(emails) * row_h
    return page_bottom - y


def email_fallback_main():
    pygame.init()
    WINDOW_WIDTH, WINDOW_HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Mail — Embedded Inbox (Fallback)")
    clock = pygame.time.Clock()

    print("=" * 50)
    print("EMBEDDED INBOX (FALLBACK)")
    print("Pinch-drag to scroll • Three-finger rotate to zoom • A-OK to quit")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

    # state
    scroll_y = 0.0
    smooth_scroll_y = 0.0
    scroll_gain = 4.0
    is_pinching = False
    last_pinch_y = None
    pinch_threshold = 0.08
    ok_prev = False
    ok_touch_threshold = 0.035
    wheel_active = False
    wheel_angle = 0.0
    last_finger_angle = None
    zoom_level = 1.0
    zoom_min, zoom_max = 0.5, 2.0
    zoom_sense = 0.15

    emails = _make_fake_emails(80)
    page_height = 3000

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        ret, frame = cap.read()
        if not ret:
            pygame.display.flip()
            clock.tick(60)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                if hd.classification[0].label == "Right":
                    right_hand = hl.landmark

        # gestures
        if right_hand and detect_three_finger_gesture(right_hand):
            if not wheel_active:
                wheel_active = True
                last_finger_angle = None
            ang = calculate_finger_angle(right_hand)
            if last_finger_angle is not None:
                diff = ang - last_finger_angle
                if diff > math.pi:
                    diff -= 2 * math.pi
                elif diff < -math.pi:
                    diff += 2 * math.pi
                wheel_angle = (wheel_angle + diff * 2) % (2 * math.pi)
                zoom_level = clamp(zoom_level + diff * zoom_sense, zoom_min, zoom_max)
            last_finger_angle = ang
        else:
            wheel_active = False
            last_finger_angle = None

        ok_now = detect_ok_gesture(right_hand, ok_touch_threshold) if right_hand else False
        if ok_now and not ok_prev:
            print("A-OK — closing inbox fallback.")
            running = False
        ok_prev = ok_now

        if right_hand and not wheel_active:
            pinch_now = is_pinching(right_hand, pinch_threshold)
            pos = get_pinch_position(right_hand)
            if pinch_now and pos:
                py = pos[1] * WINDOW_HEIGHT
                if is_pinching and last_pinch_y is not None:
                    dy = (py - last_pinch_y) * scroll_gain
                    scroll_y += dy
                    max_scroll_y = max(0, page_height - (WINDOW_HEIGHT - 80))
                    scroll_y = clamp(scroll_y, -max_scroll_y, 0)
                last_pinch_y = py
                is_pinching = True
            else:
                is_pinching = False
                last_pinch_y = None
        else:
            is_pinching = False
            last_pinch_y = None

        smooth_scroll_y += (scroll_y - smooth_scroll_y) * 0.35

        # draw
        screen.fill((240, 242, 245))
        # top bar
        pygame.draw.rect(screen, (255, 255, 255), (0, 0, WINDOW_WIDTH, 72))
        pygame.draw.rect(screen, (245, 247, 250), (12, 12, 220, 48), border_radius=12)
        pygame.draw.rect(screen, (245, 247, 250), (244, 20, WINDOW_WIDTH - 244 - 20, 32), border_radius=16)

        content_x = 220
        content_y = 80
        content_w = WINDOW_WIDTH - content_x
        content_h = WINDOW_HEIGHT - content_y

        # sidebar
        pygame.draw.rect(screen, (248, 249, 250), (0, content_y, 220, content_h))
        pygame.draw.rect(screen, (215, 227, 252), (16, content_y + 16, 188, 44), border_radius=22)
        sidebar_font = pygame.font.SysFont('arial', 18, bold=True)
        screen.blit(sidebar_font.render("Compose", True, (30, 64, 175)), (58, content_y + 27))
        pygame.draw.line(screen, (230, 232, 235), (220, content_y), (220, WINDOW_HEIGHT), 1)

        # list area
        s = zoom_level
        if abs(s - 1.0) > 0.001:
            zw = int(content_w * s)
            zh = int(content_h * s)
            zoom_surf = pygame.Surface((zw, zh), pygame.SRCALPHA).convert_alpha()
            zoom_surf.fill((255, 255, 255, 255))
            ph = _draw_inbox_list(zoom_surf, 0, 0, zw, zh, emails, smooth_scroll_y * s)
            scaled = pygame.transform.smoothscale(zoom_surf, (content_w, content_h)).convert_alpha()
            screen.blit(scaled, (content_x, content_y))
            page_height = int(ph / s)
        else:
            ph = _draw_inbox_list(screen, content_x, content_y, content_w, content_h, emails, smooth_scroll_y)
            page_height = ph

        # status
        font = pygame.font.SysFont('arial', 16, bold=True)
        status = f"Zoom {zoom_level:.2f}x • Scroll {int(-smooth_scroll_y)}/{max(0, page_height - (WINDOW_HEIGHT - 80))}"
        screen.blit(font.render(status, True, (90, 98, 110)), (16, WINDOW_HEIGHT - 28))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()


# ==============================
# Entrypoint
# ==============================
if __name__ == "__main__":
    if "--email-fallback" in sys.argv:
        email_fallback_main()
    else:
        carousel_main()

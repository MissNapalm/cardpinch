import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
from collections import deque
import subprocess
import time

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

        # NEW: scroll sensitivity (higher = more scroll per movement)
        self.scroll_gain = 2.0

        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.selected_card = None
        self.selected_category = None
        self.zoom_progress = 0.0
        self.zoom_target = 0.0
        self.finger_smoother = FingerSmoother(window_size=5)
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
        self.pinch_threshold = 0.08
        self.pinch_count = 0
        self.last_pinch_time = 0
        self.double_pinch_window = 0.4
        self.pinch_prev = False
        self.pinch_start_pos = None
        self.movement_threshold = 10
        self.thumbs_up_prev = False
        self.thumbs_up_cooldown = 0
        self.browser_process = None
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

def detect_thumbs_up(landmarks):
    """Detect thumbs up: thumb extended up, all other fingers folded"""
    if not landmarks:
        return False
    
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_mcp = landmarks[2]
    thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y
    
    index_fold = landmarks[8].y  > landmarks[6].y  - 0.03
    middle_fold = landmarks[12].y > landmarks[10].y - 0.03
    ring_fold   = landmarks[16].y > landmarks[14].y - 0.03
    pinky_fold  = landmarks[20].y > landmarks[18].y - 0.03
    
    return thumb_up and index_fold and middle_fold and ring_fold and pinky_fold

def detect_ok_gesture(landmarks, touch_thresh=0.035):
    if not landmarks: 
        return False
    a = landmarks[4]; b = landmarks[8]
    touching = math.hypot(a.x - b.x, a.y - b.y) < touch_thresh
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_ext   = is_finger_extended(landmarks, 16, 14)
    pinky_ext  = is_finger_extended(landmarks, 20, 18)
    return touching and middle_ext and ring_ext and pinky_ext

def get_hand_center(landmarks): 
    return landmarks[9]

def calculate_finger_angle(landmarks):
    c = get_hand_center(landmarks)
    idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)

def launch_browser_window(state):
    if state.browser_process is None or state.browser_process.poll() is not None:
        try:
            state.browser_process = subprocess.Popen([sys.executable, "gesture_webview.py"])
            print("✓ BROWSER WINDOW LAUNCHED!")
        except Exception as e:
            print(f"Error launching browser: {e}")
            print("Make sure 'gesture_webview.py' exists in the same directory!")

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

def draw_cards(surface, center_x, center_y, card_offset, category_idx, selected_card=None, selected_category=None, zoom_progress=0.0, window_width=1280, gui_scale=1.0, base_w=280, base_h=280, base_spacing=50):
    app_names = CAROUSEL_CATEGORIES[category_idx]
    card_rects = []
    sw = int(base_w * gui_scale)
    ss = int(base_spacing * gui_scale)
    stride = sw + ss
    first_vis = int((-card_offset - window_width // 2) / stride) - 1
    last_vis = int((-card_offset + window_width // 2) / stride) + 2
    first_vis = max(0, first_vis)
    last_vis = min(CARD_COUNT, last_vis)
    for i in range(first_vis, last_vis):
        x = int(center_x + (i * stride) + card_offset)
        y = int(center_y)
        sel = (selected_card == i and selected_category == category_idx)
        if not sel:
            rect = draw_app_icon(surface, app_names[i], x, y, base_w, base_h, False, 1.0, gui_scale)
            card_rects.append((rect, i, category_idx))
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

def main():
    pygame.init()
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Gesture Carousel • Thumbs up Browser to launch")
    clock = pygame.time.Clock()
    
    print("=" * 50)
    print("GESTURE CAROUSEL STARTED")
    print("=" * 50)
    print("Initializing camera...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera initialized successfully")
    print("Initializing hand tracking...")
    
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
    print("Hand tracking ready")
    print("\nINSTRUCTIONS:")
    print("1. Pinch-tap a card to select it")
    print("2. Thumbs up on 'Browser' to launch window")
    print("3. Three-finger gesture to resize GUI")
    print("4. A-OK to reset GUI zoom to 1.0x")
    print("=" * 50)
    
    state = HandState()
    tap_to_check = None
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
        
        # Thumbs up cooldown
        if state.thumbs_up_cooldown > 0:
            state.thumbs_up_cooldown -= 1
        
        # Thumbs up detection to launch Browser card (card=2, category=0)
        thumbs_up_now = detect_thumbs_up(right_hand) if right_hand else False
        if thumbs_up_now and not state.thumbs_up_prev and state.thumbs_up_cooldown == 0:
            if state.selected_card == 2 and state.selected_category == 0:
                launch_browser_window(state)
                print("👍 THUMBS UP ON BROWSER - LAUNCHING WINDOW!")
                state.thumbs_up_cooldown = 30
            else:
                print(f"👍 Thumbs up detected, but not on Browser (card={state.selected_card}, cat={state.selected_category})")
        state.thumbs_up_prev = thumbs_up_now

        pinch_now = is_pinching(right_hand, state.pinch_threshold) if right_hand else False

        # Three-finger wheel gesture
        if right_hand:
            if detect_three_finger_gesture(right_hand):
                if not state.wheel_active:
                    hc = get_hand_center(right_hand)
                    state.wheel_active = True
                    state.wheel_center_x = int(hc.x * WINDOW_WIDTH)
                    state.wheel_center_y = int(hc.y * WINDOW_HEIGHT)
                    state.last_finger_angle = None
                ang = calculate_finger_angle(right_hand)
                if state.last_finger_angle is not None:
                    diff = ang - state.last_finger_angle
                    if diff > math.pi: diff -= 2 * math.pi
                    elif diff < -math.pi: diff += 2 * math.pi
                    state.wheel_angle = (state.wheel_angle + diff * 2) % (2 * math.pi)
                    state.gui_scale = clamp(
                        state.gui_scale + diff * state.gui_scale_sensitivity, 
                        state.gui_scale_min, state.gui_scale_max
                    )
                state.last_finger_angle = ang
            else:
                state.wheel_active = False
                state.last_finger_angle = None

        # Pinch scrolling + tap detection (first path)
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
                    
                    if state.pinch_start_pos:
                        total_dx = px - state.pinch_start_pos[0]
                        total_dy = py - state.pinch_start_pos[1]
                        total_movement = math.hypot(total_dx, total_dy)
                        
                        if total_movement > state.movement_threshold:
                            # APPLY SCROLL GAIN HERE
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
                if state.pinch_start_pos and state.last_pinch_x is not None:
                    total_dx = state.last_pinch_x - state.pinch_start_pos[0]
                    total_dy = state.last_pinch_y - state.pinch_start_pos[1]
                    total_movement = math.hypot(total_dx, total_dy)
                    
                    if total_movement <= state.movement_threshold:
                        tap_to_check = state.pinch_start_pos
                        
                        current_time = time.time()
                        time_since_last = current_time - state.last_pinch_time
                        
                        print(f"TAP! Δt={time_since_last:.3f}s, Selected: card={state.selected_card}, cat={state.selected_category}")
                        
                        if time_since_last < state.double_pinch_window and time_since_last > 0.05:
                            if state.selected_card == 2 and state.selected_category == 0:
                                launch_browser_window(state)
                                print("✓✓✓ DOUBLE TAP ON BROWSER - LAUNCHING! ✓✓✓")
                            else:
                                print(f"Double tap, but not on Browser (card={state.selected_card}, cat={state.selected_category})")
                        
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

        # Three-finger wheel gesture handled above

        # Pinch scrolling + tap detection (second path—kept for stability)
        if right_hand and not state.wheel_active:
            pos = get_pinch_position(right_hand)
            if pinch_now and pos:
                px = pos[0] * WINDOW_WIDTH
                py = pos[1] * WINDOW_HEIGHT
                if state.is_pinching and state.last_pinch_x is not None:
                    dx = px - state.last_pinch_x
                    dy = py - state.last_pinch_y
                    
                    if abs(dx) > 5 or abs(dy) > 5:
                        # APPLY SCROLL GAIN HERE AS WELL
                        state.card_offset += dx * state.scroll_gain
                        state.category_offset += dy * state.scroll_gain
                        stride_x = int((CARD_WIDTH + CARD_SPACING) * state.gui_scale)
                        min_x = -(CARD_COUNT - 1) * stride_x
                        state.card_offset = clamp(state.card_offset, min_x, 0)
                        row_stride = int(ROW_BASE_SPACING * state.gui_scale)
                        min_y = -(NUM_CATEGORIES - 1) * row_stride
                        state.category_offset = clamp(state.category_offset, min_y, 0)
                    else:
                        tap_to_check = (px, py)
                else:
                    tap_to_check = (px, py)
                
                state.last_pinch_x = px
                state.last_pinch_y = py
                state.is_pinching = True
            else:
                if state.is_pinching:
                    current_time = time.time()
                    time_since_last = current_time - state.last_pinch_time
                    print(f"Pinch released! Δt={time_since_last:.3f}s, Selected: card={state.selected_card}, cat={state.selected_category}")
                    
                    if time_since_last < state.double_pinch_window:
                        print("Double pinch detected!")
                        if state.selected_card == 2 and state.selected_category == 0:
                            launch_browser_window(state)
                            print("✓ Double pinch on Browser - launching window!")
                        else:
                            print(f"✗ Not on Browser (need card=2, cat=0, got card={state.selected_card}, cat={state.selected_category})")
                    
                    state.last_pinch_time = current_time
                
                state.is_pinching = False
                state.last_pinch_x = None
                state.last_pinch_y = None
        else:
            state.is_pinching = False
            state.last_pinch_x = None
            state.last_pinch_y = None
        
        state.pinch_prev = pinch_now

        state.zoom_progress += (state.zoom_target - state.zoom_progress) * 0.15
        if abs(state.zoom_progress - state.zoom_target) < 0.01:
            state.zoom_progress = state.zoom_target

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
            all_rects += draw_cards(screen, cx, int(y), state.smooth_card_offset, cat_idx, state.selected_card, state.selected_category, state.zoom_progress, WINDOW_WIDTH, state.gui_scale, CARD_WIDTH, CARD_HEIGHT, CARD_SPACING)
        
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
        
        draw_wheel(screen, state, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Draw camera feed
        frame_surface = pygame.surfarray.make_surface(cv2.transpose(rgb))
        frame_surface = pygame.transform.scale(frame_surface, (320, 240))
        screen.blit(frame_surface, (WINDOW_WIDTH - 330, 10))

        # Draw hand tracking
        if right_hand:
            tt = right_hand[4]
            it = right_hand[8]
            (tx, ty), (ix, iy) = state.finger_smoother.update(
                lm_to_screen(tt, WINDOW_WIDTH, WINDOW_HEIGHT),
                lm_to_screen(it, WINDOW_WIDTH, WINDOW_HEIGHT)
            )
            is_p = pinch_now
            if not state.wheel_active and is_p:
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
            status = "Ready • Thumbs up Browser to launch"
        screen.blit(font.render(status, True, (255, 255, 255)), (30, 30))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

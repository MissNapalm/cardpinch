import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
from collections import deque

# Constants
CARD_COUNT = 7
CARD_WIDTH = 280
CARD_HEIGHT = 280
CARD_SPACING = 50
CAROUSEL_CATEGORIES = [
    ["Mail", "Music", "Safari", "Messages", "Calendar", "Maps", "Camera"],
    ["Photos", "Notes", "Reminders", "Clock", "Weather", "Stocks", "News"],
    ["YouTube", "Netflix", "Twitch", "Spotify", "Podcasts", "Books", "Games"]
]
NUM_CATEGORIES = len(CAROUSEL_CATEGORIES)

# Color schemes for apps (RGB)
APP_COLORS = {
    "Mail": (74, 144, 226),
    "Music": (252, 61, 86),
    "Safari": (35, 142, 250),
    "Messages": (76, 217, 100),
    "Calendar": (252, 61, 57),
    "Maps": (89, 199, 249),
    "Camera": (138, 138, 142),
    "Photos": (252, 203, 47),
    "Notes": (255, 214, 10),
    "Reminders": (255, 69, 58),
    "Clock": (30, 30, 30),
    "Weather": (99, 204, 250),
    "Stocks": (30, 30, 30),
    "News": (252, 61, 86),
    "YouTube": (255, 0, 0),
    "Netflix": (229, 9, 20),
    "Twitch": (145, 70, 255),
    "Spotify": (30, 215, 96),
    "Podcasts": (146, 72, 223),
    "Books": (255, 124, 45),
    "Games": (255, 45, 85)
}

mp_hands = mp.solutions.hands

class FingerSmoother:
    """Smooths finger positions using moving average"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.thumb_history = deque(maxlen=window_size)
        self.index_history = deque(maxlen=window_size)
    
    def update(self, thumb_pos, index_pos):
        self.thumb_history.append(thumb_pos)
        self.index_history.append(index_pos)
        
        thumb_smooth = (
            sum(p[0] for p in self.thumb_history) / len(self.thumb_history),
            sum(p[1] for p in self.thumb_history) / len(self.thumb_history)
        )
        index_smooth = (
            sum(p[0] for p in self.index_history) / len(self.index_history),
            sum(p[1] for p in self.index_history) / len(self.index_history)
        )
        
        return thumb_smooth, index_smooth
    
    def reset(self):
        self.thumb_history.clear()
        self.index_history.clear()

class HandState:
    def __init__(self):
        self.card_offset = 0.0
        self.category_offset = 0.0
        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.selected_card = None
        self.selected_category = None
        self.zoom_progress = 0.0
        self.zoom_target = 0.0
        self.finger_smoother = FingerSmoother(window_size=5)
        self.smooth_card_offset = 0.0
        self.smooth_category_offset = 0.0
        
        # Wheel mode variables
        self.wheel_active = False
        self.wheel_angle = math.pi
        self.last_finger_angle = None
        self.wheel_center_x = 0
        self.wheel_center_y = 0
        self.wheel_radius = 110

def get_pinch_distance(landmarks):
    if not landmarks:
        return None
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return dist

def is_pinching(landmarks, threshold=0.08):
    dist = get_pinch_distance(landmarks)
    return dist is not None and dist < threshold

def get_pinch_position(landmarks):
    if not landmarks:
        return None
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return ((thumb_tip.x + index_tip.x) / 2, (thumb_tip.y + index_tip.y) / 2)

def is_finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

def detect_three_finger_gesture(landmarks):
    """Detect three finger gesture (thumb, index, middle extended)"""
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    
    thumb_extended = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
    index_extended = is_finger_extended(landmarks, 8, 6)
    middle_extended = is_finger_extended(landmarks, 12, 10)
    ring_folded = landmarks[16].y > landmarks[14].y - 0.02
    pinky_folded = landmarks[20].y > landmarks[18].y - 0.02
    
    return thumb_extended and index_extended and middle_extended and ring_folded and pinky_folded

def get_hand_center(landmarks):
    return landmarks[9]

def calculate_finger_angle(landmarks):
    hand_center = get_hand_center(landmarks)
    index_tip = landmarks[8]
    dx = index_tip.x - hand_center.x
    dy = index_tip.y - hand_center.y
    return math.atan2(dy, dx)

def draw_app_icon(surface, app_name, x, y, width, height, is_selected=False, zoom_scale=1.0):
    color = APP_COLORS.get(app_name, (100, 100, 100))
    
    if is_selected:
        width = int(width * zoom_scale)
        height = int(height * zoom_scale)
    
    border_radius = 50
    rect = pygame.Rect(x - width//2, y - height//2, width, height)
    color = tuple(min(255, int(c * 1.2)) for c in color)
    
    pygame.draw.rect(surface, color, rect, border_radius=border_radius)
    
    if is_selected:
        selection_rect = pygame.Rect(rect.x - 6, rect.y - 6, rect.width + 12, rect.height + 12)
        pygame.draw.rect(surface, (255, 255, 255), selection_rect, width=8, border_radius=border_radius)
    
    icon_font = pygame.font.Font(None, int(120 * (width / 280)))
    icon_text = icon_font.render(app_name[0], True, (255, 255, 255, 180))
    icon_rect = icon_text.get_rect(center=(x, y - 20))
    surface.blit(icon_text, icon_rect)
    
    text_size = int(36 * (width / 280))
    font = pygame.font.Font(None, text_size)
    text = font.render(app_name, True, (255, 255, 255))
    text_rect = text.get_rect(center=(x, y + 60))
    surface.blit(text, text_rect)
    
    return rect

def draw_cards(surface, center_x, center_y, card_offset, category_idx, selected_card=None, selected_category=None, zoom_progress=0.0, window_width=1280):
    app_names = CAROUSEL_CATEGORIES[category_idx]
    card_rects = []
    
    first_visible_card = int((-card_offset - window_width // 2) / (CARD_WIDTH + CARD_SPACING)) - 1
    last_visible_card = int((-card_offset + window_width // 2) / (CARD_WIDTH + CARD_SPACING)) + 2
    
    first_visible_card = max(0, first_visible_card)
    last_visible_card = min(CARD_COUNT, last_visible_card)
    
    for i in range(first_visible_card, last_visible_card):
        x = int(center_x + (i * (CARD_WIDTH + CARD_SPACING)) + card_offset)
        y = int(center_y)
        
        is_selected = (selected_card == i and selected_category == category_idx)
        
        if not is_selected:
            rect = draw_app_icon(surface, app_names[i], x, y, CARD_WIDTH, CARD_HEIGHT, False, 1.0)
            card_rects.append((rect, i, category_idx))
    
    for i in range(first_visible_card, last_visible_card):
        x = int(center_x + (i * (CARD_WIDTH + CARD_SPACING)) + card_offset)
        y = int(center_y)
        
        is_selected = (selected_card == i and selected_category == category_idx)
        
        if is_selected:
            zoom_scale = 1.0 + (zoom_progress * 0.3)
            rect = draw_app_icon(surface, app_names[i], x, y, CARD_WIDTH, CARD_HEIGHT, is_selected, zoom_scale)
            card_rects.append((rect, i, category_idx))
    
    return card_rects

def draw_wheel(surface, state, window_width, window_height):
    """Draw futuristic wheel overlay"""
    if not state.wheel_active:
        return
    
    cx = state.wheel_center_x
    cy = state.wheel_center_y
    radius = state.wheel_radius
    white = (255, 255, 255)
    
    # Outer glow rings
    for i in range(5):
        r = radius + 15 + i * 10
        opacity = int(100 - (i * 20))
        color = (255, 255, 255, opacity)
        s = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        pygame.draw.circle(s, (*white, opacity), (cx, cy), r, 2)
        surface.blit(s, (0, 0))
    
    # Main circle
    pygame.draw.circle(surface, white, (cx, cy), radius, 4)
    
    # Inner circle
    pygame.draw.circle(surface, white, (cx, cy), radius - 20, 2)
    
    # Progress arc
    angle_deg = int(math.degrees(state.wheel_angle) % 360)
    
    # Draw arc segments
    num_segments = 48
    for i in range(num_segments):
        if i < int((state.wheel_angle / (2 * math.pi)) * num_segments):
            start_angle = math.radians(i * 360 / num_segments) - math.pi/2
            end_angle = math.radians((i + 1) * 360 / num_segments) - math.pi/2
            
            start_x = cx + int((radius - 10) * math.cos(start_angle))
            start_y = cy + int((radius - 10) * math.sin(start_angle))
            end_x = cx + int((radius - 10) * math.cos(end_angle))
            end_y = cy + int((radius - 10) * math.sin(end_angle))
            
            pygame.draw.line(surface, white, (start_x, start_y), (end_x, end_y), 6)
    
    # Pointer
    pointer_length = radius - 30
    pointer_x = cx + int(pointer_length * math.cos(state.wheel_angle))
    pointer_y = cy + int(pointer_length * math.sin(state.wheel_angle))
    
    pygame.draw.line(surface, white, (cx, cy), (pointer_x, pointer_y), 3)
    pygame.draw.circle(surface, white, (pointer_x, pointer_y), 6)
    pygame.draw.circle(surface, white, (cx, cy), 8)
    
    # Display angle
    font = pygame.font.Font(None, 40)
    text = font.render(f"{angle_deg:03d}°", True, white)
    text_rect = text.get_rect(center=(cx, cy + radius + 50))
    
    # Draw text background
    bg_rect = pygame.Rect(text_rect.x - 10, text_rect.y - 5, text_rect.width + 20, text_rect.height + 10)
    pygame.draw.rect(surface, (20, 20, 20), bg_rect)
    pygame.draw.rect(surface, white, bg_rect, 2)
    
    surface.blit(text, text_rect)

def main():
    pygame.init()
    
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hand Gesture Carousel with Wheel")
    clock = pygame.time.Clock()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    
    state = HandState()
    ROW_SPACING = CARD_HEIGHT + 80
    
    tap_to_check = None
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    right_hand = hand_landmarks.landmark
        
        if right_hand is None:
            state.finger_smoother.reset()
            state.wheel_active = False
            state.last_finger_angle = None
        
        # Handle wheel mode with three-finger gesture (no card selection needed)
        if right_hand:
            three_finger = detect_three_finger_gesture(right_hand)
            
            if three_finger:
                if not state.wheel_active:
                    state.wheel_active = True
                    hand_center = get_hand_center(right_hand)
                    state.wheel_center_x = int(hand_center.x * WINDOW_WIDTH)
                    state.wheel_center_y = int(hand_center.y * WINDOW_HEIGHT)
                    state.last_finger_angle = None
                
                # Update wheel rotation
                current_angle = calculate_finger_angle(right_hand)
                if state.last_finger_angle is not None:
                    angle_diff = current_angle - state.last_finger_angle
                    
                    if angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    elif angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    
                    state.wheel_angle += angle_diff * 2
                    state.wheel_angle = state.wheel_angle % (2 * math.pi)
                    
                    # Update zoom based on rotation direction
                    # Clockwise (positive angle_diff) = zoom in, Counter-clockwise = zoom out
                    if angle_diff > 0:  # Clockwise
                        state.zoom_target = min(1.0, state.zoom_target + 0.02)
                    elif angle_diff < 0:  # Counter-clockwise
                        state.zoom_target = max(0.0, state.zoom_target - 0.02)
                
                state.last_finger_angle = current_angle
            else:
                state.wheel_active = False
                state.last_finger_angle = None
        
        # Process pinch gesture for scrolling
        current_pinch = False
        if right_hand and not state.wheel_active:
            current_pinch = is_pinching(right_hand, threshold=0.08)
            pinch_pos = get_pinch_position(right_hand)
            
            if current_pinch and pinch_pos:
                pinch_x = pinch_pos[0] * WINDOW_WIDTH
                pinch_y = pinch_pos[1] * WINDOW_HEIGHT
                
                if state.is_pinching and state.last_pinch_x is not None:
                    delta_x = pinch_x - state.last_pinch_x
                    delta_y = pinch_y - state.last_pinch_y
                    
                    state.card_offset += delta_x
                    state.category_offset += delta_y
                    
                    max_card_offset = 0
                    min_card_offset = -(CARD_COUNT - 1) * (CARD_WIDTH + CARD_SPACING)
                    state.card_offset = max(min_card_offset, min(max_card_offset, state.card_offset))
                    
                    max_category_offset = 0
                    min_category_offset = -(NUM_CATEGORIES - 1) * ROW_SPACING
                    state.category_offset = max(min_category_offset, min(max_category_offset, state.category_offset))
                else:
                    # Just started pinching - record position for tap detection
                    tap_to_check = (pinch_x, pinch_y)
                
                state.last_pinch_x = pinch_x
                state.last_pinch_y = pinch_y
                state.is_pinching = True
            else:
                if state.is_pinching:
                    # Released - tap_to_check already set when pinch started
                    pass
                
                state.is_pinching = False
                state.last_pinch_x = None
                state.last_pinch_y = None
        else:
            state.is_pinching = False
            state.last_pinch_x = None
            state.last_pinch_y = None
        
        # Animate zoom
        ZOOM_SPEED = 0.15
        state.zoom_progress += (state.zoom_target - state.zoom_progress) * ZOOM_SPEED
        if abs(state.zoom_progress - state.zoom_target) < 0.01:
            state.zoom_progress = state.zoom_target
        
        # Smooth scroll positions
        SCROLL_SMOOTHING = 0.25
        state.smooth_card_offset += (state.card_offset - state.smooth_card_offset) * SCROLL_SMOOTHING
        state.smooth_category_offset += (state.category_offset - state.smooth_category_offset) * SCROLL_SMOOTHING
        
        screen.fill((20, 20, 30))
        
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        
        all_card_rects = []
        
        first_visible_cat = int(-state.smooth_category_offset / ROW_SPACING) - 1
        last_visible_cat = int((-state.smooth_category_offset + WINDOW_HEIGHT) / ROW_SPACING) + 2
        
        first_visible_cat = max(0, first_visible_cat)
        last_visible_cat = min(NUM_CATEGORIES, last_visible_cat)
        
        for cat_idx in range(first_visible_cat, last_visible_cat):
            y_pos = center_y + (cat_idx * ROW_SPACING) + state.smooth_category_offset
            card_rects = draw_cards(screen, center_x, y_pos, state.smooth_card_offset, cat_idx,
                                   state.selected_card, state.selected_category, state.zoom_progress, WINDOW_WIDTH)
            all_card_rects.extend(card_rects)
        
        # Check for tap selection
        if tap_to_check:
            tap_x, tap_y = tap_to_check
            for rect, card_idx, cat_idx in all_card_rects:
                if rect.collidepoint(tap_x, tap_y):
                    state.selected_card = card_idx
                    state.selected_category = cat_idx
                    state.zoom_target = 1.0
                    print(f"Selected: {CAROUSEL_CATEGORIES[state.selected_category][state.selected_card]}")
                    break
            tap_to_check = None
        
        # Draw wheel overlay
        draw_wheel(screen, state, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Draw camera feed
        frame_surface = pygame.surfarray.make_surface(cv2.transpose(rgb))
        frame_surface = pygame.transform.scale(frame_surface, (320, 240))
        screen.blit(frame_surface, (WINDOW_WIDTH - 330, 10))
        
        # Draw hand cursors with smoothing (hide when wheel is active)
        if right_hand and not state.wheel_active:
            thumb_tip = right_hand[4]
            index_tip = right_hand[8]
            
            thumb_raw = (thumb_tip.x * WINDOW_WIDTH, thumb_tip.y * WINDOW_HEIGHT)
            index_raw = (index_tip.x * WINDOW_WIDTH, index_tip.y * WINDOW_HEIGHT)
            
            thumb_smooth, index_smooth = state.finger_smoother.update(thumb_raw, index_raw)
            
            thumb_x = int(thumb_smooth[0])
            thumb_y = int(thumb_smooth[1])
            index_x = int(index_smooth[0])
            index_y = int(index_smooth[1])
            
            is_pinch = is_pinching(right_hand, threshold=0.08)
            
            if is_pinch:
                pygame.draw.line(screen, (255, 255, 255), (thumb_x, thumb_y), (index_x, index_y), 2)
            
            pygame.draw.circle(screen, (255, 255, 255), (thumb_x, thumb_y), 8)
            pygame.draw.circle(screen, (255, 255, 255), (index_x, index_y), 8)
        
        # Draw status
        font = pygame.font.Font(None, 48)
        if state.wheel_active:
            status = "WHEEL MODE"
        elif state.is_pinching:
            status = "PINCHED"
        else:
            status = "Ready"
        text = font.render(status, True, (255, 255, 255))
        screen.blit(text, (30, 30))
        
        pygame.display.flip()
        clock.tick(60)
    
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

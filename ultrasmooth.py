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
        """Add new positions and return smoothed positions"""
        self.thumb_history.append(thumb_pos)
        self.index_history.append(index_pos)
        
        # Calculate average
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
        """Clear history"""
        self.thumb_history.clear()
        self.index_history.clear()

class HandState:
    def __init__(self):
        self.card_offset = 0.0  # Horizontal scroll offset in pixels
        self.category_offset = 0.0  # Vertical scroll offset in pixels
        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.selected_card = None
        self.selected_category = None
        self.zoom_progress = 0.0
        self.zoom_target = 0.0
        self.finger_smoother = FingerSmoother(window_size=5)

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

def draw_app_icon(surface, app_name, x, y, width, height, is_selected=False, zoom_scale=1.0):
    """Draw a stylized app icon"""
    # Get color for app
    color = APP_COLORS.get(app_name, (100, 100, 100))
    
    # Apply zoom if selected
    if is_selected:
        width = int(width * zoom_scale)
        height = int(height * zoom_scale)
    
    # Draw rounded rectangle background
    border_radius = 50
    rect = pygame.Rect(x - width//2, y - height//2, width, height)
    
    # Brighten color
    color = tuple(min(255, int(c * 1.2)) for c in color)
    
    pygame.draw.rect(surface, color, rect, border_radius=border_radius)
    
    # Draw white selection border if selected
    if is_selected:
        selection_rect = pygame.Rect(rect.x - 6, rect.y - 6, rect.width + 12, rect.height + 12)
        pygame.draw.rect(surface, (255, 255, 255), selection_rect, width=8, border_radius=border_radius)
    
    # Draw first letter as icon
    icon_font = pygame.font.Font(None, int(120 * (width / 280)))
    icon_text = icon_font.render(app_name[0], True, (255, 255, 255, 180))
    icon_rect = icon_text.get_rect(center=(x, y - 20))
    surface.blit(icon_text, icon_rect)
    
    # Add white app name text
    text_size = int(36 * (width / 280))
    font = pygame.font.Font(None, text_size)
    text = font.render(app_name, True, (255, 255, 255))
    text_rect = text.get_rect(center=(x, y + 60))
    surface.blit(text, text_rect)
    
    return rect

def draw_cards(surface, center_x, center_y, card_offset, category_idx, selected_card=None, selected_category=None, zoom_progress=0.0):
    """Draw a row of cards"""
    app_names = CAROUSEL_CATEGORIES[category_idx]
    card_rects = []
    
    # First pass: draw non-selected cards
    for i in range(CARD_COUNT):
        x = int(center_x + (i * (CARD_WIDTH + CARD_SPACING)) + card_offset)
        y = int(center_y)
        
        is_selected = (selected_card == i and selected_category == category_idx)
        
        if not is_selected:
            rect = draw_app_icon(surface, app_names[i], x, y, CARD_WIDTH, CARD_HEIGHT, False, 1.0)
            card_rects.append((rect, i, category_idx))
    
    # Second pass: draw selected card on top
    for i in range(CARD_COUNT):
        x = int(center_x + (i * (CARD_WIDTH + CARD_SPACING)) + card_offset)
        y = int(center_y)
        
        is_selected = (selected_card == i and selected_category == category_idx)
        
        if is_selected:
            zoom_scale = 1.0 + (zoom_progress * 0.3)
            rect = draw_app_icon(surface, app_names[i], x, y, CARD_WIDTH, CARD_HEIGHT, is_selected, zoom_scale)
            card_rects.append((rect, i, category_idx))
    
    return card_rects

def main():
    # Initialize Pygame
    pygame.init()
    
    # Set up display
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hand Gesture Carousel")
    clock = pygame.time.Clock()
    
    # Initialize camera and hand tracking
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
    
    running = True
    while running:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Find right hand
        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    right_hand = hand_landmarks.landmark
        
        # Reset smoother if no hand detected
        if right_hand is None:
            state.finger_smoother.reset()
        
        # Process pinch gesture
        current_pinch = False
        if right_hand:
            current_pinch = is_pinching(right_hand, threshold=0.08)
            pinch_pos = get_pinch_position(right_hand)
            
            if current_pinch and pinch_pos:
                pinch_x = pinch_pos[0] * WINDOW_WIDTH
                pinch_y = pinch_pos[1] * WINDOW_HEIGHT
                
                if state.is_pinching and state.last_pinch_x is not None:
                    # Calculate delta from last frame
                    delta_x = pinch_x - state.last_pinch_x
                    delta_y = pinch_y - state.last_pinch_y
                    
                    # Apply movement directly to offsets
                    state.card_offset += delta_x
                    state.category_offset += delta_y
                    
                    # Clamp horizontal scroll
                    max_card_offset = 0
                    min_card_offset = -(CARD_COUNT - 1) * (CARD_WIDTH + CARD_SPACING)
                    state.card_offset = max(min_card_offset, min(max_card_offset, state.card_offset))
                    
                    # Clamp vertical scroll (3 categories)
                    max_category_offset = 0
                    min_category_offset = -(NUM_CATEGORIES - 1) * ROW_SPACING
                    state.category_offset = max(min_category_offset, min(max_category_offset, state.category_offset))
                
                state.last_pinch_x = pinch_x
                state.last_pinch_y = pinch_y
                state.is_pinching = True
            else:
                # Pinch released - check for tap
                if state.is_pinching and state.last_pinch_x is not None:
                    # This was a quick tap, check for selection after drawing
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
        
        # Clear screen
        screen.fill((20, 20, 30))
        
        # Draw all categories
        center_x = WINDOW_WIDTH // 2
        center_y = WINDOW_HEIGHT // 2
        
        all_card_rects = []
        
        for cat_idx in range(NUM_CATEGORIES):
            y_pos = center_y + (cat_idx * ROW_SPACING) + state.category_offset
            card_rects = draw_cards(screen, center_x, y_pos, state.card_offset, cat_idx,
                                   state.selected_card, state.selected_category, state.zoom_progress)
            all_card_rects.extend(card_rects)
        
        # Draw camera feed in corner
        frame_surface = pygame.surfarray.make_surface(cv2.transpose(rgb))
        frame_surface = pygame.transform.scale(frame_surface, (320, 240))
        screen.blit(frame_surface, (WINDOW_WIDTH - 330, 10))
        
        # Draw hand cursors with smoothing
        if right_hand:
            thumb_tip = right_hand[4]
            index_tip = right_hand[8]
            
            # Get raw positions
            thumb_raw = (thumb_tip.x * WINDOW_WIDTH, thumb_tip.y * WINDOW_HEIGHT)
            index_raw = (index_tip.x * WINDOW_WIDTH, index_tip.y * WINDOW_HEIGHT)
            
            # Apply smoothing
            thumb_smooth, index_smooth = state.finger_smoother.update(thumb_raw, index_raw)
            
            thumb_x = int(thumb_smooth[0])
            thumb_y = int(thumb_smooth[1])
            index_x = int(index_smooth[0])
            index_y = int(index_smooth[1])



            
            is_pinch = is_pinching(right_hand, threshold=0.08)
            
            if is_pinch:
                pygame.draw.line(screen, (255, 255, 255), (thumb_x, thumb_y), (index_x, index_y), 2)
            
            # Draw white dots on fingertips
            pygame.draw.circle(screen, (255, 255, 255), (thumb_x, thumb_y), 8)
            pygame.draw.circle(screen, (255, 255, 255), (index_x, index_y), 8)
        
        # Draw status
        font = pygame.font.Font(None, 48)
        status = "PINCHED" if state.is_pinching else "Ready"
        text = font.render(status, True, (255, 255, 255))
        screen.blit(text, (30, 30))
        
        pygame.display.flip()
        clock.tick(60)
    
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

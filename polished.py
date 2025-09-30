import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys

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

class HandState:
    def __init__(self):
        self.card_index = 3
        self.card_anim_pos = 3.0
        self.grabbed = False
        self.grab_start_x = None
        self.grab_start_y = None
        self.grab_start_card_pos = None
        self.grab_start_category_pos = None
        self.selected_card = None
        self.velocity = 0.0
        self.vertical_velocity = 0.0
        self.category_index = 1
        self.category_anim_pos = 1.0

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

def draw_app_icon(surface, app_name, x, y, width, height, is_centered=False):
    """Draw a stylized app icon"""
    # Get color for app
    color = APP_COLORS.get(app_name, (100, 100, 100))
    
    # Draw rounded rectangle background
    border_radius = 50
    rect = pygame.Rect(x - width//2, y - height//2, width, height)
    
    # Make centered card slightly larger and brighter
    if is_centered:
        scale = 1.15
        new_width = int(width * scale)
        new_height = int(height * scale)
        rect = pygame.Rect(x - new_width//2, y - new_height//2, new_width, new_height)
        # Brighten color
        color = tuple(min(255, int(c * 1.2)) for c in color)
    
    pygame.draw.rect(surface, color, rect, border_radius=border_radius)
    
    # Add white app name text
    font = pygame.font.Font(None, 42)
    text = font.render(app_name, True, (255, 255, 255))
    text_rect = text.get_rect(center=(x, y + 50))
    surface.blit(text, text_rect)
    
    # Draw simple icon representation (first letter in larger font)
    icon_font = pygame.font.Font(None, 120)
    icon_text = icon_font.render(app_name[0], True, (255, 255, 255, 128))
    icon_rect = icon_text.get_rect(center=(x, y - 20))
    surface.blit(icon_text, icon_rect)

def draw_cards(surface, center_x, center_y, anim_pos, category_idx=0):
    app_names = CAROUSEL_CATEGORIES[category_idx]
    for i in range(CARD_COUNT):
        offset = (i - anim_pos) * (CARD_WIDTH + CARD_SPACING)
        x = int(center_x + offset)
        y = int(center_y)
        
        is_centered = round(anim_pos) == i
        draw_app_icon(surface, app_names[i], x, y, CARD_WIDTH, CARD_HEIGHT, is_centered)

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
    pinch_state = {}
    
    # Inertia parameters
    FRICTION = 0.92
    VERTICAL_FRICTION = 0.92
    MIN_VELOCITY = 0.01
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
        
        # Separate left and right hands
        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Check if it's the right hand (will be "Right" because we flipped the frame)
                if handedness.classification[0].label == "Right":
                    right_hand = hand_landmarks.landmark
        
        # Process pinch gestures (only right hand)
        any_pinching = False
        
        if right_hand:
            is_pinch = is_pinching(right_hand, threshold=0.08)
            pinch_pos = get_pinch_position(right_hand)
            
            if is_pinch and pinch_pos:
                any_pinching = True
                pinch_x = pinch_pos[0] * WINDOW_WIDTH
                pinch_y = pinch_pos[1] * WINDOW_HEIGHT
                
                if 'right' not in pinch_state:
                    pinch_state['right'] = {
                        'start_x': pinch_x,
                        'start_y': pinch_y,
                        'frames_held': 0
                    }
                else:
                    pinch_data = pinch_state['right']
                    pinch_data['frames_held'] += 1
                    
                    if pinch_data['frames_held'] > 5:
                        if not state.grabbed:
                            state.grabbed = True
                            state.grab_start_x = pinch_x
                            state.grab_start_y = pinch_y
                            state.grab_start_card_pos = state.card_anim_pos
                            state.grab_start_category_pos = state.category_anim_pos
                            state.velocity = 0
                            state.vertical_velocity = 0
                        else:
                            # Horizontal movement
                            old_pos = state.card_anim_pos
                            delta_x = pinch_x - state.grab_start_x
                            card_movement = -delta_x / (CARD_WIDTH + CARD_SPACING) * 1.5
                            state.card_anim_pos = state.grab_start_card_pos + card_movement
                            state.card_anim_pos = max(0, min(CARD_COUNT - 1, state.card_anim_pos))
                            state.velocity = state.card_anim_pos - old_pos
                            
                            # Vertical movement
                            old_category_pos = state.category_anim_pos
                            delta_y = pinch_y - state.grab_start_y
                            category_movement = -delta_y / ROW_SPACING  # Inverted: down = next category
                            state.category_anim_pos = state.grab_start_category_pos + category_movement
                            state.category_anim_pos = max(0, min(NUM_CATEGORIES - 1, state.category_anim_pos))
                            state.vertical_velocity = state.category_anim_pos - old_category_pos
            else:
                if 'right' in pinch_state:
                    pinch_data = pinch_state['right']
                    if pinch_data['frames_held'] <= 5 and not state.grabbed:
                        state.selected_card = int(round(state.card_anim_pos))
                        category_idx = int(round(state.category_anim_pos))
                        print(f"Selected: {CAROUSEL_CATEGORIES[category_idx][state.selected_card]}")
                    del pinch_state['right']
        
        if not any_pinching and state.grabbed:
            state.grabbed = False
            state.grab_start_x = None
            state.grab_start_y = None
            state.grab_start_card_pos = None
            state.grab_start_category_pos = None
        
        # Apply horizontal inertia
        if not state.grabbed:
            if abs(state.velocity) > MIN_VELOCITY:
                state.card_anim_pos += state.velocity
                state.velocity *= FRICTION
                state.card_anim_pos = max(0, min(CARD_COUNT - 1, state.card_anim_pos))
                if state.card_anim_pos <= 0 or state.card_anim_pos >= CARD_COUNT - 1:
                    state.velocity = 0
            else:
                state.velocity = 0
            
            # Apply vertical inertia
            if abs(state.vertical_velocity) > MIN_VELOCITY:
                state.category_anim_pos += state.vertical_velocity
                state.vertical_velocity *= VERTICAL_FRICTION
                state.category_anim_pos = max(0, min(NUM_CATEGORIES - 1, state.category_anim_pos))
                if state.category_anim_pos <= 0 or state.category_anim_pos >= NUM_CATEGORIES - 1:
                    state.vertical_velocity = 0
            else:
                state.vertical_velocity = 0
        
        # Clear screen with gradient background
        screen.fill((20, 20, 30))
        
        # Draw three rows of cards with smooth vertical scrolling
        center_y = WINDOW_HEIGHT // 2
        
        # Calculate vertical offset based on animation position
        base_category = int(state.category_anim_pos)
        category_fraction = state.category_anim_pos - base_category
        vertical_offset = category_fraction * ROW_SPACING
        
        # Draw all three visible categories
        for i in range(-1, 2):
            cat_idx = (base_category + i) % NUM_CATEGORIES
            y_pos = center_y + (i * ROW_SPACING) - vertical_offset
            draw_cards(screen, WINDOW_WIDTH//2, y_pos, state.card_anim_pos, cat_idx)
        
        # Draw camera feed in corner
        frame_surface = pygame.surfarray.make_surface(cv2.transpose(rgb))
        frame_surface = pygame.transform.scale(frame_surface, (320, 240))
        screen.blit(frame_surface, (WINDOW_WIDTH - 330, 10))
        
        # Draw hand cursors on main screen (only right hand)
        if right_hand:
            thumb_tip = right_hand[4]
            index_tip = right_hand[8]
            
            thumb_x = int(thumb_tip.x * WINDOW_WIDTH)
            thumb_y = int(thumb_tip.y * WINDOW_HEIGHT)
            index_x = int(index_tip.x * WINDOW_WIDTH)
            index_y = int(index_tip.y * WINDOW_HEIGHT)
            
            is_pinch = is_pinching(right_hand, threshold=0.08)
            cursor_color = (255, 0, 0) if is_pinch else (0, 255, 0)
            
            if is_pinch:
                pygame.draw.line(screen, (255, 0, 0), (thumb_x, thumb_y), (index_x, index_y), 3)
            
            pygame.draw.circle(screen, cursor_color, (thumb_x, thumb_y), 15)
            pygame.draw.circle(screen, (255, 255, 255), (thumb_x, thumb_y), 15, 2)
            pygame.draw.circle(screen, cursor_color, (index_x, index_y), 15)
            pygame.draw.circle(screen, (255, 255, 255), (index_x, index_y), 15, 2)
        
        # Draw status
        font = pygame.font.Font(None, 48)
        status = "GRABBED" if state.grabbed else "Ready"
        text = font.render(status, True, (255, 255, 255))
        screen.blit(text, (30, 30))
        
        pygame.display.flip()
        clock.tick(60)
    
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

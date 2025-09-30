import cv2
import mediapipe as mp
import numpy as np
import math

# Constants
CARD_COUNT = 7
CARD_WIDTH = 420
CARD_HEIGHT = 400
CARD_SPACING = 60
CAROUSEL_CATEGORIES = [
    ["Mail", "Music", "Safari", "Messages", "Calendar", "Maps", "Camera"],
    ["Photos", "Notes", "Reminders", "Clock", "Weather", "Stocks", "News"],
    ["YouTube", "Netflix", "Twitch", "Spotify", "Podcasts", "Books", "Games"]
]
NUM_CATEGORIES = len(CAROUSEL_CATEGORIES)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandState:
    def __init__(self):
        self.card_index = 3  # Start at middle card
        self.card_anim_pos = 3.0

# Utility: Check for pinch gesture and return pinch strength
def get_pinch_distance(landmarks):
    if not landmarks:
        return None
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
    return dist

def is_pinching(landmarks, threshold=0.05):
    dist = get_pinch_distance(landmarks)
    return dist is not None and dist < threshold

def get_pinch_position(landmarks):
    """Get the midpoint between thumb and index finger tips"""
    if not landmarks:
        return None
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    return ((thumb_tip.x + index_tip.x) / 2, (thumb_tip.y + index_tip.y) / 2)

def draw_cards(frame, center_x, center_y, anim_pos, category_idx=0):
    app_names = CAROUSEL_CATEGORIES[category_idx]
    for i in range(CARD_COUNT):
        offset = (i - anim_pos) * (CARD_WIDTH + CARD_SPACING)
        x = int(center_x + offset)
        y = int(center_y)
        
        color = (255, 255, 255) if round(anim_pos) == i else (200, 200, 200)
            
        cv2.rectangle(frame, (x - CARD_WIDTH//2, y - CARD_HEIGHT//2),
                      (x + CARD_WIDTH//2, y + CARD_HEIGHT//2), color, -1)
        cv2.rectangle(frame, (x - CARD_WIDTH//2, y - CARD_HEIGHT//2),
                      (x + CARD_WIDTH//2, y + CARD_HEIGHT//2), (0,0,0), 4)
        
        # Center the app name on the card
        if i < len(app_names):
            text = app_names[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x - text_size[0] // 2
            text_y = y + text_size[1] // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0,0,0), thickness)

def main():
    import time
    cap = cv2.VideoCapture(0)
    # Reduce resolution for better FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    hands = mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity=0  # Use lighter model for better performance
    )
    state = HandState()
    category_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        hand_landmarks = [h.landmark for h in results.multi_hand_landmarks] if results.multi_hand_landmarks else []

        # Draw cards
        draw_cards(frame, w//2, h//2, state.card_anim_pos, category_idx)
        
        # Draw cursors for each hand
        for hand in hand_landmarks:
            # Get thumb tip and index finger tip positions
            thumb_tip = hand[4]
            index_tip = hand[8]
            
            thumb_x = int(thumb_tip.x * w)
            thumb_y = int(thumb_tip.y * h)
            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)
            
            # Check if pinching
            is_pinch = is_pinching(hand, threshold=0.05)
            
            # Set cursor color based on pinch state
            cursor_color = (0, 0, 255) if is_pinch else (0, 255, 0)  # Red if pinching, green otherwise
            
            # Draw line connecting thumb and index when close
            if is_pinch:
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255), 3)
            
            # Draw thumb marker (circle)
            cv2.circle(frame, (thumb_x, thumb_y), 15, cursor_color, -1)
            cv2.circle(frame, (thumb_x, thumb_y), 15, (255, 255, 255), 2)
            
            # Draw index finger marker (circle)
            cv2.circle(frame, (index_x, index_y), 15, cursor_color, -1)
            cv2.circle(frame, (index_x, index_y), 15, (255, 255, 255), 2)
        
        # Draw hand landmarks (optional - comment out for even better FPS)
        # if results.multi_hand_landmarks:
        #     for handLms in results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                
        cv2.imshow('Hand Carousel', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
from collections import deque
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import threading
import time
import pyautogui

mp_hands = mp.solutions.hands

def clamp(v, lo, hi): 
    return lo if v < lo else hi if v > hi else v

class BrowserState:
    def __init__(self):
        self.pinch_prev = False
        self.pinch_threshold = 0.08
        self.driver = None
        self.browser_ready = False
        self.last_click_time = 0
        self.double_click_window = 0.4
        self.click_cooldown = 0
        self.three_finger_prev = False
        
        # Smooth cursor tracking
        self.smooth_cursor_x = None
        self.smooth_cursor_y = None
        self.smoothing_factor = 0.15  # Lower = smoother but slower response

def get_pinch_distance(landmarks):
    if not landmarks: return None
    a = landmarks[4]; b = landmarks[8]
    return math.hypot(a.x - b.x, a.y - b.y)

def is_pinching(landmarks, thresh):
    d = get_pinch_distance(landmarks)
    return (d is not None) and (d < thresh)

def is_finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

def detect_three_finger_up(landmarks):
    if not landmarks: return False
    index_ext = is_finger_extended(landmarks, 8, 6)
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_ext = is_finger_extended(landmarks, 16, 14)
    pinky_down = not is_finger_extended(landmarks, 20, 18)
    return index_ext and middle_ext and ring_ext and pinky_down

def start_chrome_browser(state):
    try:
        print("Starting Chrome browser...")
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        print("Installing ChromeDriver...")
        service = Service(ChromeDriverManager().install())
        
        print("Launching Chrome...")
        state.driver = webdriver.Chrome(service=service, options=chrome_options)
        
        print("Loading Google...")
        state.driver.get("https://www.google.com")
        
        time.sleep(2)
        state.browser_ready = True
        print("✓ Chrome ready! Control it with hand gestures")
    except Exception as e:
        print(f"✗ Error starting browser: {e}")
        print("You can still use cursor control without the browser")
        state.browser_ready = False

def main():
    pygame.init()
    WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Gesture Cursor Control")
    clock = pygame.time.Clock()

    print("=" * 60)
    print("GESTURE CURSOR CONTROL")
    print("=" * 60)
    print("Initializing...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=0
    )

    state = BrowserState()
    
    browser_thread = threading.Thread(target=start_chrome_browser, args=(state,), daemon=True)
    browser_thread.start()
    
    print("Controls:")
    print("  Hand = Move cursor")
    print("  Pinch = Click")
    print("  Double-pinch = Double-click")
    print("  3 fingers up = Right-click")
    print("=" * 60)
    
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                if hd.classification[0].label == "Right":
                    right_hand = hl.landmark
                    break

        if right_hand:
            # Always use index finger tip for cursor position (doesn't shift when pinching)
            index_tip = right_hand[8]
            screen_width, screen_height = pyautogui.size()
            
            # Target cursor position based on hand
            target_x = int(index_tip.x * screen_width)
            target_y = int(index_tip.y * screen_height)
            
            # Initialize smooth position on first frame
            if state.smooth_cursor_x is None:
                state.smooth_cursor_x = target_x
                state.smooth_cursor_y = target_y
            
            # Smooth interpolation (exponential moving average)
            state.smooth_cursor_x = state.smooth_cursor_x * (1 - state.smoothing_factor) + target_x * state.smoothing_factor
            state.smooth_cursor_y = state.smooth_cursor_y * (1 - state.smoothing_factor) + target_y * state.smoothing_factor
            
            # Move cursor to smoothed position
            pyautogui.moveTo(int(state.smooth_cursor_x), int(state.smooth_cursor_y), duration=0)
            
            # Three fingers up for right click
            three_finger_now = detect_three_finger_up(right_hand)
            if three_finger_now and not state.three_finger_prev:
                pyautogui.rightClick()
                print("RIGHT CLICK!")
            state.three_finger_prev = three_finger_now
            
            # Pinch detection for clicks (using pinch distance, not position)
            pinch_now = is_pinching(right_hand, state.pinch_threshold)
            
            # Debug: print pinch distance
            pinch_dist = get_pinch_distance(right_hand)
            if pinch_dist is not None and pinch_dist < 0.15:  # Only print when fingers are close
                print(f"Pinch distance: {pinch_dist:.3f} (threshold: {state.pinch_threshold})")
            
            if pinch_now and not state.pinch_prev and state.click_cooldown == 0:
                current_time = time.time()
                time_since_last = current_time - state.last_click_time
                
                if time_since_last < state.double_click_window:
                    pyautogui.doubleClick()
                    print("DOUBLE CLICK!")
                    state.click_cooldown = 20
                else:
                    pyautogui.click()
                    print("CLICK!")
                    state.click_cooldown = 10
                
                state.last_click_time = current_time
            
            state.pinch_prev = pinch_now
        else:
            # Reset smooth position when hand is lost
            state.smooth_cursor_x = None
            state.smooth_cursor_y = None
        
        if state.click_cooldown > 0:
            state.click_cooldown -= 1

        screen.fill((30, 30, 40))

        frame_surface = pygame.surfarray.make_surface(cv2.transpose(rgb))
        frame_surface = pygame.transform.scale(frame_surface, (400, 300))
        screen.blit(frame_surface, (WINDOW_WIDTH - 410, 10))

        if right_hand:
            # Draw circles on thumb and index finger for visual feedback
            tt = right_hand[4]
            it = right_hand[8]
            tx = int(tt.x * WINDOW_WIDTH)
            ty = int(tt.y * WINDOW_HEIGHT)
            ix = int(it.x * WINDOW_WIDTH)
            iy = int(it.y * WINDOW_HEIGHT)
            
            pygame.draw.circle(screen, (255, 255, 255), (tx, ty), 10)
            pygame.draw.circle(screen, (255, 255, 255), (ix, iy), 10)

        font_large = pygame.font.Font(None, 56)
        font_small = pygame.font.Font(None, 36)
        
        if right_hand:
            status = "Cursor Active"
            color = (100, 255, 100)
        else:
            status = "Show hand to control"
            color = (255, 255, 255)
        
        screen.blit(font_large.render(status, True, color), (30, 30))
        
        instructions = [
            "Hand = Move cursor",
            "Pinch = Click",
            "Double-pinch = Double-click",
            "3 fingers up = Right-click",
            f"Browser: {'✓ Ready' if state.browser_ready else '⏳ Loading...'}"
        ]
        
        y_offset = 100
        for instruction in instructions:
            screen.blit(font_small.render(instruction, True, (200, 200, 200)), (30, y_offset))
            y_offset += 40

        pygame.display.flip()
        clock.tick(60)

    if state.driver:
        state.driver.quit()
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

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

class FingerSmoother:
    def __init__(self, window_size=5):
        self.thumb_history = deque(maxlen=window_size)
        self.index_history = deque(maxlen=window_size)
    
    def update(self, thumb_pos, index_pos):
        self.thumb_history.append(thumb_pos)
        self.index_history.append(index_pos)
        tx = sum(p[0] for p in self.thumb_history)/len(self.thumb_history)
        ty = sum(p[1] for p in self.thumb_history)/len(self.thumb_history)
        ix = sum(p[0] for p in self.index_history)/len(self.index_history)
        iy = sum(p[1] for p in self.index_history)/len(self.index_history)
        return (tx, ty), (ix, iy)
    
    def reset(self): 
        self.thumb_history.clear()
        self.index_history.clear()

class BrowserState:
    def __init__(self):
        self.pinch_prev = False
        self.pinch_threshold = 0.08
        self.finger_smoother = FingerSmoother(window_size=8)
        self.driver = None
        self.browser_ready = False
        self.last_click_time = 0
        self.double_click_window = 0.4
        self.click_cooldown = 0
        self.three_finger_prev = False

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
        service = Service(ChromeDriverManager().install())
        state.driver = webdriver.Chrome(service=service, options=chrome_options)
        state.driver.get("https://www.google.com")
        time.sleep(2)
        state.browser_ready = True
        print("✓ Chrome ready! Control it with hand gestures")
    except Exception as e:
        print(f"✗ Error starting browser: {e}")
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

        if right_hand is None:
            state.finger_smoother.reset()

        if right_hand:
            index_tip = right_hand[8]
            screen_width, screen_height = pyautogui.size()
            cursor_x = int(index_tip.x * screen_width)
            cursor_y = int(index_tip.y * screen_height)
            
            current_x, current_y = pyautogui.position()
            smooth_x = int(current_x * 0.7 + cursor_x * 0.3)
            smooth_y = int(current_y * 0.7 + cursor_y * 0.3)
            pyautogui.moveTo(smooth_x, smooth_y, duration=0)
            
            three_finger_now = detect_three_finger_up(right_hand)
            if three_finger_now and not state.three_finger_prev:
                pyautogui.rightClick()
                print("RIGHT CLICK!")
            state.three_finger_prev = three_finger_now
            
            pinch_now = is_pinching(right_hand, state.pinch_threshold)
            
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
        
        if state.click_cooldown > 0:
            state.click_cooldown -= 1

        screen.fill((30, 30, 40))

        frame_surface = pygame.surfarray.make_surface(cv2.transpose(rgb))
        frame_surface = pygame.transform.scale(frame_surface, (400, 300))
        screen.blit(frame_surface, (WINDOW_WIDTH - 410, 10))

        if right_hand:
            tt = right_hand[4]
            it = right_hand[8]
            (tx, ty), (ix, iy) = state.finger_smoother.update(
                (tt.x * WINDOW_WIDTH, tt.y * WINDOW_HEIGHT),
                (it.x * WINDOW_WIDTH, it.y * WINDOW_HEIGHT)
            )
            
            pygame.draw.circle(screen, (255, 255, 255), (int(tx), int(ty)), 10)
            pygame.draw.circle(screen, (255, 255, 255), (int(ix), int(iy)), 10)
        else:
            state.finger_smoother.reset()

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

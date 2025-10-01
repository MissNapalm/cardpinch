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
        self.ok_prev = False
        self.ok_threshold = 0.035
        self.cursor_x = None
        self.cursor_y = None

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

def detect_ok_gesture(landmarks, touch_thresh):
    if not landmarks: return False
    a = landmarks[4]; b = landmarks[8]
    touching = math.hypot(a.x - b.x, a.y - b.y) < touch_thresh
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_ext = is_finger_extended(landmarks, 16, 14)
    pinky_ext = is_finger_extended(landmarks, 20, 18)
    return touching and middle_ext and ring_ext and pinky_ext

def start_chrome_browser(state):
    try:
        print("Starting Chrome...")
        chrome_options = Options()
        chrome_options.add_argument("--start-maximized")
        service = Service(ChromeDriverManager().install())
        state.driver = webdriver.Chrome(service=service, options=chrome_options)
        state.driver.get("https://www.google.com")
        time.sleep(2)
        state.browser_ready = True
        print("✓ Chrome ready!")
    except Exception as e:
        print(f"✗ Browser error: {e}")
        state.browser_ready = False

def main():
    pygame.init()
    WINDOW_WIDTH, WINDOW_HEIGHT = 400, 300
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Gesture Control")
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5, model_complexity=0)
    state = BrowserState()
    threading.Thread(target=start_chrome_browser, args=(state,), daemon=True).start()
    
    print("Pinch=Click | Double-pinch=Double-click | 3-fingers=Right-click | OK=Quit")
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        ret, frame = cap.read()
        if not ret: continue
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
            # OK gesture to quit
            ok_now = detect_ok_gesture(right_hand, state.ok_threshold)
            if ok_now and not state.ok_prev:
                print("OK gesture detected - Closing Chrome and exiting...")
                if state.driver:
                    try:
                        state.driver.quit()
                        print("✓ Chrome closed")
                    except:
                        pass
                running = False
            state.ok_prev = ok_now
            
            # Move cursor to midpoint between thumb and index (the pinch point)
            thumb_tip = right_hand[4]
            index_tip = right_hand[8]
            screen_width, screen_height = pyautogui.size()
            
            # Calculate midpoint
            mid_x = (thumb_tip.x + index_tip.x) / 2
            mid_y = (thumb_tip.y + index_tip.y) / 2
            
            state.cursor_x = int(mid_x * screen_width)
            state.cursor_y = int(mid_y * screen_height)
            
            # Move system cursor to pinch point
            pyautogui.moveTo(state.cursor_x, state.cursor_y, duration=0)
            
            # Three fingers for right click
            three_finger_now = detect_three_finger_up(right_hand)
            if three_finger_now and not state.three_finger_prev:
                pyautogui.rightClick()
                print("RIGHT CLICK!")
            state.three_finger_prev = three_finger_now
            
            # Pinch for click
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
        else:
            state.cursor_x = None
            state.cursor_y = None
        
        if state.click_cooldown > 0:
            state.click_cooldown -= 1

        screen.fill((30, 30, 40))

        # Draw crosshair showing where clicks will happen
        if state.cursor_x is not None:
            screen_info = pygame.display.Info()
            scale_x = WINDOW_WIDTH / screen_info.current_w
            scale_y = WINDOW_HEIGHT / screen_info.current_h
            vis_x = int(state.cursor_x * scale_x)
            vis_y = int(state.cursor_y * scale_y)
            
            color = (0, 255, 0)
            size = 50
            pygame.draw.line(screen, color, (vis_x - size, vis_y), (vis_x - 10, vis_y), 4)
            pygame.draw.line(screen, color, (vis_x + 10, vis_y), (vis_x + size, vis_y), 4)
            pygame.draw.line(screen, color, (vis_x, vis_y - size), (vis_x, vis_y - 10), 4)
            pygame.draw.line(screen, color, (vis_x, vis_y + 10), (vis_x, vis_y + size), 4)
            pygame.draw.circle(screen, color, (vis_x, vis_y), 8)
            pygame.draw.circle(screen, (30, 30, 40), (vis_x, vis_y), 4)
            pygame.draw.circle(screen, color, (vis_x, vis_y), 15, 2)

        font_large = pygame.font.Font(None, 48)
        font_small = pygame.font.Font(None, 28)
        
        if state.cursor_x is not None:
            status = f"({state.cursor_x}, {state.cursor_y})"
            color = (100, 255, 100)
        else:
            status = "Show hand"
            color = (255, 255, 255)
        
        screen.blit(font_large.render(status, True, color), (20, 20))
        
        instructions = ["Pinch=Click", "2x=Double", "3-fingers=Right", "OK=Quit"]
        y_offset = 80
        for instruction in instructions:
            screen.blit(font_small.render(instruction, True, (200, 200, 200)), (20, y_offset))
            y_offset += 35

        pygame.display.flip()
        clock.tick(60)

    if state.driver:
        state.driver.quit()
    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

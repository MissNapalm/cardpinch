import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
from collections import deque

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

class BrowserState:
    def __init__(self):
        self.scroll_y = 0.0
        self.scroll_x = 0.0
        self.smooth_scroll_y = 0.0
        self.smooth_scroll_x = 0.0
        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_threshold = 0.08
        self.finger_smoother = FingerSmoother(window_size=5)
        self.ok_prev = False
        self.ok_touch_threshold = 0.035
        self.page_height = 2700
        self.page_width = 1200
        
        # Zoom controls
        self.zoom_level = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 2.0
        self.zoom_sensitivity = 0.15
        
        # Three-finger wheel gesture
        self.wheel_active = False
        self.wheel_angle = 0.0
        self.last_finger_angle = None
        self.wheel_center_x = 0
        self.wheel_center_y = 0
        self.wheel_radius = 110

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
    if not landmarks:
        return False
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    wrist = landmarks[0]
    thumb_ext = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
    index_ext = is_finger_extended(landmarks, 8, 6)
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_fold = landmarks[16].y > landmarks[14].y - 0.02
    pinky_fold = landmarks[20].y > landmarks[18].y - 0.02
    return thumb_ext and index_ext and middle_ext and ring_fold and pinky_fold

def get_hand_center(landmarks):
    return landmarks[9]

def calculate_finger_angle(landmarks):
    c = get_hand_center(landmarks)
    idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)

def detect_ok_gesture(landmarks, touch_thresh):
    if not landmarks:
        return False
    a = landmarks[4]
    b = landmarks[8]
    touching = math.hypot(a.x - b.x, a.y - b.y) < touch_thresh
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_ext = is_finger_extended(landmarks, 16, 14)
    pinky_ext = is_finger_extended(landmarks, 20, 18)
    return touching and middle_ext and ring_ext and pinky_ext

def draw_wheel(surface, state, window_width, window_height):
    if not state.wheel_active:
        return
    
    cx = state.wheel_center_x
    cy = state.wheel_center_y
    r = state.wheel_radius
    white = (255, 255, 255)
    
    # Outer glow rings
    for i in range(5):
        rr = r + 15 + i * 10
        op = int(100 - i * 20)
        s = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        pygame.draw.circle(s, (*white, op), (cx, cy), rr, 2)
        surface.blit(s, (0, 0))
    
    # Main wheel circles
    pygame.draw.circle(surface, white, (cx, cy), r, 4)
    pygame.draw.circle(surface, white, (cx, cy), r - 20, 2)
    
    # Progress segments
    segs = 48
    prog = int((state.wheel_angle / (2 * math.pi)) * segs) % segs
    ir = r - 10
    for i in range(prog):
        sa = math.radians(i * 360 / segs) - math.pi / 2
        ea = math.radians((i + 1) * 360 / segs) - math.pi / 2
        sx = cx + int(ir * math.cos(sa))
        sy = cy + int(ir * math.sin(sa))
        ex = cx + int(ir * math.cos(ea))
        ey = cy + int(ir * math.sin(ea))
        pygame.draw.line(surface, white, (sx, sy), (ex, ey), 6)
    
    # Pointer line
    pl = r - 30
    px = cx + int(pl * math.cos(state.wheel_angle))
    py = cy + int(pl * math.sin(state.wheel_angle))
    pygame.draw.line(surface, white, (cx, cy), (px, py), 3)
    pygame.draw.circle(surface, white, (px, py), 6)
    pygame.draw.circle(surface, white, (cx, cy), 8)
    
    # Zoom level text
    font = pygame.font.SysFont('arial', 32, bold=True)
    t = font.render(f"Zoom {state.zoom_level:.2f}x", True, white)
    tr = t.get_rect(center=(cx, cy + r + 44))
    bg = pygame.Rect(tr.x - 10, tr.y - 5, tr.width + 20, tr.height + 10)
    pygame.draw.rect(surface, (20, 20, 20), bg)
    pygame.draw.rect(surface, white, bg, 2)
    surface.blit(t, tr)

def draw_browser_chrome(surface, width, height):
    # Main toolbar background
    pygame.draw.rect(surface, (255, 255, 255), (0, 0, width, 75))
    
    # Tabs background area
    pygame.draw.rect(surface, (235, 235, 237), (0, 0, width, 36))
    
    # Active tab
    tab_rect = pygame.Rect(8, 6, 220, 30)
    pygame.draw.rect(surface, (255, 255, 255), tab_rect, border_radius=8)
    
    # Favicon
    pygame.draw.circle(surface, (66, 133, 244), (22, 21), 7)
    
    font_tab = pygame.font.SysFont('arial', 18)
    tab_text = font_tab.render("TechFlow - Latest News", True, (32, 33, 36))
    surface.blit(tab_text, (35, 15))
    
    # Tab close button
    pygame.draw.line(surface, (95, 99, 104), (208, 16), (218, 26), 2)
    pygame.draw.line(surface, (95, 99, 104), (218, 16), (208, 26), 2)
    
    # Inactive tab
    inactive_tab = pygame.Rect(232, 6, 180, 30)
    pygame.draw.rect(surface, (218, 220, 224), inactive_tab, border_radius=8)
    pygame.draw.circle(surface, (189, 193, 198), (246, 21), 6)
    inactive_text = font_tab.render("New Tab", True, (95, 99, 104))
    surface.blit(inactive_text, (258, 15))
    
    # New tab button
    pygame.draw.rect(surface, (218, 220, 224), (418, 10, 26, 20), border_radius=4)
    pygame.draw.line(surface, (95, 99, 104), (425, 20), (437, 20), 2)
    pygame.draw.line(surface, (95, 99, 104), (431, 14), (431, 26), 2)
    
    # Navigation buttons
    pygame.draw.polygon(surface, (95, 99, 104), [(25, 52), (32, 46), (32, 58)])
    pygame.draw.polygon(surface, (189, 193, 198), [(52, 46), (52, 58), (45, 52)])
    pygame.draw.circle(surface, (95, 99, 104), (73, 52), 8, 2)
    pygame.draw.polygon(surface, (95, 99, 104), [(77, 48), (82, 48), (77, 43)])
    
    # Home button
    pygame.draw.rect(surface, (95, 99, 104), (93, 49, 10, 8))
    pygame.draw.polygon(surface, (95, 99, 104), [(98, 47), (91, 52), (105, 52)])
    
    # Address bar
    addr_rect = pygame.Rect(115, 45, width - 275, 22)
    pygame.draw.rect(surface, (241, 243, 244), addr_rect, border_radius=11)
    
    # Lock icon
    pygame.draw.rect(surface, (32, 33, 36), (125, 52, 6, 8))
    pygame.draw.arc(surface, (32, 33, 36), (123, 49, 10, 8), 0, 3.14159, 2)
    
    font_url = pygame.font.SysFont('arial', 18)
    url_text = font_url.render("techflow.io/news/innovation", True, (32, 33, 36))
    surface.blit(url_text, (138, 50))
    
    # Star bookmark icon
    star_x = width - 230
    pygame.draw.polygon(surface, (95, 99, 104), [
        (star_x, 47), (star_x+2, 53), (star_x+8, 53), 
        (star_x+3, 57), (star_x+5, 63), (star_x, 59),
        (star_x-5, 63), (star_x-3, 57), (star_x-8, 53), (star_x-2, 53)
    ], 2)
    
    # Extensions icons
    ext_x = width - 180
    for i in range(3):
        x = ext_x + i * 30
        pygame.draw.rect(surface, (189, 193, 198), (x, 48, 18, 18), border_radius=4)
        pygame.draw.circle(surface, (66, 133, 244), (x+9, 57), 4)
    
    # Profile button
    pygame.draw.circle(surface, (234, 67, 53), (width - 50, 52), 14)
    profile_font = pygame.font.SysFont('arial', 16, bold=True)
    profile_text = profile_font.render("U", True, (255, 255, 255))
    surface.blit(profile_text, (width - 55, 46))
    
    # Menu button
    menu_x = width - 25
    for i in range(3):
        pygame.draw.circle(surface, (95, 99, 104), (menu_x, 48 + i*5), 2)
    
    # Bottom shadow
    for i in range(5):
        alpha = 20 - i * 4
        color = (0, 0, 0) if alpha > 0 else (0, 0, 0)
        pygame.draw.line(surface, color, (0, 75+i), (width, 75+i))

def draw_fake_webpage(surface, width, height, scroll_x, scroll_y):
    content_y_offset = 80
    
    # Background
    pygame.draw.rect(surface, (255, 255, 255), (0, content_y_offset, width, height - content_y_offset))
    
    # Top navigation bar
    nav_y = content_y_offset + int(scroll_y)
    if nav_y > content_y_offset - 60 and nav_y < height:
        pygame.draw.rect(surface, (255, 255, 255), (0, max(nav_y, content_y_offset), width, 60))
        pygame.draw.line(surface, (229, 231, 235), (0, max(nav_y + 59, content_y_offset)), (width, max(nav_y + 59, content_y_offset)), 1)
        
        # Logo
        logo_y = max(nav_y + 20, content_y_offset + 5)
        if logo_y < height - 20:
            logo_font = pygame.font.SysFont('arial', 32, bold=True)
            logo_text = logo_font.render("TechFlow", True, (59, 130, 246))
            surface.blit(logo_text, (40, logo_y))
        
        # Nav links
        nav_font = pygame.font.SysFont('arial', 20)
        nav_items = ["News", "Reviews", "Videos", "Podcasts", "Events"]
        nav_x = 200
        for item in nav_items:
            text = nav_font.render(item, True, (75, 85, 99))
            if logo_y < height - 20:
                surface.blit(text, (nav_x, logo_y + 8))
            nav_x += 100
        
        # Search box
        search_rect = pygame.Rect(width - 240, logo_y, 180, 32)
        if logo_y < height - 32:
            pygame.draw.rect(surface, (249, 250, 251), search_rect, border_radius=16)
            pygame.draw.rect(surface, (209, 213, 219), search_rect, 1, border_radius=16)
            search_text = nav_font.render("Search...", True, (156, 163, 175))
            surface.blit(search_text, (search_rect.x + 12, search_rect.y + 8))
    
    # Hero section
    hero_y = content_y_offset + int(scroll_y + 60)
    if hero_y > content_y_offset - 500 and hero_y < height:
        hero_height = 480
        for i in range(hero_height):
            progress = i / hero_height
            r = int(30 + (59 - 30) * progress)
            g = int(58 + (130 - 58) * progress)
            b = int(138 + (246 - 138) * progress)
            y_pos = hero_y + i
            if content_y_offset <= y_pos < height:
                pygame.draw.line(surface, (r, g, b), (0, y_pos), (width, y_pos))
        
        overlay_y = hero_y + 140
        if content_y_offset <= overlay_y < height:
            hero_title_font = pygame.font.SysFont('arial', 56, bold=True)
            hero_title = hero_title_font.render("The Future of AI", True, (255, 255, 255))
            surface.blit(hero_title, (60, overlay_y))
            
            hero_title2 = hero_title_font.render("Is Here", True, (255, 255, 255))
            surface.blit(hero_title2, (60, overlay_y + 65))
            
            subtitle_font = pygame.font.SysFont('arial', 24)
            subtitle = subtitle_font.render("Breakthrough innovations reshaping technology", True, (226, 232, 240))
            surface.blit(subtitle, (60, overlay_y + 140))
            
            # CTA button
            cta_rect = pygame.Rect(60, overlay_y + 195, 180, 48)
            pygame.draw.rect(surface, (255, 255, 255), cta_rect, border_radius=24)
            cta_font = pygame.font.SysFont('arial', 22, bold=True)
            cta_text = cta_font.render("Read More →", True, (59, 130, 246))
            surface.blit(cta_text, (cta_rect.x + 25, cta_rect.y + 15))
    
    # Trending section
    trending_y = content_y_offset + int(scroll_y + 580)
    if trending_y > content_y_offset - 100 and trending_y < height:
        section_font = pygame.font.SysFont('arial', 28, bold=True)
        section_title = section_font.render("TRENDING NOW", True, (15, 23, 42))
        surface.blit(section_title, (60, trending_y))
        pygame.draw.rect(surface, (239, 68, 68), (60, trending_y + 40, 80, 3))
    
    # Article cards
    articles = [
        ("Quantum Computing Reaches New Milestone", "Technology", 640, (239, 68, 68)),
        ("AI Models Learn to Collaborate", "Artificial Intelligence", 640, (16, 185, 129)),
        ("Sustainable Tech: Solar Innovation", "Green Energy", 1020, (234, 179, 8)),
        ("Robotics in Healthcare Breakthrough", "Medicine", 1020, (147, 51, 234)),
        ("5G Networks Transform Cities", "Infrastructure", 1400, (59, 130, 246)),
        ("Cybersecurity Trends 2025", "Security", 1400, (236, 72, 153)),
    ]
    
    card_width = 360
    card_height = 320
    cards_per_row = 3
    
    for idx, (title, category, base_y, color) in enumerate(articles):
        row = idx // cards_per_row
        col = idx % cards_per_row
        
        card_x = 60 + col * (card_width + 30)
        card_y = content_y_offset + int(scroll_y + base_y + row * 360)
        
        if card_y > content_y_offset - card_height and card_y < height:
            # Card container
            card_rect = pygame.Rect(card_x, card_y, card_width, card_height)
            pygame.draw.rect(surface, (255, 255, 255), card_rect, border_radius=12)
            pygame.draw.rect(surface, (229, 231, 235), card_rect, 2, border_radius=12)
            
            # Image placeholder
            img_height = 200
            for i in range(img_height):
                progress = i / img_height
                r = int(color[0] * (1 - progress * 0.3))
                g = int(color[1] * (1 - progress * 0.3))
                b = int(color[2] * (1 - progress * 0.3))
                img_y = card_y + i
                if content_y_offset <= img_y < height:
                    pygame.draw.line(surface, (r, g, b), (card_x, img_y), (card_x + card_width, img_y))
            
            # Category badge
            badge_y = card_y + 15
            if content_y_offset <= badge_y < height - 25:
                badge_font = pygame.font.SysFont('arial', 14, bold=True)
                badge_text = badge_font.render(category.upper(), True, (255, 255, 255))
                badge_rect = pygame.Rect(card_x + 15, badge_y, badge_text.get_width() + 20, 25)
                pygame.draw.rect(surface, color, badge_rect, border_radius=12)
                surface.blit(badge_text, (badge_rect.x + 10, badge_rect.y + 7))
            
            # Article title
            title_y = card_y + 215
            if content_y_offset <= title_y < height - 60:
                title_font = pygame.font.SysFont('arial', 20, bold=True)
                words = title.split()
                line1 = " ".join(words[:4])
                line2 = " ".join(words[4:]) if len(words) > 4 else ""
                
                title_text1 = title_font.render(line1, True, (15, 23, 42))
                surface.blit(title_text1, (card_x + 15, title_y))
                if line2:
                    title_text2 = title_font.render(line2, True, (15, 23, 42))
                    surface.blit(title_text2, (card_x + 15, title_y + 25))
                
                # Meta info
                meta_y = title_y + 60
                meta_font = pygame.font.SysFont('arial', 16)
                meta_text = meta_font.render("5 min read • 2 hours ago", True, (107, 114, 128))
                surface.blit(meta_text, (card_x + 15, meta_y))
    
    # Newsletter section
    newsletter_y = content_y_offset + int(scroll_y + 2200)
    if newsletter_y > content_y_offset - 200 and newsletter_y < height:
        pygame.draw.rect(surface, (30, 58, 138), (0, newsletter_y, width, 200))
        
        news_font = pygame.font.SysFont('arial', 40, bold=True)
        news_title = news_font.render("Stay Updated", True, (255, 255, 255))
        surface.blit(news_title, (width // 2 - news_title.get_width() // 2, newsletter_y + 40))
        
        news_sub = pygame.font.SysFont('arial', 20)
        news_subtitle = news_sub.render("Get the latest tech news delivered to your inbox", True, (191, 219, 254))
        surface.blit(news_subtitle, (width // 2 - news_subtitle.get_width() // 2, newsletter_y + 95))
        
        # Email input
        input_rect = pygame.Rect(width // 2 - 200, newsletter_y + 135, 280, 42)
        pygame.draw.rect(surface, (255, 255, 255), input_rect, border_radius=21)
        input_font = pygame.font.SysFont('arial', 18)
        input_text = input_font.render("Enter your email", True, (156, 163, 175))
        surface.blit(input_text, (input_rect.x + 18, input_rect.y + 12))
        
        # Subscribe button
        btn_rect = pygame.Rect(width // 2 + 90, newsletter_y + 135, 140, 42)
        pygame.draw.rect(surface, (239, 68, 68), btn_rect, border_radius=21)
        btn_text = input_font.render("Subscribe", True, (255, 255, 255))
        surface.blit(btn_text, (btn_rect.x + 28, btn_rect.y + 12))
    
    # Footer
    footer_y = content_y_offset + int(scroll_y + 2420)
    if footer_y > content_y_offset - 250 and footer_y < height:
        pygame.draw.rect(surface, (17, 24, 39), (0, footer_y, width, 250))
        
        footer_font = pygame.font.SysFont('arial', 22, bold=True)
        footer_small = pygame.font.SysFont('arial', 18)
        
        columns = [
            ("Product", ["Features", "Pricing", "News", "Support"]),
            ("Company", ["About", "Blog", "Jobs", "Press"]),
            ("Resources", ["Docs", "Guide", "API", "Community"]),
            ("Legal", ["Privacy", "Terms", "Cookies", "Licenses"])
        ]
        
        col_x = 80
        for col_title, items in columns:
            col_title_text = footer_font.render(col_title, True, (255, 255, 255))
            surface.blit(col_title_text, (col_x, footer_y + 40))
            
            item_y = footer_y + 75
            for item in items:
                item_text = footer_small.render(item, True, (156, 163, 175))
                surface.blit(item_text, (col_x, item_y))
                item_y += 30
            
            col_x += 280
        
        # Copyright
        copyright_font = pygame.font.SysFont('arial', 16)
        copyright_text = copyright_font.render("© 2025 TechFlow. All rights reserved.", True, (107, 114, 128))
        surface.blit(copyright_text, (width // 2 - copyright_text.get_width() // 2, footer_y + 210))
    
    # Scrollbar
    scrollbar_height = max(40, int((height - content_y_offset) * (height - content_y_offset) / 2700))
    scrollbar_y = content_y_offset + int(-scroll_y * (height - content_y_offset - scrollbar_height) / (2700 - height + content_y_offset))
    scrollbar_y = clamp(scrollbar_y, content_y_offset, height - scrollbar_height)
    pygame.draw.rect(surface, (203, 213, 225), (width - 8, scrollbar_y, 6, scrollbar_height), border_radius=3)

def main():
    pygame.init()
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("TechFlow - Latest Tech News")
    clock = pygame.time.Clock()
    
    print("=" * 50)
    print("FAKE BROWSER STARTED")
    print("=" * 50)
    print("Controls:")
    print("  • Pinch and drag to scroll the page")
    print("  • Three-finger gesture (thumb+index+middle) to zoom")
    print("  • Rotate your index finger to adjust zoom level")
    print("  • A-OK gesture to quit")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, 
                           min_tracking_confidence=0.5, model_complexity=0)
    
    state = BrowserState()
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

        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                if hd.classification[0].label == "Right":
                    right_hand = hl.landmark

        # Three-finger wheel gesture for zoom control
        if right_hand:
            if detect_three_finger_gesture(right_hand):
                if not state.wheel_active:
                    hc = get_hand_center(right_hand)
                    state.wheel_active = True
                    state.wheel_center_x = int(hc.x * WINDOW_WIDTH)
                    state.wheel_center_y = int(hc.y * WINDOW_HEIGHT)
                    state.last_finger_angle = None
                    print("Wheel gesture activated - rotate to zoom")
                
                ang = calculate_finger_angle(right_hand)
                if state.last_finger_angle is not None:
                    diff = ang - state.last_finger_angle
                    if diff > math.pi:
                        diff -= 2 * math.pi
                    elif diff < -math.pi:
                        diff += 2 * math.pi
                    
                    state.wheel_angle = (state.wheel_angle + diff * 2) % (2 * math.pi)
                    state.zoom_level = clamp(
                        state.zoom_level + diff * state.zoom_sensitivity, 
                        state.zoom_min, 
                        state.zoom_max
                    )
                
                state.last_finger_angle = ang
            else:
                if state.wheel_active:
                    print(f"Wheel gesture ended - zoom set to {state.zoom_level:.2f}x")
                state.wheel_active = False
                state.last_finger_angle = None
        else:
            state.wheel_active = False
            state.last_finger_angle = None

        # Check for OK gesture to quit
        ok_now = detect_ok_gesture(right_hand, state.ok_touch_threshold) if right_hand else False
        if ok_now and not state.ok_prev:
            print("Closing...")
            running = False
        state.ok_prev = ok_now

        # Handle pinch scrolling (only when wheel is not active)
        if right_hand and not state.wheel_active:
            pinch_now = is_pinching(right_hand, state.pinch_threshold)
            pos = get_pinch_position(right_hand)
            
            if pinch_now and pos:
                px = pos[0] * WINDOW_WIDTH
                py = pos[1] * WINDOW_HEIGHT
                
                if state.is_pinching and state.last_pinch_x is not None:
                    dx = px - state.last_pinch_x
                    dy = py - state.last_pinch_y
                    
                    state.scroll_y += dy
                    state.scroll_x += dx
                    
                    max_scroll_y = state.page_height - (WINDOW_HEIGHT - 80)
                    state.scroll_y = clamp(state.scroll_y, -max_scroll_y, 0)
                    state.scroll_x = clamp(state.scroll_x, 0, 0)
                
                state.last_pinch_x = px
                state.last_pinch_y = py
                state.is_pinching = True
            else:
                state.is_pinching = False
                state.last_pinch_x = None
                state.last_pinch_y = None
        else:
            state.is_pinching = False
            state.last_pinch_x = None
            state.last_pinch_y = None
            state.finger_smoother.reset()

        # Smooth scrolling
        state.smooth_scroll_y += (state.scroll_y - state.smooth_scroll_y) * 0.3
        state.smooth_scroll_x += (state.scroll_x - state.smooth_scroll_x) * 0.3

        # Draw everything
        screen.fill((255, 255, 255))
        
        # Create a surface for the zoomed content
        if state.zoom_level != 1.0:
            zoomed_width = int(WINDOW_WIDTH * state.zoom_level)
            zoomed_height = int(WINDOW_HEIGHT * state.zoom_level)
            zoom_surface = pygame.Surface((zoomed_width, zoomed_height))
            
            draw_fake_webpage(zoom_surface, zoomed_width, zoomed_height, 
                             state.smooth_scroll_x * state.zoom_level, 
                             state.smooth_scroll_y * state.zoom_level)
            
            draw_browser_chrome(zoom_surface, zoomed_width, zoomed_height)
            
            scaled_surface = pygame.transform.smoothscale(zoom_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
            screen.blit(scaled_surface, (0, 0))
        else:
            draw_fake_webpage(screen, WINDOW_WIDTH, WINDOW_HEIGHT, 
                             state.smooth_scroll_x, state.smooth_scroll_y)
            draw_browser_chrome(screen, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Draw wheel visualization
        draw_wheel(screen, state, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        # Draw hand tracking visualization
        if right_hand:
            tt = right_hand[4]
            it = right_hand[8]
            (tx, ty), (ix, iy) = state.finger_smoother.update(
                (tt.x * WINDOW_WIDTH, tt.y * WINDOW_HEIGHT),
                (it.x * WINDOW_WIDTH, it.y * WINDOW_HEIGHT)
            )
            
            if state.is_pinching:
                pygame.draw.line(screen, (255, 100, 100), (int(tx), int(ty)), (int(ix), int(iy)), 3)
                pygame.draw.circle(screen, (255, 100, 100), (int(tx), int(ty)), 10, 2)
                pygame.draw.circle(screen, (255, 100, 100), (int(ix), int(iy)), 10, 2)
            else:
                pygame.draw.circle(screen, (200, 200, 200), (int(tx), int(ty)), 8, 2)
                pygame.draw.circle(screen, (200, 200, 200), (int(ix), int(iy)), 8, 2)
        
        # Status text
        font = pygame.font.SysFont('arial', 28, bold=True)
        if state.wheel_active:
            status = f"ZOOM: {state.zoom_level:.2f}x"
        elif state.is_pinching:
            status = "SCROLLING"
        else:
            status = "Pinch to scroll • 3-finger to zoom"
        status_bg = pygame.Rect(10, WINDOW_HEIGHT - 50, 400, 40)
        pygame.draw.rect(screen, (0, 0, 0, 180), status_bg, border_radius=8)
        status_text = font.render(status, True, (255, 255, 255))
        screen.blit(status_text, (20, WINDOW_HEIGHT - 45))
        
        # Scroll position indicator
        scroll_pct = int((-state.scroll_y / (state.page_height - WINDOW_HEIGHT + 80)) * 100)
        scroll_pct = clamp(scroll_pct, 0, 100)
        scroll_info = f"Scroll: {scroll_pct}%"
        info_text = font.render(scroll_info, True, (100, 100, 100))
        screen.blit(info_text, (WINDOW_WIDTH - 180, WINDOW_HEIGHT - 45))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

# tech_blog_browser.py
# Gestures:
#  - Pinch + drag to scroll
#  - Three-finger (thumb+index+middle) rotate index to zoom (0.5x–2.0x)
#  - A-OK to quit
#
# Render: realistic tech blog layout with hero, latest posts, featured, and a sidebar.
# Local-only rendering (no network).

import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
from collections import deque

mp_hands = mp.solutions.hands

# ------------------------- utils -------------------------
def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

def lerp(a, b, t): 
    return a + (b - a) * t

def in_view(y, h, top, bottom):
    return (y + h) > top and y < bottom

def push_clip(surface, rect):
    """Save current clip and set a new one; returns previous clip to restore later."""
    prev = surface.get_clip()
    surface.set_clip(rect)
    return prev

def pop_clip(surface, prev_clip):
    surface.set_clip(prev_clip)

# word-wrapping with optional max_height and ellipsis
def draw_text_wrapped_clipped(surface, text, rect, font, color, line_spacing=6, antialias=True, max_lines=None):
    """
    Draw wrapped text inside rect, clipping if needed.
    If the text exceeds rect.height, truncate with an ellipsis.
    Returns: height used (not exceeding rect.height).
    """
    words = text.split()
    if not words:
        return 0

    # Build lines that fit width
    lines = []
    cur = []
    w_limit = rect.width

    for w in words:
        candidate = (" ".join(cur + [w])).strip()
        cw, _ = font.size(candidate)
        if cw <= w_limit or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))

    # Clip to rect while drawing
    prev_clip = push_clip(surface, rect)
    y = rect.y
    used_lines = 0
    height_used = 0

    def can_fit_one_more(line):
        return y + font.size(line)[1] <= rect.bottom

    # If max_lines enforced, use it
    line_cap = max_lines if max_lines is not None else len(lines)

    for i, line in enumerate(lines[:line_cap]):
        img = font.render(line, antialias, color)
        line_h = img.get_height()
        if y + line_h > rect.bottom:
            # no space left—try adding ellipsis to previous line if possible
            if used_lines > 0:
                # Replace last drawn line with its ellipsis version if it fits better
                # (Simple approach: overdraw ellipsis at end of rect)
                ell_text = lines[used_lines - 1] + "…"
                # erase last line area by drawing white rect (assumes white bg)
                erase_h = line_h
                pygame.draw.rect(surface, (255,255,255), 
                                 (rect.x, y - (line_h + line_spacing), rect.width, line_h + line_spacing))
                # re-render wrapped ellipsis line (ensure it fits width)
                ell = ell_text
                while font.size(ell)[0] > w_limit and len(ell) > 1:
                    ell = ell[:-2] + "…"
                img2 = font.render(ell, antialias, color)
                surface.blit(img2, (rect.x, y - (line_h + line_spacing)))
                height_used = (y - rect.y - line_spacing) + img2.get_height()
            break

        # draw this line
        surface.blit(img, (rect.x, y))
        y += line_h + line_spacing
        used_lines += 1
        height_used = y - rect.y

    pop_clip(surface, prev_clip)
    # clamp height_used to rect height
    return min(height_used, rect.height)

def draw_pill(surface, text, x, y, font, bg, fg, pad_x=10, pad_y=6, radius=12):
    timg = font.render(text, True, fg)
    rect = pygame.Rect(x, y, timg.get_width() + pad_x*2, timg.get_height() + pad_y*2)
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    surface.blit(timg, (rect.x + pad_x, rect.y + pad_y))
    return rect

# ------------------------- smoothing -------------------------
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
        self.thumb_history.clear(); self.index_history.clear()

# ------------------------- state -------------------------
class BrowserState:
    def __init__(self):
        # scroll (content_y_offset=80, page moves with scroll_y in [-max,0])
        self.scroll_y = 0.0
        self.smooth_scroll_y = 0.0
        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_threshold = 0.08

        # text + hand smoothing
        self.finger_smoother = FingerSmoother(window_size=5)

        # A-OK to quit
        self.ok_prev = False
        self.ok_touch_threshold = 0.035

        # page metrics (updated per frame by layout function)
        self.page_height = 3400

        # Zoom controls (three-finger rotate)
        self.zoom_level = 1.0
        self.zoom_min = 0.5
        self.zoom_max = 2.0
        self.zoom_sensitivity = 0.15

        self.wheel_active = False
        self.wheel_angle = 0.0
        self.last_finger_angle = None
        self.wheel_center_x = 0
        self.wheel_center_y = 0
        self.wheel_radius = 110

# ------------------------- gestures -------------------------
def get_pinch_distance(landmarks):
    if not landmarks: return None
    a = landmarks[4]; b = landmarks[8]
    return math.hypot(a.x - b.x, a.y - b.y)

def is_pinching(landmarks, thresh):
    d = get_pinch_distance(landmarks)
    return (d is not None) and (d < thresh)

def get_pinch_position(landmarks):
    if not landmarks: return None
    a = landmarks[4]; b = landmarks[8]
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)

def is_finger_extended(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

def detect_three_finger_gesture(landmarks):
    if not landmarks: return False
    thumb_tip = landmarks[4]; thumb_mcp = landmarks[2]; wrist = landmarks[0]
    thumb_ext = abs(thumb_tip.x - wrist.x) > abs(thumb_mcp.x - wrist.x) * 0.8
    index_ext = is_finger_extended(landmarks, 8, 6)
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_fold = landmarks[16].y > landmarks[14].y - 0.02
    pinky_fold = landmarks[20].y > landmarks[18].y - 0.02
    return thumb_ext and index_ext and middle_ext and ring_fold and pinky_fold

def get_hand_center(landmarks): 
    return landmarks[9]

def calculate_finger_angle(landmarks):
    c = get_hand_center(landmarks); idx = landmarks[8]
    return math.atan2(idx.y - c.y, idx.x - c.x)

def detect_ok_gesture(landmarks, touch_thresh):
    if not landmarks: return False
    a = landmarks[4]; b = landmarks[8]
    touching = math.hypot(a.x - b.x, a.y - b.y) < touch_thresh
    middle_ext = is_finger_extended(landmarks, 12, 10)
    ring_ext   = is_finger_extended(landmarks, 16, 14)
    pinky_ext  = is_finger_extended(landmarks, 20, 18)
    return touching and middle_ext and ring_ext and pinky_ext

# ------------------------- chrome -------------------------
def draw_browser_chrome(surface, width, height):
    # top toolbar
    pygame.draw.rect(surface, (255,255,255), (0,0,width,72))
    # subtle shadow
    for i in range(6):
        alpha = 90 - i*15
        s = pygame.Surface((width,1), pygame.SRCALPHA)
        s.fill((0,0,0,alpha))
        surface.blit(s,(0,72+i))

    # tabs area
    pygame.draw.rect(surface, (240,242,245), (0,0,width,34))
    active_tab = pygame.Rect(12,6,240,26)
    pygame.draw.rect(surface, (255,255,255), active_tab, border_radius=8)
    icon_col = (66,133,244)
    pygame.draw.circle(surface, icon_col, (active_tab.x+14, active_tab.y+13), 6)
    font_tab = pygame.font.SysFont('arial', 16, bold=True)
    surface.blit(font_tab.render("TechFlow – Latest", True, (32,33,36)),
                 (active_tab.x+28, active_tab.y+6))

    # address bar
    addr_rect = pygame.Rect(120, 40, width-240, 24)
    pygame.draw.rect(surface, (245,247,250), addr_rect, border_radius=12)
    pygame.draw.rect(surface, (226,232,240), addr_rect, 1, border_radius=12)
    font_url = pygame.font.SysFont('consolas', 16)
    surface.blit(font_url.render("https://techflow.example/blog", True, (60,64,67)),
                 (addr_rect.x+12, addr_rect.y+4))

    # right controls (profile dot + menu)
    pygame.draw.circle(surface, (234,67,53), (width-56, 52), 12)
    for i in range(3):
        pygame.draw.circle(surface, (95,99,104), (width-24, 48+i*6), 2)

# ------------------------- page -------------------------
ARTICLES = [
    {
        "title": "Inside Small Language Models: Why Tiny Can Be Mighty",
        "author": "Mina Patel",
        "date": "Oct 1, 2025",
        "tag": "AI",
        "excerpt": "Distilled architectures and retrieval-aware adapters are letting sub-1B parameter models punch far above their weight. Here’s how teams are shipping fast inference without giving up quality."
    },
    {
        "title": "RISC-V on the Edge: Real Workloads, Real Numbers",
        "author": "Alex Garner",
        "date": "Sep 28, 2025",
        "tag": "Chips",
        "excerpt": "Benchmarks can mislead. We profile camera, speech, and tinyML pipelines on commodity RISC-V boards, then contrast them against ARM in identical power envelopes."
    },
    {
        "title": "Privacy by Construction: Telemetry That Doesn’t Phone Home",
        "author": "Jules Ortega",
        "date": "Sep 22, 2025",
        "tag": "Privacy",
        "excerpt": "Event logs are invaluable—until they aren’t. We show a pattern for on-device aggregation with noise injection and delayed release, keeping dashboards useful and users untracked."
    },
    {
        "title": "WebGPU in Production: What Broke and What Stuck",
        "author": "Sofie N.",
        "date": "Sep 18, 2025",
        "tag": "Frontend",
        "excerpt": "Shader portability is better than expected, debugging is worse, and the DX12/Vulkan drivers are the real story. Our migration notes from Canvas2D to compute shaders."
    },
    {
        "title": "Postgres as a Vector DB Without the Pain",
        "author": "Dee Park",
        "date": "Sep 12, 2025",
        "tag": "Data",
        "excerpt": "With pgvector, HNSW indexes, and logical partitioning, we sustain 50K QPS of hybrid search on a single cluster—no separate datastore, no mystery billing line items."
    },
]

POPULAR = [
    "The Forgotten Costs of Cold Starts",
    "Ten Debugging Habits of Senior Engineers",
    "A Practical Guide to GPU Memory Fragmentation",
    "What We Learned Running 500 Canary Releases a Day",
    "Queues, Backpressure, and Sane Retries"
]

TAGS = ["AI","ML","Chips","Privacy","Frontend","Data","DevOps","Security","Edge","Open Source"]

def draw_hero(surface, x, y, w, h):
    # gradient hero
    prev = push_clip(surface, pygame.Rect(x, y, w, h))
    for i in range(h):
        t = i / max(1,h-1)
        r = int(lerp(26, 59, t))
        g = int(lerp(47,130, t))
        b = int(lerp(96,246, t))
        pygame.draw.line(surface, (r,g,b), (x, y+i), (x+w, y+i))
    # overlay text (clipped)
    title_font = pygame.font.SysFont('arial', 54, bold=True)
    sub_font   = pygame.font.SysFont('arial', 22)
    surface.blit(title_font.render("The Future of Building", True, (255,255,255)), (x+36, y+120))
    surface.blit(title_font.render("Is Smaller, Faster, Local", True, (255,255,255)), (x+36, y+180))
    surface.blit(sub_font.render("Opinion • Tooling • Deep Dives", True, (230,240,255)), (x+36, y+240))
    # call to action
    btn = pygame.Rect(x+36, y+290, 200, 44)
    pygame.draw.rect(surface, (255,255,255), btn, border_radius=22)
    cta = pygame.font.SysFont('arial', 20, bold=True).render("Start Reading →", True, (40,90,200))
    surface.blit(cta, (btn.x+24, btn.y+10))
    pop_clip(surface, prev)

def draw_article_card(surface, rect, title, author, date, tag, excerpt):
    # card bg
    pygame.draw.rect(surface, (255,255,255), rect, border_radius=12)
    pygame.draw.rect(surface, (226,232,240), rect, 1, border_radius=12)

    # image placeholder (top)
    img_h = 140
    prev = push_clip(surface, rect)
    for i in range(img_h):
        t = i / max(1,img_h-1)
        r = int(lerp(240, 200, t))
        g = int(lerp(248, 210, t))
        b = int(lerp(255, 240, t))
        pygame.draw.line(surface, (r,g,b), (rect.x, rect.y+i), (rect.x+rect.w, rect.y+i))
    # tag pill (clip-safe)
    tag_font = pygame.font.SysFont('arial', 14, bold=True)
    draw_pill(surface, tag.upper(), rect.x+12, rect.y+12, tag_font, (59,130,246), (255,255,255))
    pop_clip(surface, prev)

    # inner text area (ensure we never overflow)
    pad = 14
    y = rect.y + img_h + 12
    inner = pygame.Rect(rect.x+pad, y, rect.w-2*pad, rect.h - img_h - 2*pad)

    title_font = pygame.font.SysFont('arial', 22, bold=True)
    meta_font  = pygame.font.SysFont('arial', 14)
    body_font  = pygame.font.SysFont('arial', 16)

    # title (max ~2 lines)
    title_h = draw_text_wrapped_clipped(surface, title, 
                                        pygame.Rect(inner.x, inner.y, inner.w, 56),
                                        title_font, (15,23,42), line_spacing=4, max_lines=2)
    cur_y = inner.y + title_h + 6

    # meta row (reserve 24px)
    meta = f"{author}  •  {date}"
    meta_img = meta_font.render(meta, True, (100,116,139))
    if cur_y + meta_img.get_height() <= inner.bottom:
        surface.blit(meta_img, (inner.x, cur_y))
    cur_y += 24

    # excerpt uses whatever vertical space remains
    remain_h = max(0, inner.bottom - cur_y)
    if remain_h > 0:
        draw_text_wrapped_clipped(surface, excerpt, 
                                  pygame.Rect(inner.x, cur_y, inner.w, remain_h),
                                  body_font, (55,65,81))

def draw_sidebar_card(surface, rect, title, items, kind="list"):
    pygame.draw.rect(surface, (255,255,255), rect, border_radius=12)
    pygame.draw.rect(surface, (226,232,240), rect, 1, border_radius=12)
    title_font = pygame.font.SysFont('arial', 18, bold=True)
    surface.blit(title_font.render(title, True, (17,24,39)), (rect.x+14, rect.y+12))

    content_area = pygame.Rect(rect.x+12, rect.y+44, rect.w-24, rect.h-56)
    prev = push_clip(surface, content_area)

    if kind == "list":
        item_font = pygame.font.SysFont('arial', 16)
        y = content_area.y
        for t in items[:12]:
            line_h = item_font.get_height()
            if y + line_h > content_area.bottom:
                break
            dot = pygame.Rect(content_area.x, y+6, 6, 6)
            pygame.draw.ellipse(surface, (59,130,246), dot)
            # reserve width minus dot + gap
            text_rect = pygame.Rect(content_area.x+16, y, content_area.w-18, line_h)
            draw_text_wrapped_clipped(surface, t, text_rect, item_font, (55,65,81), line_spacing=0, max_lines=1)
            y += max(22, line_h+6)

    elif kind == "tags":
        tag_font = pygame.font.SysFont('arial', 14)
        x = content_area.x; y = content_area.y
        for tag in items:
            pill = tag_font.render(tag, True, (31,41,55))
            pill_rect = pygame.Rect(x, y, pill.get_width()+16, pill.get_height()+10)
            if pill_rect.right > content_area.right:
                x = content_area.x
                y += pill_rect.height + 6
                pill_rect = pygame.Rect(x, y, pill.get_width()+16, pill.get_height()+10)
            if pill_rect.bottom > content_area.bottom:
                break
            pygame.draw.rect(surface, (241,245,249), pill_rect, border_radius=12)
            surface.blit(pill, (pill_rect.x+8, pill_rect.y+5))
            x = pill_rect.right + 6

    elif kind == "newsletter":
        body = "Get weekly deep dives and no-fluff tutorials. No ads. Unsubscribe anytime."
        body_font = pygame.font.SysFont('arial', 16)
        used = draw_text_wrapped_clipped(surface, body, content_area, body_font, (71,85,105))
        y = content_area.y + used + 10
        # input
        inp = pygame.Rect(content_area.x, y, content_area.w, 32)
        if inp.bottom <= content_area.bottom:
            pygame.draw.rect(surface, (249,250,251), inp, border_radius=16)
            pygame.draw.rect(surface, (226,232,240), inp, 1, border_radius=16)
            input_font = pygame.font.SysFont('arial', 16)
            surface.blit(input_font.render("you@company.com", True, (148,163,184)), (inp.x+12, inp.y+7))
            y = inp.bottom + 8
        # button
        btn = pygame.Rect(content_area.x, y, content_area.w, 34)
        if btn.bottom <= content_area.bottom:
            pygame.draw.rect(surface, (59,130,246), btn, border_radius=17)
            btn_font = pygame.font.SysFont('arial', 16, bold=True)
            bt = btn_font.render("Subscribe", True, (255,255,255))
            surface.blit(bt, (btn.x + btn.w//2 - bt.get_width()//2, btn.y + 8))

    pop_clip(surface, prev)

def draw_footer(surface, x, y, w):
    pygame.draw.rect(surface, (17,24,39), (x, y, w, 220))
    title = pygame.font.SysFont('arial', 18, bold=True).render("TechFlow", True, (255,255,255))
    surface.blit(title, (x+28, y+26))
    col_font = pygame.font.SysFont('arial', 16, bold=True)
    item_font = pygame.font.SysFont('arial', 14)
    cols = [
        ("Product", ["Features","Pricing","Changelog","Support"]),
        ("Company", ["About","Blog","Jobs","Press"]),
        ("Resources", ["Docs","Guides","API","Community"]),
        ("Legal", ["Privacy","Terms","Cookies"]),
    ]
    cx = x + 220
    for name, items in cols:
        surface.blit(col_font.render(name, True, (226,232,240)), (cx, y+28))
        yy = y+56
        for it in items:
            surface.blit(item_font.render(it, True, (148,163,184)), (cx, yy))
            yy += 22
        cx += 200
    copy = pygame.font.SysFont('arial', 14).render("© 2025 TechFlow. All rights reserved.", True, (148,163,184))
    surface.blit(copy, (x + w - copy.get_width() - 28, y + 26))

def layout_and_draw_blog(surface, width, height, scroll_y):
    content_top = 80
    view_top, view_bottom = content_top, height

    # background
    pygame.draw.rect(surface, (255,255,255), (0, content_top, width, height-content_top))

    # hero
    hero_h = 420
    hero_rect = pygame.Rect(0, content_top + int(scroll_y), width, hero_h)
    if in_view(hero_rect.y, hero_rect.h, view_top, view_bottom):
        draw_hero(surface, hero_rect.x, hero_rect.y, hero_rect.w, hero_rect.h)

    # grid columns
    pad = 64
    gutter = 28
    main_w = int(width * 0.62)
    side_w = width - pad*2 - gutter - main_w
    main_x = pad
    side_x = main_x + main_w + gutter

    y = content_top + int(scroll_y) + hero_h + 28

    # section: Latest Posts
    section_font = pygame.font.SysFont('arial', 26, bold=True)
    surface.blit(section_font.render("Latest Posts", True, (17,24,39)), (main_x, y))
    y += 12
    pygame.draw.rect(surface, (59,130,246), (main_x, y, 60, 3))
    y += 20

    card_h = 270
    for i, a in enumerate(ARTICLES[:4]):
        rect = pygame.Rect(main_x, y, main_w, card_h)
        if in_view(rect.y, rect.h, view_top, view_bottom):
            draw_article_card(surface, rect, a["title"], a["author"], a["date"], a["tag"], a["excerpt"])
        y += card_h + 16

    # section: Featured
    surface.blit(section_font.render("Featured", True, (17,24,39)), (main_x, y+8))
    pygame.draw.rect(surface, (234,88,12), (main_x, y+40, 80, 3))
    y += 56

    feat_rect = pygame.Rect(main_x, y, main_w, 220)
    if in_view(feat_rect.y, feat_rect.h, view_top, view_bottom):
        draw_article_card(surface, feat_rect,
                          "A Playbook for Sub-Second UIs on the Modern Web",
                          "Rhea Singh", "Sep 10, 2025", "Frontend",
                          "Latency budgets, speculative rendering, streaming HTML, and where hydration actually helps. What we shipped to keep complex dashboards at 60fps.")
    y += feat_rect.h + 40

    # sidebar
    side_y = content_top + int(scroll_y) + hero_h + 28
    sb1 = pygame.Rect(side_x, side_y, side_w, 240)
    if in_view(sb1.y, sb1.h, view_top, view_bottom):
        draw_sidebar_card(surface, sb1, "Popular", POPULAR, kind="list")
    side_y += sb1.h + 16

    sb2 = pygame.Rect(side_x, side_y, side_w, 180)
    if in_view(sb2.y, sb2.h, view_top, view_bottom):
        draw_sidebar_card(surface, sb2, "Tags", TAGS, kind="tags")
    side_y += sb2.h + 16

    sb3 = pygame.Rect(side_x, side_y, side_w, 220)
    if in_view(sb3.y, sb3.h, view_top, view_bottom):
        draw_sidebar_card(surface, sb3, "Newsletter", [], kind="newsletter")
    side_y += sb3.h + 40

    # footer
    page_bottom = max(y, side_y) + 40
    footer_y = page_bottom
    draw_footer(surface, 0, footer_y, width)
    page_total_height = (footer_y + 220) - content_top
    return page_total_height

def draw_wheel(surface, state, window_width, window_height):
    if not state.wheel_active: return
    cx = state.wheel_center_x; cy = state.wheel_center_y; r = state.wheel_radius
    white = (255,255,255)
    for i in range(5):
        rr = r + 15 + i*10; op = int(100 - i*20)
        s = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        pygame.draw.circle(s, (*white, op), (cx, cy), rr, 2); surface.blit(s,(0,0))
    pygame.draw.circle(surface, white, (cx, cy), r, 3)
    pygame.draw.circle(surface, white, (cx, cy), r-20, 1)
    segs = 48; prog = int((state.wheel_angle/(2*math.pi))*segs) % segs; ir = r-10
    for i in range(prog):
        sa = math.radians(i*360/segs)-math.pi/2; ea = math.radians((i+1)*360/segs)-math.pi/2
        sx = cx+int(ir*math.cos(sa)); sy = cy+int(ir*math.sin(sa))
        ex = cx+int(ir*math.cos(ea)); ey = cy+int(ir*math.sin(ea))
        pygame.draw.line(surface, white, (sx,sy), (ex,ey), 5)
    pl = r-28; px = cx+int(pl*math.cos(state.wheel_angle)); py = cy+int(pl*math.sin(state.wheel_angle))
    pygame.draw.line(surface, white, (cx,cy), (px,py), 2)
    pygame.draw.circle(surface, white, (px,py), 5)
    pygame.draw.circle(surface, white, (cx,cy), 7)
    font = pygame.font.SysFont('arial', 24, bold=True)
    t = font.render(f"Zoom {state.zoom_level:.2f}x", True, white)
    surface.blit(t, (cx - t.get_width()//2, cy + r + 18))

# ------------------------- main -------------------------
def main():
    pygame.init()
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("TechFlow — A Hands-First Tech Blog")
    clock = pygame.time.Clock()

    print("="*50)
    print("TECH BLOG BROWSER")
    print("Pinch & drag to scroll • Three-finger rotate to zoom • A-OK to quit")
    print("="*50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera!"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5, model_complexity=0)

    state = BrowserState()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        ret, frame = cap.read()
        if not ret:
            pygame.display.flip(); clock.tick(60); continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        right_hand = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                if hd.classification[0].label == "Right":
                    right_hand = hl.landmark

        # Three-finger wheel -> zoom
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
                if diff > math.pi: diff -= 2*math.pi
                elif diff < -math.pi: diff += 2*math.pi
                state.wheel_angle = (state.wheel_angle + diff*2) % (2*math.pi)
                state.zoom_level = clamp(state.zoom_level + diff*state.zoom_sensitivity,
                                         state.zoom_min, state.zoom_max)
            state.last_finger_angle = ang
        else:
            state.wheel_active = False
            state.last_finger_angle = None

        # A-OK to quit
        ok_now = detect_ok_gesture(right_hand, state.ok_touch_threshold) if right_hand else False
        if ok_now and not state.ok_prev:
            print("A-OK received — closing.")
            running = False
        state.ok_prev = ok_now

        # pinch scroll (only when not in wheel mode)
        if right_hand and not state.wheel_active:
            pinch_now = is_pinching(right_hand, state.pinch_threshold)
            pos = get_pinch_position(right_hand)
            if pinch_now and pos:
                px = pos[0]*WINDOW_WIDTH; py = pos[1]*WINDOW_HEIGHT
                if state.is_pinching and state.last_pinch_y is not None:
                    dy = py - state.last_pinch_y
                    # Drag-style scroll (content follows finger). Flip sign for "natural".
                    state.scroll_y += dy
                    max_scroll_y = max(0, state.page_height - (WINDOW_HEIGHT - 80))
                    state.scroll_y = clamp(state.scroll_y, -max_scroll_y, 0)
                state.last_pinch_x = px; state.last_pinch_y = py; state.is_pinching = True
            else:
                state.is_pinching = False
                state.last_pinch_x = None; state.last_pinch_y = None
        else:
            state.is_pinching = False
            state.last_pinch_x = None; state.last_pinch_y = None
            state.finger_smoother.reset()

        # smooth scroll
        state.smooth_scroll_y += (state.scroll_y - state.smooth_scroll_y) * 0.28

        # draw
        screen.fill((255,255,255))

        # --- ZOOM RENDERING PATH: alpha-safe (no black boxes) ---
        s = state.zoom_level
        if abs(s - 1.0) > 0.001:
            zw = int(WINDOW_WIDTH * s)
            zh = int(WINDOW_HEIGHT * s)

            # Alpha surface + opaque white background to avoid artifacts
            zoom_surf = pygame.Surface((zw, zh), pygame.SRCALPHA).convert_alpha()
            zoom_surf.fill((255, 255, 255, 255))

            # page + chrome onto zoom_surf
            ph = layout_and_draw_blog(zoom_surf, zw, zh, state.smooth_scroll_y * s)
            draw_browser_chrome(zoom_surf, zw, zh)

            # scale back to window, preserving alpha
            scaled = pygame.transform.smoothscale(zoom_surf, (WINDOW_WIDTH, WINDOW_HEIGHT)).convert_alpha()
            screen.blit(scaled, (0,0))

            # update page height for clamp
            state.page_height = int(ph / s)
        else:
            ph = layout_and_draw_blog(screen, WINDOW_WIDTH, WINDOW_HEIGHT, state.smooth_scroll_y)
            draw_browser_chrome(screen, WINDOW_WIDTH, WINDOW_HEIGHT)
            state.page_height = ph

        # wheel overlay
        draw_wheel(screen, state, WINDOW_WIDTH, WINDOW_HEIGHT)

        # hand HUD
        if right_hand:
            tt = right_hand[4]; it = right_hand[8]
            (tx,ty),(ix,iy) = state.finger_smoother.update((tt.x*WINDOW_WIDTH, tt.y*WINDOW_HEIGHT),
                                                           (it.x*WINDOW_WIDTH, it.y*WINDOW_HEIGHT))
            if state.is_pinching:
                pygame.draw.line(screen, (255,100,100), (int(tx),int(ty)), (int(ix),int(iy)), 2)
                pygame.draw.circle(screen, (255,100,100), (int(ix),int(iy)), 8, 2)
            else:
                pygame.draw.circle(screen, (200,200,200), (int(ix),int(iy)), 6, 2)

        # status
        font = pygame.font.SysFont('arial', 18, bold=True)
        status = f"Zoom {state.zoom_level:.2f}x • Scroll {int(-state.scroll_y)}/{max(0, state.page_height - (WINDOW_HEIGHT-80))}"
        srf = font.render(status, True, (90, 98, 110))
        screen.blit(srf, (16, WINDOW_HEIGHT-28))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

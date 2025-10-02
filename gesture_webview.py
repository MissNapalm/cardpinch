# gesture_webview.py
# Gestures:
#  - Quick pinch (short + still) => switch page (cycle tabs)
#  - Pinch + drag => scroll
#  - Three-finger (thumb+index+middle) rotate index => zoom (0.5x–2.0x)
#  - A-OK => quit
#
# Pages: Tech Blog, Twitter, Reddit, Hacker News, Docs
# Design: no images; typographic layouts (lists, pills, code blocks, avatars).

import cv2
import mediapipe as mp
import numpy as np
import math
import pygame
import sys
import random
import time
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
    prev = surface.get_clip()
    surface.set_clip(rect)
    return prev

def pop_clip(surface, prev_clip):
    surface.set_clip(prev_clip)

def draw_rule(surface, x, y, w, color=(230,235,240)):
    pygame.draw.rect(surface, color, (x, y, w, 1))

# word wrap with clipping + optional max_lines + ellipsis
def draw_text_wrapped_clipped(surface, text, rect, font, color,
                              line_spacing=4, antialias=True, max_lines=None, bg=(255,255,255)):
    words = text.split()
    if not words:
        return 0

    # Build lines that fit width
    lines, cur = [], []
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

    prev_clip = push_clip(surface, rect)
    y = rect.y
    used_lines = 0
    height_used = 0
    line_cap = max_lines if max_lines is not None else len(lines)

    for i, line in enumerate(lines[:line_cap]):
        img = font.render(line, antialias, color)
        line_h = img.get_height()
        if y + line_h > rect.bottom:
            # try to ellipsize previous line
            if used_lines > 0:
                ell_text = lines[used_lines - 1] + "…"
                erase_y = y - (line_h + line_spacing)
                pygame.draw.rect(surface, bg, (rect.x, erase_y, rect.width, line_h + line_spacing))
                ell = ell_text
                while font.size(ell)[0] > w_limit and len(ell) > 1:
                    ell = ell[:-2] + "…"
                img2 = font.render(ell, antialias, color)
                surface.blit(img2, (rect.x, erase_y))
                height_used = (erase_y - rect.y) + img2.get_height()
            break

        surface.blit(img, (rect.x, y))
        y += line_h + line_spacing
        used_lines += 1
        height_used = y - rect.y

    pop_clip(surface, prev_clip)
    return min(height_used, rect.height)

def draw_pill(surface, text, x, y, font, bg, fg, pad_x=10, pad_y=6, radius=12, border=None):
    timg = font.render(text, True, fg)
    rect = pygame.Rect(x, y, timg.get_width() + pad_x*2, timg.get_height() + pad_y*2)
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border, rect, 1, border_radius=radius)
    surface.blit(timg, (rect.x + pad_x, rect.y + pad_y))
    return rect

def draw_avatar(surface, cx, cy, r, bg, fg, initials, font):
    pygame.draw.circle(surface, bg, (cx, cy), r)
    img = font.render(initials, True, fg)
    surface.blit(img, (cx - img.get_width()//2, cy - img.get_height()//2))

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
TABS = ["Tech Blog", "Twitter", "Reddit", "Hacker News", "Docs"]
TAB_COUNT = len(TABS)

class BrowserState:
    def __init__(self):
        # scroll (content starts at y=80; page moves with scroll_y in [-max,0])
        self.scroll_y = 0.0
        self.smooth_scroll_y = 0.0
        self.is_pinching = False
        self.last_pinch_x = None
        self.last_pinch_y = None
        self.pinch_threshold = 0.08

        # Quick pinch -> tab switch
        self.pinch_prev = False
        self.pinch_start_time = 0.0
        self.pinch_start_pos = None
        self.pinch_moved = False
        self.pinch_move_threshold = 10  # px
        self.pinch_switch_max_duration = 0.28  # s
        self.pinch_switch_cooldown = 0.45     # s
        self.last_switch_time = 0.0

        # Scroll sensitivity
        self.scroll_gain = 2.2

        # smoothing
        self.finger_smoother = FingerSmoother(window_size=5)

        # A-OK to quit
        self.ok_prev = False
        self.ok_touch_threshold = 0.035

        # page metrics
        self.page_height = 3400

        # Zoom (three-finger rotate)
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

        # active tab
        self.page_style = 0  # 0..4

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

# ------------------------- chrome (tab strip) -------------------------
def draw_tab(surface, rect, text, active=False):
    # slightly rounded top-only tab
    radius = 10
    bg = (255,255,255) if active else (248,249,251)
    border = (210,215,222)
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, 1, border_radius=radius)
    if active:
        pygame.draw.rect(surface, (59,130,246), (rect.x+1, rect.bottom-2, rect.w-2, 2), border_radius=2)
    font = pygame.font.SysFont('arial', 16, bold=True)
    img = font.render(text, True, (25,28,35))
    surface.blit(img, (rect.x + (rect.w - img.get_width())//2, rect.y + (rect.h - img.get_height())//2))

def draw_browser_chrome(surface, width, height, active_tab):
    # top toolbar
    pygame.draw.rect(surface, (255,255,255), (0,0,width,72))
    for i in range(6):
        alpha = 70 - i*12
        s = pygame.Surface((width,1), pygame.SRCALPHA)
        s.fill((0,0,0,alpha))
        surface.blit(s,(0,72+i))

    # tab lane
    lane_y = 6
    lane_h = 28
    x = 10
    for i, name in enumerate(TABS):
        w = 150 if name != "Hacker News" else 160
        r = pygame.Rect(x, lane_y, w, lane_h)
        draw_tab(surface, r, name, active=(i==active_tab))
        x += w + 8

    # address bar (read-only)
    addr_rect = pygame.Rect(120, 40, width-240, 24)
    pygame.draw.rect(surface, (245,247,250), addr_rect, border_radius=12)
    pygame.draw.rect(surface, (226,232,240), addr_rect, 1, border_radius=12)
    font_url = pygame.font.SysFont('consolas', 16)
    addresses = [
        "https://bytecraft.local",
        "https://twitter.local/home",
        "https://reddit.local/r/programming",
        "https://news.ycombinator.local",
        "https://docs.local/guide"
    ]
    surface.blit(font_url.render(addresses[active_tab], True, (60,64,67)), (addr_rect.x+12, addr_rect.y+4))

    # right controls
    pygame.draw.circle(surface, (234,67,53), (width-56, 52), 12)
    for i in range(3):
        pygame.draw.circle(surface, (95,99,104), (width-24, 48+i*6), 2)

# ------------------------- DATA -------------------------
DOMAINS = ["example.com","github.com","medium.com","dev.to","arxiv.org","pypi.org","rust-lang.org","golang.org","fast.ai","numpy.org"]
HN_TITLES = [
    "Ask HN: What's your most underrated CLI trick?",
    "Show HN: Tiny LLM that fits in 20MB",
    "Skip the framework: HTML-first apps that scale",
    "SQLite is enough (most of the time)",
    "We cut CI time by 80% with one flag",
    "What I learned reading the CPython source",
    "A proof that bugs love global state",
    "WebGPU compute made practical",
    "Rust traits clicked for me with this pattern",
    "Notes on writing fast NumPy",
]
def make_hn_items(n=60):
    items = []
    for i in range(n):
        title = random.choice(HN_TITLES) + ("" if random.random()<0.5 else f" #{random.randint(2,9)}")
        points = random.randint(15, 620)
        comments = random.randint(4, 280)
        domain = random.choice(DOMAINS)
        by = random.choice(["alice","bob","carol","dave","eve","mallory","trent"])
        age = random.choice(["1 hour ago","3 hours ago","8 hours ago","1 day ago","2 days ago"])
        items.append({"title": title, "points": points, "comments": comments, "domain": domain, "by": by, "age": age})
    return items
HN_ITEMS = make_hn_items()

REDDIT_TITLES = [
    "TIFU by reinventing a priority queue with lists",
    "PSA: Please stop catching Exception in Python",
    "We shipped a 3ms p99 cache over gRPC — notes",
    "I wrote my first kernel module (it panicked)",
    "Do you log every queue length? You should.",
    "TypeScript: stop exporting everything",
    "Go generics were worth the wait",
    "I removed 14k lines of CI hacks",
    "Regex is a programming language (fight me)",
]
SUBREDDITS = ["programming","devops","learnpython","rust","golang","datascience","machinelearning","cpp","webdev"]
def make_reddit_posts(n=48):
    posts = []
    for _ in range(n):
        title = random.choice(REDDIT_TITLES)
        votes = random.randint(-12, 4200)
        comments = random.randint(5, 800)
        author = random.choice(["u/throwaway42","u/itsjustcode","u/prod_pager","u/nullpointer","u/arraybound"])
        sub = random.choice(SUBREDDITS)
        flair = random.choice(["Discussion","Show","Help","Guide","Story","Opinion","Release"])
        age = random.choice(["1h","3h","8h","1d","2d"])
        posts.append({"title": title, "votes": votes, "comments": comments, "author": author, "sub": sub, "flair": flair, "age": age})
    return posts
REDDIT_POSTS = make_reddit_posts()

# Twitter-like data
TWITTER_AUTHORS = ["Mina Patel","Alex Garner","Jules Ortega","Sofie Novak","Dee Park","Rhea Singh","Bruno Dias","Priya Mehta"]
def initials(name):
    parts = name.split()
    return (parts[0][0] + (parts[1][0] if len(parts)>1 else "")).upper()
TWEETS_TEXT = [
    "Shipping is a skill. Start with boring. Measure. Iterate. Repeat.",
    "TIL: Python's dict resizing strategy is wild — read the source!",
    "We cut cold starts by 60% just by prewarming a tiny pool (like, 3).",
    "WebGPU compute shaders are finally ‘not scary’. Docs are decent too.",
    "Rust makes whole classes of bugs impossible… and also some mistakes louder.",
    "CLI UX matters. Good error messages save hours across a team.",
    "Don’t fight your queue. Embrace eventual consistency where you can.",
    "Every ‘quick script’ becomes infra. Treat it like code on day one.",
]
def make_tweets(n=50):
    tweets = []
    for _ in range(n):
        author = random.choice(TWITTER_AUTHORS)
        handle = "@" + "".join(author.lower().split())
        text = random.choice(TWEETS_TEXT)
        time_label = random.choice(["1m","8m","35m","2h","5h","1d"])
        stats = {
            "replies": random.randint(0, 150),
            "retweets": random.randint(0, 800),
            "likes": random.randint(0, 2500),
            "views": random.randint(1000, 120000)
        }
        tweets.append({"author": author, "handle": handle, "text": text, "time": time_label, "stats": stats})
    return tweets
TWEETS = make_tweets()

# Tech blog data
BLOG_AUTHORS = ["Editorial Team","Nadia (Infra)","Product Announce","Priya @ ML","Ahmed • Security","Bruno (Design)"]
BLOG_TAGS = ["AI","Data","Infra","Frontend","Privacy","DevOps"]
def make_blog_posts(n=24):
    posts = []
    for _ in range(n):
        title = random.choice([
            "Inside Small Language Models: Why Tiny Can Be Mighty",
            "RISC-V on the Edge: Real Workloads, Real Numbers",
            "Privacy by Construction: Telemetry That Doesn’t Phone Home",
            "WebGPU in Production: What Broke and What Stuck",
            "Postgres as a Vector DB Without the Pain",
            "A Playbook for Sub-Second UIs on the Modern Web",
        ])
        author = random.choice(BLOG_AUTHORS)
        minutes = random.randint(5, 16)
        date = random.choice(["Oct 1, 2025","Sep 28, 2025","Sep 22, 2025","Sep 18, 2025"])
        excerpt = random.choice([
            "Distilled architectures and retrieval-aware adapters let sub-1B models punch above their weight.",
            "We profile camera, speech, and tinyML pipelines on commodity boards and contrast them with ARM.",
            "On-device aggregation with noise injection keeps dashboards useful and users untracked.",
            "Shader portability was fine; debugging, not so much. Migration notes from Canvas2D to compute.",
            "With pgvector and HNSW we sustain 50K QPS of hybrid search on a single cluster.",
        ])
        tags = random.sample(BLOG_TAGS, k=random.randint(1,3))
        posts.append({"title": title, "author": author, "date": date, "minutes": minutes, "excerpt": excerpt, "tags": tags})
    return posts
BLOG_POSTS = make_blog_posts()

DOC_SECTIONS = [
    ("Getting Started", "Install the CLI, authenticate, and deploy your first service in under five minutes."),
    ("Configuration", "Every option is a key-value pair. We support YAML, TOML, and JSON with the same schema."),
    ("Scaling", "Choose between request-based autoscaling or concurrency-based autoscaling. Both support cooldowns."),
    ("Observability", "Use structured logs, traces, and custom metrics. All export to OpenTelemetry by default."),
    ("Security", "Service-to-service auth is mutual TLS by default. Rotate credentials without redeploys."),
]

# ------------------------- drawing primitives -------------------------
def draw_code_block(surface, rect, lines):
    pygame.draw.rect(surface, (250, 250, 250), rect, border_radius=8)
    pygame.draw.rect(surface, (225, 228, 232), rect, 1, border_radius=8)
    mono = pygame.font.SysFont('consolas', 16)
    y = rect.y + 8
    clip_prev = push_clip(surface, rect)
    for ln in lines:
        img = mono.render(ln, True, (40, 40, 45))
        if y + img.get_height() > rect.bottom:
            break
        surface.blit(img, (rect.x + 10, y))
        y += img.get_height() + 4
    pop_clip(surface, clip_prev)

# ------------------------- PAGE: Tech Blog -------------------------
def layout_blog(surface, width, height, scroll_y):
    content_top = 80
    view_top, view_bottom = content_top, height
    pygame.draw.rect(surface, (255,255,255), (0, content_top, width, height-content_top))

    # site masthead (left: brand, right: nav)
    mast_y = content_top + int(scroll_y)
    pygame.draw.rect(surface, (255,255,255), (0, mast_y, width, 56))
    title = pygame.font.SysFont('arial', 22, bold=True).render("Bytecraft", True, (30,34,39))
    surface.blit(title, (24, mast_y + 16))
    nav_font = pygame.font.SysFont('arial', 16, bold=True)
    nav_items = ["Articles","Guides","Topics","About"]
    nx = width - 24
    for item in reversed(nav_items):
        img = nav_font.render(item, True, (80,90,100))
        nx -= img.get_width()
        surface.blit(img, (nx, mast_y + 18))
        nx -= 18
    draw_rule(surface, 0, mast_y+56, width)

    # list of posts (clean cardless list, Medium-like)
    y = mast_y + 68
    left_pad = 64
    max_w = min(860, width - left_pad*2)
    h1 = pygame.font.SysFont('arial', 22, bold=True)
    meta = pygame.font.SysFont('arial', 14)
    body = pygame.font.SysFont('arial', 16)
    tag_font = pygame.font.SysFont('arial', 12, bold=True)

    row_h = 128
    first_idx = max(0, (view_top - y) // row_h - 1)
    max_rows = (view_bottom - y) // row_h + 3
    last_idx = min(len(BLOG_POSTS), first_idx + max_rows)

    for i in range(first_idx, last_idx):
        cy = y + i*row_h + first_idx*(-row_h)
        if not in_view(cy, row_h, view_top, view_bottom): 
            continue

        # Title
        draw_text_wrapped_clipped(surface, BLOG_POSTS[i]["title"],
                                  pygame.Rect(left_pad, cy, max_w, 46), h1, (28,32,38), max_lines=2)

        # meta row
        m = f"{BLOG_POSTS[i]['author']} • {BLOG_POSTS[i]['date']} • {BLOG_POSTS[i]['minutes']} min read"
        surface.blit(meta.render(m, True, (120,130,140)), (left_pad, cy+50))

        # excerpt
        draw_text_wrapped_clipped(surface, BLOG_POSTS[i]["excerpt"],
                                  pygame.Rect(left_pad, cy+70, max_w, 36), body, (65,70,78), max_lines=2)

        # tags
        tx = left_pad
        for t in BLOG_POSTS[i]["tags"]:
            pill = draw_pill(surface, t.upper(), tx, cy+104, tag_font, (241,245,249), (31,41,55), radius=10)
            tx = pill.right + 6

        draw_rule(surface, left_pad, cy+row_h-8, max_w)

    page_bottom = y + len(BLOG_POSTS)*row_h + 40
    return (page_bottom - content_top)

# ------------------------- PAGE: Twitter-like -------------------------
def draw_tweet(surface, rect, tw):
    # avatar + name/handle/time
    name_font   = pygame.font.SysFont('arial', 16, bold=True)
    handle_font = pygame.font.SysFont('arial', 14)
    text_font   = pygame.font.SysFont('arial', 18)
    meta_font   = pygame.font.SysFont('arial', 12)

    # avatar
    av_r = 18
    av_bg = (220, 235, 255)
    av_fg = (40, 80, 180)
    draw_avatar(surface, rect.x+av_r+12, rect.y+av_r+12, av_r, av_bg, av_fg, initials(tw["author"]), name_font)

    # name line
    nx = rect.x + 12 + av_r*2 + 12
    ny = rect.y + 8
    name_img = name_font.render(tw["author"], True, (22,24,28))
    handle_img = handle_font.render(f"{tw['handle']} · {tw['time']}", True, (120,130,140))
    surface.blit(name_img, (nx, ny))
    surface.blit(handle_img, (nx + name_img.get_width() + 8, ny + 2))

    # text
    draw_text_wrapped_clipped(surface, tw["text"], pygame.Rect(nx, ny+24, rect.w - (nx-rect.x) - 20, 60),
                              text_font, (25,28,33), max_lines=3)

    # stat bar
    sy = rect.bottom - 28
    icon_font = pygame.font.SysFont('arial', 14, bold=True)
    items = [
        ("💬", tw["stats"]["replies"]),
        ("🔁", tw["stats"]["retweets"]),
        ("❤️", tw["stats"]["likes"]),
        ("👁", tw["stats"]["views"])
    ]
    x = nx
    for ic, n in items:
        label = icon_font.render(f"{ic} {n}", True, (100,110,120))
        surface.blit(label, (x, sy))
        x += label.get_width() + 26

def layout_twitter(surface, width, height, scroll_y):
    content_top = 80
    view_top, view_bottom = content_top, height
    pygame.draw.rect(surface, (255,255,255), (0, content_top, width, height-content_top))

    # left sidebar (Home/Explore/etc) minimal
    left_w = 220
    sidebar_y = content_top + int(scroll_y)
    pygame.draw.rect(surface, (255,255,255), (0, sidebar_y, left_w, height))
    nav = ["Home","Explore","Notifications","Messages","Bookmarks","Profile"]
    nav_font = pygame.font.SysFont('arial', 18, bold=True)
    y = sidebar_y + 8
    for i, item in enumerate(nav):
        col = (25,28,33) if i==0 else (90,100,110)
        surface.blit(nav_font.render(item, True, col), (24, y))
        y += 28

    # main timeline
    main_x = left_w
    main_w = width - main_x - 0
    y = content_top + int(scroll_y)

    # sticky top "Home" header
    header = pygame.Rect(main_x, content_top, main_w, 44)
    pygame.draw.rect(surface, (255,255,255), header)
    draw_rule(surface, main_x, header.bottom, main_w)
    surface.blit(pygame.font.SysFont('arial', 20, bold=True).render("Home", True, (20,20,22)),
                 (main_x + 12, header.y + 10))

    # tweet composer placeholder
    comp = pygame.Rect(main_x, header.bottom, main_w, 64)
    pygame.draw.rect(surface, (255,255,255), comp)
    draw_rule(surface, main_x, comp.bottom, main_w)
    pygame.draw.circle(surface, (230,235,240), (main_x+24, comp.y+32), 12)
    surface.blit(pygame.font.SysFont('arial', 16).render("What's happening?", True, (140,150,160)),
                 (main_x+48, comp.y+22))

    # tweets list
    row_h = 128
    list_y = comp.bottom + int(scroll_y) - content_top  # follow scroll
    first_idx = max(0, (view_top - list_y) // row_h - 2)
    max_rows = (view_bottom - list_y) // row_h + 4
    last_idx = min(len(TWEETS), first_idx + max_rows)

    for i in range(first_idx, last_idx):
        cy = list_y + i*row_h + first_idx*(-row_h)
        rect = pygame.Rect(main_x, cy, main_w, row_h-6)
        if not in_view(rect.y, rect.h, view_top, view_bottom):
            continue
        pygame.draw.rect(surface, (255,255,255), rect)
        draw_tweet(surface, rect, TWEETS[i])
        draw_rule(surface, main_x, rect.bottom, main_w)

    page_bottom = list_y + len(TWEETS)*row_h + 40
    return (page_bottom - content_top)

# ------------------------- PAGE: Reddit-like -------------------------
def draw_vote_column(surface, x, y, h, votes):
    col_w = 56
    block = pygame.Rect(x, y, col_w, h)
    pygame.draw.rect(surface, (250,250,252), block)
    # up arrow
    pygame.draw.polygon(surface, (150,160,170), [(x+28, y+10),(x+18,y+22),(x+38,y+22)])
    # count
    v = pygame.font.SysFont('arial', 16, bold=True).render(str(votes), True, (70,74,80))
    surface.blit(v, (x + (col_w - v.get_width())//2, y + 26))
    # down arrow
    pygame.draw.polygon(surface, (150,160,170), [(x+18, y+48),(x+38,y+48),(x+28,y+60)])

def layout_reddit(surface, width, height, scroll_y):
    content_top = 80
    pygame.draw.rect(surface, (255,255,255), (0, content_top, width, height-content_top))

    # subreddit header
    y0 = content_top + int(scroll_y)
    header = pygame.Rect(0, y0, width, 54)
    pygame.draw.rect(surface, (255,255,255), header)
    title = pygame.font.SysFont('arial', 22, bold=True).render("r/programming", True, (20,20,22))
    surface.blit(title, (16, y0 + 14))
    draw_rule(surface, 0, header.bottom, width)

    # posts
    y = header.bottom + 6
    card_h = 96
    pad_x = 16
    title_font = pygame.font.SysFont('arial', 18, bold=True)
    meta_font  = pygame.font.SysFont('arial', 14)

    first_idx = max(0, (content_top - y) // card_h - 2)
    max_rows = ((height - content_top) // card_h) + 4
    last_idx = min(len(REDDIT_POSTS), first_idx + max_rows)

    for i in range(first_idx, last_idx):
        cy = y + i * card_h + first_idx * (-card_h)
        rect = pygame.Rect(0, cy, width, card_h - 6)
        if not in_view(rect.y, rect.h, content_top, height):
            continue

        pygame.draw.rect(surface, (255,255,255), rect)
        draw_rule(surface, 0, rect.bottom, width)

        draw_vote_column(surface, rect.x, rect.y, rect.h, REDDIT_POSTS[i]["votes"])

        mx = rect.x + 56 + pad_x
        mw = width - mx - 180
        flair = REDDIT_POSTS[i]["flair"]
        pill = draw_pill(surface, flair.upper(), mx, rect.y+10, pygame.font.SysFont('arial', 12, bold=True),
                         (240,245,255), (50,90,200), radius=10, border=(210,220,240))
        draw_text_wrapped_clipped(surface, REDDIT_POSTS[i]["title"],
                                  pygame.Rect(pill.right + 8, rect.y+10, mw - (pill.width + 8), 24),
                                  title_font, (24,28,33), max_lines=1)

        meta = f"{REDDIT_POSTS[i]['author']}  •  r/{REDDIT_POSTS[i]['sub']}  •  {REDDIT_POSTS[i]['age']}  •  {REDDIT_POSTS[i]['comments']} comments"
        surface.blit(meta_font.render(meta, True, (120,130,140)), (mx, rect.y + 40))

        actions = ["Share","Save","Report"]
        ax = width - 12
        a_font = pygame.font.SysFont('arial', 14, bold=True)
        for a in reversed(actions):
            img = a_font.render(a, True, (90,100,115))
            ax -= img.get_width()
            surface.blit(img, (ax, rect.y + 36))
            ax -= 16

    page_bottom = y + len(REDDIT_POSTS)*card_h + 40
    return (page_bottom - content_top)

# ------------------------- PAGE: Hacker News -------------------------
def layout_hn(surface, width, height, scroll_y):
    content_top = 80
    view_top, view_bottom = content_top, height
    pygame.draw.rect(surface, (255,255,255), (0, content_top, width, height-content_top))

    # header bar
    head_h = 44
    bar = pygame.Rect(0, content_top + int(scroll_y), width, head_h)
    pygame.draw.rect(surface, (255, 102, 0), bar)
    name = pygame.font.SysFont('arial', 18, bold=True).render("Hacker News", True, (34,34,34))
    surface.blit(name, (12, bar.y + 12))

    y = bar.bottom + 8
    row_h = 58
    left_pad = 18
    rank_w = 36
    title_font = pygame.font.SysFont('arial', 18)
    meta_font = pygame.font.SysFont('arial', 14)
    num_font  = pygame.font.SysFont('arial', 18, bold=True)

    first_idx = max(0, (view_top - y) // row_h - 2)
    max_rows = (view_bottom - y) // row_h + 4
    last_idx = min(len(HN_ITEMS), first_idx + max_rows)

    for i in range(first_idx, last_idx):
        item_y = y + i * row_h + first_idx * (-row_h)
        rect = pygame.Rect(0, item_y, width, row_h)
        if not in_view(rect.y, rect.h, view_top, view_bottom):
            continue

        pygame.draw.rect(surface, (255,255,255), (0, rect.y, width, rect.h))

        rk = num_font.render(str(i+1)+".", True, (120,120,120))
        surface.blit(rk, (left_pad - 6, rect.y + 8))

        tx = left_pad + rank_w
        title_w = width - tx - 20
        draw_text_wrapped_clipped(surface, HN_ITEMS[i]["title"], pygame.Rect(tx, rect.y + 6, title_w, 24),
                                  title_font, (26, 62, 163), max_lines=1)  # a linky blue
        domain = meta_font.render("(" + HN_ITEMS[i]["domain"] + ")", True, (120,130,140))
        surface.blit(domain, (tx, rect.y + 28))

        meta_str = f"{HN_ITEMS[i]['points']} points by {HN_ITEMS[i]['by']} {HN_ITEMS[i]['age']}  |  {HN_ITEMS[i]['comments']} comments"
        meta = meta_font.render(meta_str, True, (120,130,140))
        surface.blit(meta, (tx, rect.y + 40))

        draw_rule(surface, 0, rect.bottom, width)

    page_bottom = y + len(HN_ITEMS) * row_h + 40
    return (page_bottom - content_top)

# ------------------------- PAGE: Docs -------------------------
def layout_docs(surface, width, height, scroll_y):
    content_top = 80
    pygame.draw.rect(surface, (255,255,255), (0, content_top, width, height-content_top))

    # left nav
    nav_w = 260
    nav_x = 40
    nav_y = content_top + int(scroll_y) + 10
    pygame.draw.rect(surface, (248,249,251), (nav_x, nav_y, nav_w, 520), border_radius=12)
    pygame.draw.rect(surface, (226,232,240), (nav_x, nav_y, nav_w, 520), 1, border_radius=12)
    nav_font = pygame.font.SysFont('arial', 16)
    y = nav_y + 16
    for i, (title, _) in enumerate(DOC_SECTIONS):
        bullet = "▶ " if i == 0 else "• "
        img = nav_font.render(bullet + title, True, (40,44,52))
        surface.blit(img, (nav_x + 12, y))
        y += img.get_height() + 8

    # right "on this page"
    right_w = 220
    right_x = width - right_w - 40
    pygame.draw.rect(surface, (248,249,251), (right_x, nav_y, right_w, 200), border_radius=12)
    pygame.draw.rect(surface, (226,232,240), (right_x, nav_y, right_w, 200), 1, border_radius=12)
    small = pygame.font.SysFont('arial', 14, bold=True)
    surface.blit(small.render("On this page", True, (35,38,45)), (right_x+12, nav_y+12))
    sec_small = pygame.font.SysFont('arial', 14)
    yy = nav_y + 36
    for title, _ in DOC_SECTIONS:
        surface.blit(sec_small.render(title, True, (90,100,110)), (right_x+12, yy))
        yy += 20

    # main content
    main_x = nav_x + nav_w + 24
    main_w = right_x - main_x - 24
    y = content_top + int(scroll_y) + 10

    h1 = pygame.font.SysFont('arial', 30, bold=True)
    h2 = pygame.font.SysFont('arial', 22, bold=True)
    p  = pygame.font.SysFont('arial', 18)

    surface.blit(h1.render("Deploys: A Practical Guide", True, (20,20,25)), (main_x, y))
    y += 42
    para = "Everything you need to ship services reliably: environments, progressive delivery, rollbacks, and safety valves."
    draw_text_wrapped_clipped(surface, para, pygame.Rect(main_x, y, main_w, 60), p, (60,60,70))
    y += 76

    for title, body in DOC_SECTIONS:
        surface.blit(h2.render(title, True, (25,30,40)), (main_x, y))
        y += 30
        used = draw_text_wrapped_clipped(surface, body, pygame.Rect(main_x, y, main_w, 74), p, (65,70,78))
        y += used + 8
        code = [
            "$ tool auth login",
            "$ tool deploy --env=prod",
            "Deploying… done",
            "URL: https://service.example",
        ]
        cb = pygame.Rect(main_x, y, main_w, 110)
        draw_code_block(surface, cb, code)
        y += cb.height + 22

    foot_y = y + 20
    pygame.draw.rect(surface, (245,246,248), (0, foot_y, width, 140))
    surface.blit(pygame.font.SysFont('arial', 16).render("Docs © 2025", True, (80,80,90)), (40, foot_y+20))
    return (foot_y + 140) - content_top

# ------------------------- wheel overlay -------------------------
def draw_wheel(surface, state, window_width, window_height):
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
    pygame.display.set_caption("Gesture WebView — Hands-First Browser")
    clock = pygame.time.Clock()

    print("="*50)
    print("GESTURE WEBVIEW")
    print("Quick pinch (short & still) => switch tab • Pinch + drag => scroll")
    print("Three-finger rotate => zoom • A-OK => quit")
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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                state.page_style = (state.page_style + 1) % TAB_COUNT
                print(f"🔁 Tab switched -> {TABS[state.page_style]}")
                state.scroll_y = state.smooth_scroll_y = 0.0

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

        # Pinch gestures (switch tab vs scroll)
        if right_hand and not state.wheel_active:
            pinch_now = is_pinching(right_hand, state.pinch_threshold)
            pos = get_pinch_position(right_hand)

            if pinch_now and not state.pinch_prev:
                # pinch started
                state.pinch_start_time = time.time()
                if pos:
                    state.last_pinch_x = pos[0]*WINDOW_WIDTH
                    state.last_pinch_y = pos[1]*WINDOW_HEIGHT
                    state.pinch_start_pos = (state.last_pinch_x, state.last_pinch_y)
                state.pinch_moved = False
                state.is_pinching = True

            elif pinch_now and state.pinch_prev and pos:
                # pinch continuing -> if moved, scroll
                px = pos[0]*WINDOW_WIDTH; py = pos[1]*WINDOW_HEIGHT
                if state.last_pinch_y is not None:
                    dy = (py - state.last_pinch_y)
                    if abs(dy) > state.pinch_move_threshold:
                        state.pinch_moved = True
                    if state.pinch_moved:
                        state.scroll_y += dy * state.scroll_gain
                        max_scroll_y = max(0, state.page_height - (WINDOW_HEIGHT - 80))
                        state.scroll_y = clamp(state.scroll_y, -max_scroll_y, 0)
                state.last_pinch_x = px; state.last_pinch_y = py
                state.is_pinching = True

            elif not pinch_now and state.pinch_prev:
                # pinch released -> decide if it's a tab switch
                duration = time.time() - state.pinch_start_time
                now = time.time()
                if (not state.pinch_moved and duration <= state.pinch_switch_max_duration and
                        (now - state.last_switch_time) > state.pinch_switch_cooldown):
                    state.page_style = (state.page_style + 1) % TAB_COUNT
                    state.last_switch_time = now
                    print(f"🔁 Tab switched -> {TABS[state.page_style]}")
                    state.scroll_y = state.smooth_scroll_y = 0.0

                state.is_pinching = False
                state.last_pinch_x = state.last_pinch_y = None
                state.pinch_start_pos = None

            state.pinch_prev = pinch_now
        else:
            state.is_pinching = False
            state.last_pinch_x = None; state.last_pinch_y = None
            state.pinch_prev = False
            state.pinch_start_pos = None
            state.pinch_moved = False

        # smooth scroll
        state.smooth_scroll_y += (state.scroll_y - state.smooth_scroll_y) * 0.28

        # draw
        screen.fill((255,255,255))

        # zoom rendering path (alpha-safe)
        s = state.zoom_level
        if abs(s - 1.0) > 0.001:
            zw = int(WINDOW_WIDTH * s)
            zh = int(WINDOW_HEIGHT * s)
            zoom_surf = pygame.Surface((zw, zh), pygame.SRCALPHA).convert_alpha()
            zoom_surf.fill((255, 255, 255, 255))

            if state.page_style == 0:
                ph = layout_blog(zoom_surf, zw, zh, state.smooth_scroll_y * s)
            elif state.page_style == 1:
                ph = layout_twitter(zoom_surf, zw, zh, state.smooth_scroll_y * s)
            elif state.page_style == 2:
                ph = layout_reddit(zoom_surf, zw, zh, state.smooth_scroll_y * s)
            elif state.page_style == 3:
                ph = layout_hn(zoom_surf, zw, zh, state.smooth_scroll_y * s)
            else:
                ph = layout_docs(zoom_surf, zw, zh, state.smooth_scroll_y * s)

            draw_browser_chrome(zoom_surf, zw, zh, state.page_style)
            scaled = pygame.transform.smoothscale(zoom_surf, (WINDOW_WIDTH, WINDOW_HEIGHT)).convert_alpha()
            screen.blit(scaled, (0,0))
            state.page_height = int(ph / s)
        else:
            if state.page_style == 0:
                ph = layout_blog(screen, WINDOW_WIDTH, WINDOW_HEIGHT, state.smooth_scroll_y)
            elif state.page_style == 1:
                ph = layout_twitter(screen, WINDOW_WIDTH, WINDOW_HEIGHT, state.smooth_scroll_y)
            elif state.page_style == 2:
                ph = layout_reddit(screen, WINDOW_WIDTH, WINDOW_HEIGHT, state.smooth_scroll_y)
            elif state.page_style == 3:
                ph = layout_hn(screen, WINDOW_WIDTH, WINDOW_HEIGHT, state.smooth_scroll_y)
            else:
                ph = layout_docs(screen, WINDOW_WIDTH, WINDOW_HEIGHT, state.smooth_scroll_y)
            draw_browser_chrome(screen, WINDOW_WIDTH, WINDOW_HEIGHT, state.page_style)
            state.page_height = ph

        # wheel overlay
        if state.wheel_active:
            draw_wheel(screen, state, WINDOW_WIDTH, WINDOW_HEIGHT)

        # hand HUD (index fingertip only)
        if right_hand:
            it = right_hand[8]
            ix = int(it.x * WINDOW_WIDTH); iy = int(it.y * WINDOW_HEIGHT)
            pygame.draw.circle(screen, (200,200,200), (ix, iy), 6, 2)

        # status
        font = pygame.font.SysFont('arial', 18, bold=True)
        status = f"{TABS[state.page_style]} • Zoom {state.zoom_level:.2f}x • Scroll {int(-state.scroll_y)}/{max(0, state.page_height - (WINDOW_HEIGHT-80))}"
        srf = font.render(status, True, (90, 98, 110))
        screen.blit(srf, (16, WINDOW_HEIGHT-28))

        pygame.display.flip()
        clock.tick(60)

    cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

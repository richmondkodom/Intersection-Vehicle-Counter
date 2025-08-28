import os
import cv2
import time
import math
import urllib.request
import tempfile
import numpy as np
import streamlit as st
import pandas as pd
import json, hashlib

from collections import deque, defaultdict

# ===============================
# APPLY CUSTOM THEME (Dark/Light)
# ===============================
def apply_custom_theme(theme="dark", colors=("ff6ec7", "00c9a7")):
    """Apply CSS theme: 'dark' or 'light'. colors used as primary/accent (hex without #)."""
    if theme == "dark":
        color1, color2 = colors
        custom_css = f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
            font-family: "Segoe UI", sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(0, 0, 0, 0.75);
            color: white;
            border-right: 2px solid rgba(255,255,255,0.2);
        }}
        .block-container {{
            background: rgba(0, 0, 0, 0.55);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
        }}
        .stImage img {{
            border-radius: 20px;
            box-shadow: 0px 6px 25px rgba(0,0,0,0.7);
            border: 3px solid rgba(255, 255, 255, 0.35);
        }}
        .dataframe {{
            border-radius: 12px !important;
            overflow: hidden !important;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
            background: rgba(255, 255, 255, 0.05);
            color: white !important;
        }}
        [data-testid="stMetric"] {{
            background: rgba(255, 255, 255, 0.08);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
        }}
        [data-testid="stMetricValue"] {{
            color: #{color1};
            font-weight: 700;
        }}
        .stButton>button, .stDownloadButton>button {{
            background: linear-gradient(135deg, #{color1}, #{color2});
            color: white;
            font-weight: 600;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1.2rem;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
            transition: all 0.2s ease-in-out;
        }}
        .stButton>button:hover, .stDownloadButton>button:hover {{
            transform: translateY(-2px) scale(1.02);
        }}
        </style>
        """
    else:
        # simple light theme
        color1, color2 = colors
        custom_css = f"""
        <style>
        .stApp {{
            background: linear-gradient(to bottom right, #f0f0f0, #ffffff);
            color: #222;
            font-family: "Segoe UI", sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background: #f7f7f7;
            color: #333;
            border-right: 2px solid #ddd;
        }}
        .block-container {{
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.12);
        }}
        .stImage img {{
            border-radius: 12px;
            box-shadow: 0px 4px 16px rgba(0,0,0,0.12);
            border: 2px solid rgba(0,0,0,0.06);
        }}
        .dataframe {{
            border-radius: 10px !important;
            overflow: hidden !important;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
            background: white;
            color: black !important;
        }}
        [data-testid="stMetric"] {{
            background: #f9f9f9;
            padding: 0.8rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        }}
        [data-testid="stMetricValue"] {{
            color: #{color1};
            font-weight: 700;
        }}
        .stButton>button, .stDownloadButton>button {{
            background: linear-gradient(135deg, #{color1}, #{color2});
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            border: none;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.12);
        }}
        </style>
        """
    st.markdown(custom_css, unsafe_allow_html=True)

# ===============================
# USER & THEME MANAGEMENT (files)
# ===============================
USERS_FILE = "users.json"
THEMES_FILE = "user_themes.json"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def load_themes():
    if os.path.exists(THEMES_FILE):
        with open(THEMES_FILE, "r") as f:
            return json.load(f)
    return {}

def save_themes(themes):
    with open(THEMES_FILE, "w") as f:
        json.dump(themes, f)

def load_user_theme(username):
    themes = load_themes()
    return themes.get(username, {"theme": "dark", "colors": ("ff6ec7", "00c9a7")})

def save_user_theme(username, theme_data):
    themes = load_themes()
    themes[username] = theme_data
    save_themes(themes)

# ===============================
# MODEL SETUP (YOLOv4-tiny)
# ===============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
URLS = {
    "weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}
FILES = {
    "weights": os.path.join(MODEL_DIR, "yolov4-tiny.weights"),
    "cfg": os.path.join(MODEL_DIR, "yolov4-tiny.cfg"),
    "names": os.path.join(MODEL_DIR, "coco.names"),
}

def ensure_model_files():
    for k, path in FILES.items():
        if not os.path.exists(path):
            try:
                st.info(f"Downloading {k}...")
                urllib.request.urlretrieve(URLS[k], path)
            except Exception as e:
                st.error(f"Failed to download {k}: {e}")
                st.stop()

ensure_model_files()

with open(FILES["names"], "r") as f:
    CLASSES = [c.strip() for c in f.readlines()]
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle"}

net = cv2.dnn.readNetFromDarknet(FILES["cfg"], FILES["weights"])
if net.empty():
    st.error("Failed to load YOLOv4-tiny network.")
    st.stop()
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ===============================
# TRACKER
# ===============================
class Track:
    def __init__(self, track_id, centroid):
        self.id = track_id
        self.trace = deque(maxlen=20)
        self.trace.append(centroid)
        self.counted_crossings = {"h": False, "v": False}
        self.cls = None
        self.last_seen = time.time()

class CentroidTracker:
    def __init__(self, max_distance=50, max_age=2.0):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.max_age = max_age

    @staticmethod
    def _euclidean(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, detections):
        if detections is None:
            return {}

        now = time.time()
        to_del = [tid for tid, t in self.tracks.items() if (now - t.last_seen) > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        assigned = set()
        out = {}

        for det in detections:
            dcx, dcy, w, h, cname, conf = det
            best_id, best_dist = None, 1e9
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                dist = self._euclidean((dcx, dcy), tr.trace[-1])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None and best_dist <= self.max_distance:
                tr = self.tracks[best_id]
                tr.trace.append((dcx, dcy))
                tr.last_seen = now
                if tr.cls is None:
                    tr.cls = cname
                assigned.add(best_id)
                out[best_id] = (dcx, dcy, w, h, tr.cls or cname, conf)
            else:
                tid = self.next_id
                self.next_id += 1
                tr = Track(tid, (dcx, dcy))
                tr.cls = cname
                tr.last_seen = now
                self.tracks[tid] = tr
                assigned.add(tid)
                out[tid] = (dcx, dcy, w, h, cname, conf)

        return out

# ===============================
# VEHICLE DETECTION
# ===============================
def detect_vehicles(frame, conf_thresh=0.2, nms_thresh=0.4, target_classes=None, input_size=416):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    try:
        outs = net.forward(output_layers)
    except cv2.error as e:
        st.error(f"Error during forward pass: {e}")
        return []

    boxes, confs, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_thresh:
                cx = int(det[0] * w)
                cy = int(det[1] * h)
                bw = int(det[2] * w)
                bh = int(det[3] * h)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                cname = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
                if target_classes and cname not in target_classes:
                    continue
                boxes.append([x, y, bw, bh])
                confs.append(confidence)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            cx = x + bw // 2
            cy = y + bh // 2
            cname = CLASSES[class_ids[i]] if class_ids[i] < len(CLASSES) else str(class_ids[i])
            detections.append((cx, cy, bw, bh, cname, confs[i]))
    return detections

# ===============================
# APP ENTRY POINT
# ===============================
st.set_page_config(page_title="🚦 Vehicle Counter Dashboard", layout="wide")

# ensure session keys exist
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "theme_settings" not in st.session_state:
    st.session_state.theme_settings = {"theme": "dark", "colors": ("ff6ec7", "00c9a7")}

# LOGIN / REGISTER
if not st.session_state.logged_in:
    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        st.header("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            users = load_users()
            if username in users and users[username] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.theme_settings = load_user_theme(username)
                apply_custom_theme(st.session_state.theme_settings["theme"], st.session_state.theme_settings["colors"])
                st.success(f"Welcome back, {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    with tab_register:
        st.header("Register New Account")
        new_user = st.text_input("Choose a username", key="reg_user")
        new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        if st.button("Register"):
            users = load_users()
            if new_user in users:
                st.error("Username already exists.")
            elif new_user.strip() == "" or new_pass.strip() == "":
                st.error("Please enter a valid username and password.")
            else:
                users[new_user] = hash_password(new_pass)
                save_users(users)
                save_user_theme(new_user, {"theme": "dark", "colors": ("ff6ec7", "00c9a7")})
                st.success("Registration successful! Please log in.")

# MAIN APP AFTER LOGIN
else:
    # load/apply user's theme
    if "username" not in st.session_state:
        st.session_state.username = st.session_state.get("username", "user")
    user_theme = st.session_state.theme_settings
    apply_custom_theme(user_theme.get("theme", "dark"), user_theme.get("colors", ("ff6ec7","00c9a7")))

    st.sidebar.success(f"👤 Logged in as: {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    # THEME CONTROLS (persist)
    st.sidebar.header("🎨 Theme Settings")
    theme_choice = st.sidebar.selectbox(
        "Theme", ["dark", "light"],
        index=0 if st.session_state.theme_settings["theme"] == "dark" else 1
    )
    color1 = st.sidebar.color_picker("Primary Color", "#" + st.session_state.theme_settings["colors"][0])
    color2 = st.sidebar.color_picker("Accent Color", "#" + st.session_state.theme_settings["colors"][1])
    st.session_state.theme_settings = {"theme": theme_choice, "colors": (color1.lstrip("#"), color2.lstrip("#"))}
    save_user_theme(st.session_state.username, st.session_state.theme_settings)
    # reapply after user changes
    apply_custom_theme(st.session_state.theme_settings["theme"], st.session_state.theme_settings["colors"])

    # APP UI
    st.title("🚗 Intersection Vehicle Counter")

    with st.sidebar:
        st.header("Settings")
        source = st.radio("Source", ["Upload Video", "Webcam"], index=0)
        conf_thresh = st.slider("Detection confidence", 0.1, 0.9, 0.20, 0.05)
        nms_thresh = st.slider("NMS threshold", 0.1, 0.9, 0.45, 0.05)
        input_size = st.select_slider("Model input size", options=[320, 416, 512, 608], value=416)
        max_distance = st.slider("Tracker max match distance (px)", 10, 150, 60, 5)
        max_age = st.slider("Tracker max age (sec)", 1.0, 5.0, 2.0, 0.5)

        st.markdown("**Count Lines**")
        line_mode = st.selectbox("Which lines to use for counting?", ["Horizontal & Vertical", "Horizontal only", "Vertical only"], index=0)
        h_ratio = st.slider("Horizontal line position (height ratio)", 0.1, 0.9, 0.5, 0.05)
        v_ratio = st.slider("Vertical line position (width ratio)", 0.1, 0.9, 0.5, 0.05)

        st.markdown("**Classes**")
        selected_classes = st.multiselect("Vehicle classes to detect", sorted(list(VEHICLE_CLASSES)), default=list(VEHICLE_CLASSES))

        draw_boxes = st.checkbox("Draw boxes", value=True)
        show_ids = st.checkbox("Show track IDs", value=True)
        show_trace = st.checkbox("Draw motion trails", value=True)
        fps_display = st.checkbox("Show FPS", value=True)

    uploaded_video = None
    cap = None

    if source == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    else:
        cam_index = st.number_input("Webcam index", value=0, step=1, min_value=0)

    start_btn = st.button("▶️ Start")

    direction_counts = {"left_to_right":0, "right_to_left":0, "up_to_down":0, "down_to_up":0}
    class_totals = {cls: 0 for cls in selected_classes}
    events = []

    stats_placeholder = st.sidebar.empty()
    direction_placeholder = st.sidebar.empty()
    frame_holder = st.empty()

    if start_btn:
        if source == "Upload Video":
            if uploaded_video is None:
                st.warning("Please upload a video first.")
                st.stop()
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(int(cam_index))

        if not cap.isOpened():
            st.error("Could not open video source.")
            st.stop()

        tracker = CentroidTracker(max_distance=max_distance, max_age=max_age)
        fps_time = time.time()
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            h, w = frame.shape[:2]

            h_line_y = int(h * h_ratio)
            v_line_x = int(w * v_ratio)
            use_h = line_mode in ("Horizontal & Vertical", "Horizontal only")
            use_v = line_mode in ("Horizontal & Vertical", "Vertical only")

            dets = detect_vehicles(frame, conf_thresh, nms_thresh, set(selected_classes), input_size)
            tracks = tracker.update(dets)

            if use_h:
                cv2.line(frame, (0, h_line_y), (w, h_line_y), (0, 255, 255), 2)
            if use_v:
                cv2.line(frame, (v_line_x, 0), (v_line_x, h), (255, 255, 0), 2)

            for tid, (cx, cy, bw, bh, cname, conf) in tracks.items():
                tr = tracker.tracks[tid]
                if show_trace and len(tr.trace) >= 2:
                    for i in range(1, len(tr.trace)):
                        cv2.line(frame, tr.trace[i-1], tr.trace[i], (200,200,200), 2)
                if draw_boxes:
                    x = int(cx - bw/2); y = int(cy - bh/2)
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0,255,0), 2)
                label = f"{cname} {int(conf*100)}%"
                if show_ids:
                    label = f"ID {tid} | " + label
                cv2.putText(frame, label, (int(cx - bw/2), int(max(0,y-8))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,220,10), 2)

                if len(tr.trace) >= 2:
                    px, py = tr.trace[-2]
                    dx = cx - px
                    dy = cy - py
                    event_time = time.strftime("%H:%M:%S", time.localtime())

                    # Horizontal crossings
                    if use_h and not tr.counted_crossings["h"]:
                        if (py < h_line_y <= cy) or (py > h_line_y >= cy):
                            if dy > 0:
                                direction_counts["up_to_down"] += 1
                                events.append((tid, "up_to_down", tr.cls, frame_idx, event_time))
                            else:
                                direction_counts["down_to_up"] += 1
                                events.append((tid, "down_to_up", tr.cls, frame_idx, event_time))
                            if tr.cls in class_totals:
                                class_totals[tr.cls] += 1
                            tr.counted_crossings["h"] = True

                    # Vertical crossings
                    if use_v and not tr.counted_crossings["v"]:
                        if (px < v_line_x <= cx) or (px > v_line_x >= cx):
                            if dx > 0:
                                direction_counts["left_to_right"] += 1
                                events.append((tid, "left_to_right", tr.cls, frame_idx, event_time))
                            else:
                                direction_counts["right_to_left"] += 1
                                events.append((tid, "right_to_left", tr.cls, frame_idx, event_time))
                            if tr.cls in class_totals:
                                class_totals[tr.cls] += 1
                            tr.counted_crossings["v"] = True

            # overlay totals on frame
            overlay_lines = []
            overlay_lines.append(" | ".join([f"{cls.capitalize()}: {cnt}" for cls, cnt in class_totals.items()]))
            overlay_lines.append(f"Total: {sum(class_totals.values())}")
            overlay_lines.append(f"L→R: {direction_counts['left_to_right']} | R→L: {direction_counts['right_to_left']}")
            overlay_lines.append(f"U→D: {direction_counts['up_to_down']} | D→U: {direction_counts['down_to_up']}")

            y0 = 40
            for i, line in enumerate(overlay_lines):
                y = y0 + i * 25
                cv2.putText(frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # live sidebar stats
            stats_placeholder.write("### 🚘 Vehicle Class Counts")
            stats_placeholder.write(pd.DataFrame(list(class_totals.items()), columns=["Class", "Count"]))

            direction_placeholder.write("### 🧭 Direction Counts")
            direction_placeholder.write(pd.DataFrame([
                ["Left → Right", direction_counts["left_to_right"]],
                ["Right → Left", direction_counts["right_to_left"]],
                ["Up → Down", direction_counts["up_to_down"]],
                ["Down → Up", direction_counts["down_to_up"]],
            ], columns=["Direction", "Count"]))

            # FPS overlay
            if fps_display:
                now = time.time()
                fps = 1.0 / max(1e-6, now - fps_time)
                fps_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,180,255), 2)

            # show frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_holder.image(frame_rgb, channels="RGB", use_column_width=True)

        # end while
        cap.release()
        st.success("Finished processing video.")
        total = sum(direction_counts.values())
        st.metric("Grand Total", total)

        if events:
            df = pd.DataFrame(events, columns=["track_id","direction","class","frame","timestamp"])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Log (CSV)", csv, file_name="vehicle_counts.csv", mime="text/csv")

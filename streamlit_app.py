import os
import cv2
import time
import math
import urllib.request
import tempfile
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import hashlib
import sqlite3
from datetime import datetime
import json

# Configure Streamlit before displaying any UI elements.
st.set_page_config(page_title="🚗 Intersection Vehicle Counter", layout="wide")

# ===============================
# Password Hashing
# ===============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ===============================
# Database Setup
# ===============================
DB_FILE = "users.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create users table with is_admin column (if not exists)
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            password TEXT NOT NULL,
            created_at TEXT,
            last_login TEXT,
            is_admin INTEGER DEFAULT 0
        )
    """)
    # Create activity log table
    c.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()

    # Ensure is_admin column exists for older DBs (defensive)
    # Check pragma and add column if missing (SQLite allows ADD COLUMN)
    try:
        cols = [row[1] for row in c.execute("PRAGMA table_info(users)").fetchall()]
        if "is_admin" not in cols:
            c.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
            conn.commit()
    except Exception:
        pass

    # Create a default admin if none exists
    try:
        c.execute("SELECT username FROM users WHERE is_admin=1 LIMIT 1")
        if not c.fetchone():
            default_admin_pass = hash_password("admin123")
            try:
                c.execute("INSERT OR IGNORE INTO users (username, email, password, created_at, last_login, is_admin) VALUES (?,?,?,?,?,?)",
                          ("admin", "admin@example.com", default_admin_pass, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), None, 1))
                conn.commit()
            except Exception:
                pass
    except Exception:
        pass

    conn.close()

def add_user(username, email, password_hash, is_admin=0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password, created_at, last_login, is_admin) VALUES (?,?,?,?,?,?)",
                  (username, email, password_hash,
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S"), None, is_admin))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT username, email, password, created_at, last_login, is_admin FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    return row

def update_password(username, new_hash):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET password=? WHERE username=?", (new_hash, username))
    conn.commit()
    conn.close()

def update_last_login(username):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET last_login=? WHERE username=?",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username))
    conn.commit()
    conn.close()

def log_activity(username, action):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO activity_log (username, action, timestamp) VALUES (?, ?, ?)",
              (username, action, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

# Initialize DB
init_db()

# ===============================
# Session state
# ===============================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

# ===============================
# Authentication Pages
# ===============================
def login_page():
    st.subheader("🔐 Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        user = get_user(username)
        if user and user[2] == hash_password(password):  # user[2] = password
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.session_state["is_admin"] = bool(user[5])
            update_last_login(username)
            log_activity(username, "Logged in")
            st.success(f"Welcome back, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

def register_page():
    st.subheader("🆕 Register")
    username = st.text_input("Choose a username", key="reg_username")
    email = st.text_input("Enter email", key="reg_email")
    password = st.text_input("Choose a password", type="password", key="reg_password")
    confirm = st.text_input("Confirm password", type="password", key="reg_confirm")

    if st.button("Register"):
        if not username or not email:
            st.warning("Please fill all fields")
        elif password != confirm:
            st.warning("Passwords do not match")
        elif len(password) < 4:
            st.warning("Password must be at least 4 characters")
        else:
            password_hash = hash_password(password)
            success = add_user(username, email, password_hash, is_admin=0)
            if success:
                st.success("✅ Account created! You can now log in.")
                log_activity(username, "Registered account")
            else:
                st.error("⚠️ Username or email already exists")

def reset_password_page():
    st.subheader("🔄 Reset Password")
    username = st.text_input("Username", key="reset_username")
    new_password = st.text_input("New password", type="password", key="reset_new")
    confirm = st.text_input("Confirm new password", type="password", key="reset_confirm")

    if st.button("Reset"):
        user = get_user(username)
        if not user:
            st.error("⚠️ Username not found")
        elif new_password != confirm:
            st.error("Passwords do not match")
        else:
            update_password(username, hash_password(new_password))
            log_activity(st.session_state.get("user") or username, f"Password reset for {username}")
            st.success("✅ Password updated successfully! Please log in.")

def logout_button():
    if st.sidebar.button("🚪 Logout"):
        if st.session_state.get("user"):
            log_activity(st.session_state["user"], "Logged out")
        st.session_state["logged_in"] = False
        st.session_state["user"] = None
        st.session_state["is_admin"] = False
        st.rerun()

# ===============================
# App setup & style
# ===============================
page_bg = """
<style>
[data-testid="stAppViewContainer"] { background-color: #f8fafc; }
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"], [data-testid="stSidebar"] * { color: #ffffff !important; fill: #ffffff !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.stButton > button, .stDownloadButton > button {
    background-color: #2563eb; color: white !important; border-radius: 12px; border: 0;
    padding: 0.75em 1.25em; font-weight: 700; font-size: 16px; cursor: pointer;
    transition: 0.2s ease-in-out; box-shadow: 0px 6px 20px rgba(37, 99, 235, 0.35);
}
.stButton > button:hover, .stDownloadButton > button:hover { background-color: #1e40af; transform: translateY(-2px); }
[data-testid="stMetricValue"] { font-size: 30px !important; font-weight: 800; }
[data-testid="stMetricLabel"] { font-size: 14px !important; font-weight: 600; color: #64748b !important; }
.block-container { padding-top: 1rem; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===============================
# Authentication Flow
# ===============================
if not st.session_state["logged_in"]:
    st.sidebar.title("Authentication")
    menu = st.sidebar.radio("Choose", ["Login", "Register", "Reset Password"])
    if menu == "Login":
        login_page()
    elif menu == "Register":
        register_page()
    elif menu == "Reset Password":
        reset_password_page()
    st.stop()
else:
    logout_button()

# If logged in and admin, show admin nav in sidebar
if st.session_state.get("is_admin"):
    with st.sidebar:
        st.markdown("---")
        st.subheader("Admin")
        admin_page_nav = st.radio("Admin Pages", ["Admin Dashboard", "User Manager", "Activity Logs"], index=0)
else:
    admin_page_nav = None

# ===============================
# 🚗 Main Vehicle Counter App (unchanged)
# ===============================
st.title("🚗 Intersection Vehicle Counter")
st.caption(f"Welcome, **{st.session_state['user']}**! Detecting crossings and showing live **East / West / North / South** stats.")

# -------------------------------
# Model Setup
# -------------------------------
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
            st.info(f"Downloading {k}...")
            urllib.request.urlretrieve(URLS[k], path)
ensure_model_files()

with open(FILES["names"], "r") as f:
    CLASSES = [c.strip() for c in f.readlines()]
DETECTABLE_CLASSES = {"person", "car", "bus", "truck", "motorbike", "bicycle"}

# Load the YOLOv4-tiny network. Some OpenCV builds expose readNetFromDarknet,
# while others expose the more general readNet() API.
try:
    if hasattr(cv2.dnn, "readNetFromDarknet"):
        net = cv2.dnn.readNetFromDarknet(FILES["cfg"], FILES["weights"])
    else:
        net = cv2.dnn.readNet(FILES["weights"], FILES["cfg"])

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    unconnected = np.asarray(unconnected).reshape(-1)
    output_layers = [layer_names[int(i) - 1] for i in unconnected]
    if not output_layers:
        raise RuntimeError("OpenCV did not return any YOLO output layers.")
except Exception as e:
    st.error(
        "Could not load the YOLO model. Check that OpenCV is installed correctly "
        "and that yolov4-tiny.cfg and yolov4-tiny.weights downloaded successfully."
    )
    st.exception(e)
    st.stop()

# -------------------------------
# Tracker
# -------------------------------
class Track:
    def __init__(self, track_id, centroid):
        self.id = track_id
        self.trace = deque(maxlen=20)
        self.trace.append(centroid)
        self.counted_crossings = {"h": False, "v": False}
        self.cls = None
        self.last_seen = time.time()

class CentroidTracker:
    def __init__(self, max_distance=60, max_age=2.0):
        self.next_id = 1
        self.tracks = {}
        self.max_distance = max_distance
        self.max_age = max_age
    @staticmethod
    def _euclidean(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
    def update(self, detections):
        if detections is None: return {}
        now = time.time()
        # remove stale tracks
        to_del = [tid for tid, t in self.tracks.items() if (now - t.last_seen) > self.max_age]
        for tid in to_del:
            try:
                del self.tracks[tid]
            except KeyError:
                pass
        assigned, out = set(), {}
        for det in detections:
            dcx, dcy, w, h, cname, conf = det
            best_id, best_dist = None, 1e9
            for tid, tr in self.tracks.items():
                if tid in assigned: continue
                dist = self._euclidean((dcx,dcy), tr.trace[-1])
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None and best_dist <= self.max_distance:
                tr = self.tracks[best_id]
                tr.trace.append((dcx,dcy))
                tr.last_seen = now
                if tr.cls is None: tr.cls = cname
                assigned.add(best_id)
                out[best_id] = (dcx,dcy,w,h,tr.cls or cname,conf)
            else:
                tid = self.next_id; self.next_id += 1
                tr = Track(tid,(dcx,dcy)); tr.cls=cname; tr.last_seen=now
                self.tracks[tid] = tr; assigned.add(tid)
                out[tid] = (dcx,dcy,w,h,cname,conf)
        return out

# -------------------------------
# Detection Helper
# -------------------------------
def detect_objects(frame, conf_thresh=0.2, nms_thresh=0.4, target_classes=None, input_size=416):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size,input_size), swapRB=True, crop=False)
    net.setInput(blob)
    try:
        outs = net.forward(output_layers)
    except cv2.error as e:
        st.error(f"Forward pass error: {e}")
        return []
    boxes, confs, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id]) if len(scores) > 0 else 0.0
            if confidence > conf_thresh:
                cx=int(det[0]*w); cy=int(det[1]*h)
                bw=int(det[2]*w); bh=int(det[3]*h)
                x=int(cx-bw/2); y=int(cy-bh/2)
                cname = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
                if target_classes and cname not in target_classes: continue
                boxes.append([x,y,bw,bh]); confs.append(float(confidence)); class_ids.append(class_id)
    if len(boxes) == 0:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y,bw,bh = boxes[i]
            cx=x+bw//2; cy=y+bh//2
            cname = CLASSES[class_ids[i]] if class_ids[i] < len(CLASSES) else str(class_ids[i])
            detections.append((cx,cy,bw,bh,cname,confs[i]))
    return detections

# -------------------------------
# Sidebar Settings
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio("Source", ["Upload Video", "Webcam"], index=0)
    conf_thresh = st.slider("Detection confidence",0.1,0.9,0.2,0.05)
    nms_thresh = st.slider("NMS threshold",0.1,0.9,0.45,0.05)
    input_size = st.select_slider("Model input size", options=[320,416,512,608], value=416)
    max_distance = st.slider("Tracker match distance (px)",10,200,60,5)
    max_age = st.slider("Tracker max age (sec)",0.5,5.0,2.0,0.5)
    st.markdown("**Count Lines**")
    line_mode = st.selectbox("Counting lines", ["Horizontal & Vertical","Horizontal only","Vertical only"], index=0)
    h_ratio = st.slider("Horizontal line (height ratio)",0.1,0.9,0.5,0.05)
    v_ratio = st.slider("Vertical line (width ratio)",0.1,0.9,0.5,0.05)
    st.markdown("**Classes**")
    selected_classes = st.multiselect("Detect classes", sorted(list(DETECTABLE_CLASSES)), default=list(DETECTABLE_CLASSES))
    draw_boxes = st.checkbox("Draw boxes", value=True)
    show_ids = st.checkbox("Show track IDs", value=True)
    show_trace = st.checkbox("Draw motion trails", value=True)
    fps_display = st.checkbox("Show FPS", value=True)
    st.markdown("### 📊 Dashboard View")
    dashboard_view = st.radio("Choose view", ["Bar View","Line View","Combined View"], index=0)
    show_pies = st.checkbox("Show pie charts (Combined View)", value=True)

uploaded_video = None
cap = None
if source == "Upload Video":
    st.caption("Large uploads: configure Streamlit with .streamlit/config.toml. A 10 GB limit also depends on your hosting provider.")
    uploaded_video = st.file_uploader(
        "Upload a video (up to 10 GB when supported by your host)",
        type=["mp4", "mov", "avi", "mkv"],
        help="For very large files, use a host with sufficient upload limits, disk space, RAM, and processing time."
    )
else:
    cam_index = st.number_input("Webcam index", value=0, step=1, min_value=0)
start_btn = st.button("▶️ Start")

# -------------------------------
# Admin Dashboard & Tools
# -------------------------------
def admin_dashboard():
    st.title("🛠️ Admin Dashboard")
    conn = sqlite3.connect(DB_FILE)

    # Show users table
    st.subheader("👥 Registered Users")
    users_df = pd.read_sql_query("SELECT username, email, created_at, last_login, is_admin FROM users", conn)
    st.dataframe(users_df, use_container_width=True)

    # Download users CSV
    csv_data = users_df.to_csv(index=False).encode()
    st.download_button("📥 Download User List (CSV)", csv_data, "users.csv", "text/csv")

    st.divider()

    # Delete user
    st.subheader("❌ Delete a User")
    del_col1, del_col2 = st.columns([3,1])
    with del_col1:
        username_to_delete = st.text_input("Enter username to delete", key="admin_del_user")
    with del_col2:
        if st.button("Delete", use_container_width=True):
            if not username_to_delete.strip():
                st.warning("Enter a username")
            elif username_to_delete == st.session_state["user"]:
                st.error("You cannot delete your own account while logged in.")
            else:
                c = conn.cursor()
                c.execute("DELETE FROM users WHERE username=?", (username_to_delete,))
                if c.rowcount == 0:
                    st.error(f"User '{username_to_delete}' not found.")
                else:
                    conn.commit()
                    st.success(f"User '{username_to_delete}' deleted.")
                    log_activity(st.session_state["user"], f"Deleted user '{username_to_delete}'")
                    # refresh users_df
                    users_df = pd.read_sql_query("SELECT username, email, created_at, last_login, is_admin FROM users", conn)
                    st.experimental_rerun()

    st.divider()

    # Promote / Demote
    st.subheader("🧩 Promote / Demote User")
    role_col1, role_col2 = st.columns([3,2])
    with role_col1:
        username_role = st.text_input("Username to modify role", key="admin_role_user")
    with role_col2:
        role_action = st.selectbox("Action", ["Promote to Admin", "Demote to Regular User"])
    if st.button("Apply Role Change"):
        if not username_role.strip():
            st.warning("Enter a username")
        elif username_role == st.session_state["user"]:
            st.warning("You cannot change your own role while logged in.")
        else:
            new_status = 1 if role_action == "Promote to Admin" else 0
            c = conn.cursor()
            c.execute("UPDATE users SET is_admin=? WHERE username=?", (new_status, username_role))
            if c.rowcount == 0:
                st.error(f"User '{username_role}' not found.")
            else:
                conn.commit()
                st.success(f"{username_role} role updated.")
                log_activity(st.session_state["user"], f"Changed role for '{username_role}' to {'Admin' if new_status else 'User'}")
                st.experimental_rerun()

    st.divider()

    # Reset Password
    st.subheader("🔑 Reset User Password")
    rp_col1, rp_col2 = st.columns([3,2])
    with rp_col1:
        user_to_reset = st.text_input("Enter username to reset password", key="admin_reset_user")
    with rp_col2:
        new_pass = st.text_input("New password", type="password", key="admin_new_pass")
    if st.button("Reset Password (Admin)"):
        if not user_to_reset.strip() or not new_pass.strip():
            st.warning("Fill both fields")
        else:
            hashed = hash_password(new_pass)
            c = conn.cursor()
            c.execute("UPDATE users SET password=? WHERE username=?", (hashed, user_to_reset))
            if c.rowcount == 0:
                st.error(f"User '{user_to_reset}' not found.")
            else:
                conn.commit()
                st.success(f"Password for '{user_to_reset}' has been reset.")
                log_activity(st.session_state["user"], f"Reset password for '{user_to_reset}'")

    st.divider()

    # Activity logs
    st.subheader("📋 Activity Logs")
    logs_df = pd.read_sql_query("SELECT * FROM activity_log ORDER BY id DESC LIMIT 500", conn)
    if logs_df.empty:
        st.info("No activity logs yet.")
    else:
        st.dataframe(logs_df, use_container_width=True)
        csv_logs = logs_df.to_csv(index=False).encode()
        st.download_button("📥 Download Logs (CSV)", csv_logs, "activity_logs.csv", "text/csv")

    conn.close()

def format_video_time(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# -------------------------------
# Pie Chart Helper
# -------------------------------
def render_pie(labels, sizes, title):
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.0f}%" if p >= 1 else "", startangle=90)
    ax.axis('equal')
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# -------------------------------
# Main Loop (video processing)
# -------------------------------
if start_btn:
    if source=="Upload Video":
        if uploaded_video is None:
            st.warning("Please upload a video first.")
            st.stop()
        # Copy the uploaded file to disk in chunks instead of making an additional
        # full-size bytes copy with uploaded_video.read().
        suffix = os.path.splitext(getattr(uploaded_video, "name", "video.mp4"))[1] or ".mp4"
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            uploaded_video.seek(0)
            chunk_size = 8 * 1024 * 1024  # 8 MB per disk-write chunk
            while True:
                chunk = uploaded_video.read(chunk_size)
                if not chunk:
                    break
                tfile.write(chunk)
            tfile.flush()
        except Exception:
            tfile.close()
            try:
                os.unlink(tfile.name)
            except OSError:
                pass
            raise
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error("Could not open video source.")
        st.stop()

    # Split the uploaded video's TIMELINE into consecutive reporting periods.
    interval_minutes = st.sidebar.number_input(
        "Reporting interval (minutes)", min_value=1, max_value=120, value=15, step=1
    )
    interval_seconds = int(interval_minutes * 60)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if not math.isfinite(video_fps) or video_fps <= 0:
        video_fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration = total_frames / video_fps if total_frames > 0 else None

    tracker = CentroidTracker(max_distance=max_distance, max_age=max_age)
    fps_time = time.time(); frame_idx = 0
    processing_fps = 0.0
    # Streamlit Cloud is resource constrained: avoid a browser/UI update on every frame.
    PREVIEW_EVERY = 12
    UI_UPDATE_EVERY = 120
    CHART_UPDATE_EVERY = 600
    processing_error = None
    progress_bar = st.progress(0, text="Preparing video processing...")
    processing_status = st.empty()

    direction_counts = {"left_to_right":0,"right_to_left":0,"up_to_down":0,"down_to_up":0}
    class_totals = {cls:0 for cls in selected_classes}
    events = []
    interval_data = {}
    unique_track_ids = set()
    history = {"frame":[], "East":[], "West":[], "South":[], "North":[]}
    interval_live_placeholder = st.empty()
    last_completed_segment = -1

    video_holder = st.empty()
    dashboard_placeholder = st.empty()
    sidebar_stats = st.sidebar.empty()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            video_seconds = (frame_idx - 1) / video_fps
            segment_index = int(video_seconds // interval_seconds)
            if segment_index not in interval_data:
                start_seconds = segment_index * interval_seconds
                interval_data[segment_index] = {
                    "Segment": segment_index + 1,
                    "Start": start_seconds,
                    "East": 0, "West": 0, "North": 0, "South": 0,
                    "Total Crossings": 0,
                    "Vehicle Classes": {}
                }
            interval_bucket = interval_data[segment_index]

            # At the start of each new 15-minute block, publish the block just completed.
            if segment_index > 0 and last_completed_segment < segment_index - 1:
                completed = interval_data.get(segment_index - 1)
                if completed is not None:
                    interval_live_placeholder.markdown(
                        f"""### ✅ Completed Segment {completed['Segment']}
**Video period:** {format_video_time(completed['Start'])} – {format_video_time(completed['Start'] + interval_seconds)}

| Direction | Crossings |
|---|---:|
| East | {completed['East']} |
| West | {completed['West']} |
| North | {completed['North']} |
| South | {completed['South']} |
| **Segment total** | **{completed['Total Crossings']}** |
"""
                    )
                last_completed_segment = segment_index - 1

            h,w = frame.shape[:2]
            h_line_y = int(h*h_ratio)
            v_line_x = int(w*v_ratio)
            use_h = line_mode in ("Horizontal & Vertical","Horizontal only")
            use_v = line_mode in ("Horizontal & Vertical","Vertical only")

            dets = detect_objects(frame, conf_thresh,nms_thresh,set(selected_classes),input_size)
            tracks = tracker.update(dets)

            # Draw lines
            if use_h:
                cv2.line(frame,(0,h_line_y),(w,h_line_y),(0,255,255),2)
            if use_v:
                cv2.line(frame,(v_line_x,0),(v_line_x,h),(255,255,0),2)

            # Update counts and draw boxes/traces
            for tid,(cx,cy,bw,bh,cname,conf) in tracks.items():
                tr = tracker.tracks.get(tid)
                if tr is None: continue
                if show_trace and len(tr.trace)>=2:
                    for i in range(1,len(tr.trace)):
                        cv2.line(frame, tr.trace[i-1], tr.trace[i], (200,200,200), 2)
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                if draw_boxes:
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (76, 175, 80), 2)
                label = f"{cname} {int(conf * 100)}%"
                if show_ids:
                    label = f"ID {tid} | " + label
                cv2.putText(frame, label, (x, max(10, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(10,220,10),2)

                # Crossing checks
                if len(tr.trace)>=2:
                    px,py = tr.trace[-2]
                    dx = cx - px
                    dy = cy - py
                    event_time = time.strftime("%H:%M:%S", time.localtime())
                    # Horizontal line crossings
                    if use_h and not tr.counted_crossings["h"]:
                        if (py < h_line_y <= cy) or (py > h_line_y >= cy):
                            if dy > 0:
                                direction_counts["up_to_down"] += 1
                                events.append((tid, "South", tr.cls or cname, frame_idx, event_time, segment_index + 1, format_video_time(video_seconds)))
                                interval_bucket["South"] += 1
                                interval_bucket["Total Crossings"] += 1
                                interval_bucket["Vehicle Classes"][tr.cls or cname] = interval_bucket["Vehicle Classes"].get(tr.cls or cname, 0) + 1
                                unique_track_ids.add(tid)
                            else:
                                direction_counts["down_to_up"] += 1
                                events.append((tid, "North", tr.cls or cname, frame_idx, event_time, segment_index + 1, format_video_time(video_seconds)))
                                interval_bucket["North"] += 1
                                interval_bucket["Total Crossings"] += 1
                                interval_bucket["Vehicle Classes"][tr.cls or cname] = interval_bucket["Vehicle Classes"].get(tr.cls or cname, 0) + 1
                                unique_track_ids.add(tid)
                            cls_name = tr.cls or cname
                            if cls_name in class_totals:
                                class_totals[cls_name] += 1
                            else:
                                class_totals[cls_name] = 1
                            tr.counted_crossings["h"] = True
                    # Vertical line crossings
                    if use_v and not tr.counted_crossings["v"]:
                        if (px < v_line_x <= cx) or (px > v_line_x >= cx):
                            if dx > 0:
                                direction_counts["left_to_right"] += 1
                                events.append((tid, "East", tr.cls or cname, frame_idx, event_time, segment_index + 1, format_video_time(video_seconds)))
                                interval_bucket["East"] += 1
                                interval_bucket["Total Crossings"] += 1
                                interval_bucket["Vehicle Classes"][tr.cls or cname] = interval_bucket["Vehicle Classes"].get(tr.cls or cname, 0) + 1
                                unique_track_ids.add(tid)
                            else:
                                direction_counts["right_to_left"] += 1
                                events.append((tid, "West", tr.cls or cname, frame_idx, event_time, segment_index + 1, format_video_time(video_seconds)))
                                interval_bucket["West"] += 1
                                interval_bucket["Total Crossings"] += 1
                                interval_bucket["Vehicle Classes"][tr.cls or cname] = interval_bucket["Vehicle Classes"].get(tr.cls or cname, 0) + 1
                                unique_track_ids.add(tid)
                            cls_name = tr.cls or cname
                            if cls_name in class_totals:
                                class_totals[cls_name] += 1
                            else:
                                class_totals[cls_name] = 1
                            tr.counted_crossings["v"] = True

            # History update
            if frame_idx % 10 == 0:
                history["frame"].append(frame_idx)
                history["East"].append(direction_counts["left_to_right"])
                history["West"].append(direction_counts["right_to_left"])
                history["South"].append(direction_counts["up_to_down"])
                history["North"].append(direction_counts["down_to_up"])

            # Update the preview and Streamlit widgets periodically, not on every frame.
            now = time.time()
            processing_fps = 1.0 / max(1e-6, now - fps_time)
            fps_time = now
            if fps_display:
                cv2.putText(frame, f"Processing FPS: {processing_fps:.1f}", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,180,255), 2)

            if frame_idx == 1 or frame_idx % PREVIEW_EVERY == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_holder.image(frame_rgb, channels="RGB", use_container_width=True)

            if frame_idx == 1 or frame_idx % UI_UPDATE_EVERY == 0:
                total_moves = sum(direction_counts.values())
                sidebar_stats.markdown(
                    f"### Live Totals\n"
                    f"- **Total crossings:** {total_moves}\n"
                    f"- **East (L→R):** {direction_counts['left_to_right']}\n"
                    f"- **West (R→L):** {direction_counts['right_to_left']}\n"
                    f"- **South (U→D):** {direction_counts['up_to_down']}\n"
                    f"- **North (D→U):** {direction_counts['down_to_up']}"
                )
                if total_frames > 0:
                    pct = min(100, int(frame_idx * 100 / total_frames))
                    progress_bar.progress(pct, text=f"Processed {frame_idx:,} of {total_frames:,} frames ({pct}%)")
                processing_status.caption(
                    f"Frame {frame_idx:,}" + (f" / {total_frames:,}" if total_frames else "") +
                    f" · Video time {format_video_time(video_seconds)} · {processing_fps:.1f} processing FPS"
                )

            # Expensive chart rendering is deliberately much less frequent.
            if frame_idx == 1 or frame_idx % CHART_UPDATE_EVERY == 0:
                with dashboard_placeholder.container():
                    st.subheader("📊 Live Movement Dashboard")
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("➡️ East", direction_counts["left_to_right"])
                    c2.metric("⬅️ West", direction_counts["right_to_left"])
                    c3.metric("⬇️ South", direction_counts["up_to_down"])
                    c4.metric("⬆️ North", direction_counts["down_to_up"])
                    if dashboard_view in ("Bar View", "Combined View"):
                        a,b = st.columns(2)
                        with a:
                            st.bar_chart(pd.DataFrame.from_dict({"East":direction_counts["left_to_right"],
                                "West":direction_counts["right_to_left"], "South":direction_counts["up_to_down"],
                                "North":direction_counts["down_to_up"]}, orient="index", columns=["Count"]))
                        with b:
                            st.bar_chart(pd.DataFrame.from_dict(class_totals, orient="index", columns=["Count"]))
                    if dashboard_view in ("Line View", "Combined View") and len(history["frame"]) > 1:
                        st.line_chart(pd.DataFrame(history).set_index("frame"))
                    if show_pies and dashboard_view == "Combined View" and sum(direction_counts.values()) > 0:
                        render_pie(["East","West","South","North"],
                            [direction_counts["left_to_right"], direction_counts["right_to_left"],
                             direction_counts["up_to_down"], direction_counts["down_to_up"]], "Direction %")

    except Exception as e:
        processing_error = e
        st.error("Video processing stopped unexpectedly. Partial results are being prepared below.")
        st.exception(e)
    finally:
        cap.release()
        # remove temp file if used
        try:
            if source == "Upload Video" and 'tfile' in locals():
                tfile.close()
                os.unlink(tfile.name)
        except Exception:
            pass

    if processing_error is None:
        progress_bar.progress(100, text="Processing finished")
        st.success("✅ Finished Processing")
    else:
        st.warning(f"Partial results: processing stopped at frame {frame_idx:,}. You can still download the results collected so far.")

    # On an interrupted run, report only the portion actually processed.
    processed_duration = (frame_idx / video_fps) if frame_idx else 0.0
    report_duration = video_duration if processing_error is None and video_duration else processed_duration
    # Complete 15-minute interval report, including empty intervals in the processed range.
    number_of_segments = max(1, int(math.ceil(report_duration / interval_seconds)))

    interval_rows = []
    for seg in range(number_of_segments):
        item = interval_data.get(seg, {
            "Segment": seg + 1, "Start": seg * interval_seconds,
            "East": 0, "West": 0, "North": 0, "South": 0,
            "Total Crossings": 0, "Vehicle Classes": {}
        })
        end_seconds = min((seg + 1) * interval_seconds, report_duration)
        interval_rows.append({
            "Segment": seg + 1,
            "Video Period": f"{format_video_time(seg * interval_seconds)} - {format_video_time(end_seconds)}",
            "East": item["East"], "West": item["West"],
            "North": item["North"], "South": item["South"],
            "Total Crossings": item["Total Crossings"],
            "Vehicle Classes": json.dumps(item["Vehicle Classes"])
        })

    interval_df = pd.DataFrame(interval_rows)
    st.subheader("⏱️ Results for Each 15-Minute Period")
    st.dataframe(interval_df, use_container_width=True, hide_index=True)

    grand_total = sum(direction_counts.values())
    st.subheader("🏁 Grand Total — Entire Video")
    a, b, c, d, e = st.columns(5)
    a.metric("Grand Total", grand_total)
    b.metric("East", direction_counts["left_to_right"])
    c.metric("West", direction_counts["right_to_left"])
    d.metric("North", direction_counts["down_to_up"])
    e.metric("South", direction_counts["up_to_down"])

    st.subheader("📊 Grand Total by Direction")
    direction_df = pd.DataFrame({
        "Direction": ["East", "West", "North", "South"],
        "Crossings": [direction_counts["left_to_right"], direction_counts["right_to_left"],
                      direction_counts["down_to_up"], direction_counts["up_to_down"]]
    })
    st.bar_chart(direction_df.set_index("Direction"))

    st.subheader("🚘 Vehicle Class Totals")
    st.dataframe(pd.DataFrame(
        [{"Vehicle Class": k, "Total Crossings": v} for k, v in class_totals.items()]
    ), use_container_width=True, hide_index=True)

    if events:
        df = pd.DataFrame(events, columns=[
            "track_id", "direction", "class", "frame", "timestamp", "segment", "video_time"
        ])
        st.subheader("🧾 Detailed Crossing Records")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download Detailed Events (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "vehicle_crossing_events.csv", "text/csv"
        )
    else:
        df = pd.DataFrame(columns=["track_id", "direction", "class", "frame", "timestamp", "segment", "video_time"])
        st.info("No crossing events detected.")

    st.download_button(
        "📥 Download 15-Minute Summary (CSV)",
        interval_df.to_csv(index=False).encode("utf-8"),
        "vehicle_15_minute_summary.csv", "text/csv"
    )

    try:
        import io
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            interval_df.to_excel(writer, sheet_name="15-Minute Summary", index=False)
            direction_df.to_excel(writer, sheet_name="Grand Totals", index=False)
            df.to_excel(writer, sheet_name="Crossing Events", index=False)
        st.download_button(
            "📥 Download Complete Excel Report", excel_buffer.getvalue(),
            "intersection_vehicle_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as export_error:
        st.warning(f"Excel export needs openpyxl in requirements.txt: {export_error}")

    # PDF summary export using Matplotlib (no extra PDF package required).
    try:
        import io
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_buffer = io.BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            ax.axis("off")
            ax.set_title("Intersection Vehicle Counter - Grand Total Report", fontsize=16, pad=20)
            total_table = [
                ["Direction", "Crossings"],
                ["East", direction_counts["left_to_right"]],
                ["West", direction_counts["right_to_left"]],
                ["North", direction_counts["down_to_up"]],
                ["South", direction_counts["up_to_down"]],
                ["GRAND TOTAL", grand_total],
            ]
            table = ax.table(cellText=total_table[1:], colLabels=total_table[0],
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.7)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Break the interval report into readable pages.
            rows_per_page = 22
            for start_row in range(0, len(interval_df), rows_per_page):
                page_df = interval_df.iloc[start_row:start_row + rows_per_page]
                fig, ax = plt.subplots(figsize=(11.7, 8.3))
                ax.axis("off")
                ax.set_title("15-Minute Interval Results", fontsize=15, pad=18)
                table = ax.table(cellText=page_df.values, colLabels=page_df.columns,
                                 loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.4)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        st.download_button(
            "📄 Download PDF Summary Report", pdf_buffer.getvalue(),
            "intersection_vehicle_report.pdf", "application/pdf"
        )
    except Exception as pdf_error:
        st.warning(f"PDF export could not be created: {pdf_error}")

    if processing_error is not None:
        st.info("These files contain only the crossings detected before processing stopped. Re-run the video to obtain a complete report.")

    st.caption(
        "Important: East/West use left/right screen movement; North/South use up/down "
        "screen movement. Set the camera orientation and counting lines correctly. "
        "Grand Total is crossing events: a vehicle that crosses both lines may be counted "
        "once on each axis."
    )

else:
    st.info("Upload a video or select webcam, then click **Start**.")
    st.caption("Streamlit Community Cloud has limited CPU and memory. Very long videos may take substantially longer than their playback duration; for 30–60 minute videos, use a lower-resolution copy or process shorter sections if the service restarts.")

# ===============================
# Render admin UI if selected
# ===============================
# If admin nav was created, show the selected admin page
if st.session_state.get("is_admin"):
    # show admin dashboard UI below the main app or in a modal area
    st.markdown("---")
    st.header("Admin Controls")
    if admin_page_nav == "Admin Dashboard":
        admin_dashboard()
    elif admin_page_nav == "User Manager":
        admin_dashboard()  # same content for now
    elif admin_page_nav == "Activity Logs":
        admin_dashboard()  # same content for now

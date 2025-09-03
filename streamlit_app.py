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
import json

# ===============================
# User Authentication Setup
# ===============================
USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None

# ===============================
# Authentication Pages
# ===============================
def login_page():
    st.subheader("🔐 Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        users = load_users()
        if username and username in users and users[username] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.success(f"Welcome back, {username}!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def register_page():
    st.subheader("🆕 Register")
    username = st.text_input("Choose a username", key="reg_username")
    password = st.text_input("Choose a password", type="password", key="reg_password")
    confirm = st.text_input("Confirm password", type="password", key="reg_confirm")
    if st.button("Register"):
        users = load_users()
        if not username:
            st.warning("Please choose a username")
        elif username in users:
            st.warning("Username already exists")
        elif password != confirm:
            st.warning("Passwords do not match")
        elif len(password) < 4:
            st.warning("Password must be at least 4 characters")
        else:
            users[username] = hash_password(password)
            save_users(users)
            st.success("Account created! You can now log in.")

def reset_password_page():
    st.subheader("🔄 Reset Password")
    username = st.text_input("Username", key="reset_username")
    new_password = st.text_input("New password", type="password", key="reset_new")
    confirm = st.text_input("Confirm new password", type="password", key="reset_confirm")
    if st.button("Reset"):
        users = load_users()
        if not username:
            st.error("Please enter your username")
        elif username not in users:
            st.error("Username not found")
        elif new_password != confirm:
            st.error("Passwords do not match")
        else:
            users[username] = hash_password(new_password)
            save_users(users)
            st.success("Password updated successfully! Please log in.")

def logout_button():
    if st.sidebar.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["user"] = None
        st.experimental_rerun()

# ===============================
# App setup & style
# ===============================
st.set_page_config(page_title="🚗 Intersection Vehicle Counter", layout="wide")
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

# ===============================
# 🚗 Main Vehicle Counter App
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

net = cv2.dnn.readNetFromDarknet(FILES["cfg"], FILES["weights"])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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
    uploaded_video = st.file_uploader("Upload a video", type=["mp4","mov","avi","mkv"])
else:
    cam_index = st.number_input("Webcam index", value=0, step=1, min_value=0)
start_btn = st.button("▶️ Start")

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
# Main Loop
# -------------------------------
if start_btn:
    if source=="Upload Video":
        if uploaded_video is None:
            st.warning("Please upload a video first.")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        st.error("Could not open video source.")
        st.stop()

    tracker = CentroidTracker(max_distance=max_distance, max_age=max_age)
    fps_time = time.time(); frame_idx = 0

    direction_counts = {"left_to_right":0,"right_to_left":0,"up_to_down":0,"down_to_up":0}
    class_totals = {cls:0 for cls in selected_classes}
    events = []
    history = {"frame":[], "East":[], "West":[], "South":[], "North":[]}

    video_holder = st.empty()
    dashboard_placeholder = st.empty()
    sidebar_stats = st.sidebar.empty()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
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
                if draw_boxes:
                    x=int(cx-bw/2); y=int(cy-bh/2)
                    cv2.rectangle(frame,(x,y),(x+bw,y+bh),(76,175,80),2)
                label=f"{cname} {int(conf*100)}%"
                if show_ids: label=f"ID {tid} | "+label
                cv2.putText(frame,label,(int(cx-bw/2),max(10,int(y-8))),
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
                                events.append((tid, "South", tr.cls or cname, frame_idx, event_time))
                            else:
                                direction_counts["down_to_up"] += 1
                                events.append((tid, "North", tr.cls or cname, frame_idx, event_time))
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
                                events.append((tid, "East", tr.cls or cname, frame_idx, event_time))
                            else:
                                direction_counts["right_to_left"] += 1
                                events.append((tid, "West", tr.cls or cname, frame_idx, event_time))
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

            # FPS
            if fps_display:
                now = time.time()
                fps = 1.0 / max(1e-6, now - fps_time)
                fps_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,180,255), 2)

            # Show frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_holder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Sidebar stats
            total_moves = sum(direction_counts.values())
            sidebar_stats.markdown(
                f"""
                ### Live Totals
                - **Total crossings:** {total_moves}
                - **East (L→R):** {direction_counts['left_to_right']}
                - **West (R→L):** {direction_counts['right_to_left']}
                - **South (U→D):** {direction_counts['up_to_down']}
                - **North (D→U):** {direction_counts['down_to_up']}
                """
            )

            # Dashboard
            with dashboard_placeholder.container():
                st.subheader("📊 Live Movement Dashboard")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("➡️ East", direction_counts["left_to_right"])
                c2.metric("⬅️ West", direction_counts["right_to_left"])
                c3.metric("⬇️ South", direction_counts["up_to_down"])
                c4.metric("⬆️ North", direction_counts["down_to_up"])

                if dashboard_view == "Bar View":
                    a,b = st.columns(2)
                    with a:
                        st.bar_chart(pd.DataFrame.from_dict({"East":direction_counts["left_to_right"],
                                                             "West":direction_counts["right_to_left"],
                                                             "South":direction_counts["up_to_down"],
                                                             "North":direction_counts["down_to_up"]},
                                                            orient="index", columns=["Count"]))
                    with b:
                        if class_totals:
                            st.bar_chart(pd.DataFrame.from_dict(class_totals, orient="index", columns=["Count"]))
                        else:
                            st.write("No class counts yet.")
                elif dashboard_view == "Line View":
                    if len(history["frame"]) > 1:
                        df_hist = pd.DataFrame(history).set_index("frame")
                        st.line_chart(df_hist)
                    else:
                        st.info("Play longer to collect history for line chart.")
                elif dashboard_view == "Combined View":
                    a,b = st.columns(2)
                    with a:
                        st.bar_chart(pd.DataFrame.from_dict({"East":direction_counts["left_to_right"],
                                                             "West":direction_counts["right_to_left"],
                                                             "South":direction_counts["up_to_down"],
                                                             "North":direction_counts["down_to_up"]},
                                                            orient="index", columns=["Count"]))
                        if len(history["frame"]) > 1:
                            st.line_chart(pd.DataFrame(history).set_index("frame"))
                    with b:
                        if class_totals:
                            st.bar_chart(pd.DataFrame.from_dict(class_totals, orient="index", columns=["Count"]))
                        else:
                            st.write("No class counts yet.")
                        if show_pies:
                            total = sum(direction_counts.values())
                            if total > 0:
                                render_pie(["East","West","South","North"],
                                           [direction_counts["left_to_right"],
                                            direction_counts["right_to_left"],
                                            direction_counts["up_to_down"],
                                            direction_counts["down_to_up"]],
                                           "Direction %")
                            if sum(class_totals.values()) > 0:
                                render_pie(list(class_totals.keys()), list(class_totals.values()), "Class %")

    finally:
        cap.release()
        # remove temp file if used
        try:
            if source == "Upload Video" and 'tfile' in locals():
                tfile.close()
                os.unlink(tfile.name)
        except Exception:
            pass

    st.success("✅ Finished Processing")

    # Summary & CSV
    st.metric("Grand Total Crossings", sum(direction_counts.values()))
    if events:
        df = pd.DataFrame(events, columns=["track_id","direction","class","frame","timestamp"])
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, "counts.csv", "text/csv")
    else:
        st.info("No crossing events detected.")

else:
    st.info("Upload a video or select webcam, then click **Start**.")

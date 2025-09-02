import os
import cv2
import time
import math
import urllib.request
import tempfile
import numpy as np
import streamlit as st
import pandas as pd
from collections import deque, defaultdict

###############################################################################
# App setup & style
###############################################################################
st.set_page_config(page_title="🚗 Vehicle Counter", layout="wide")
st.set_option("server.maxUploadSize", 1000)  # allow up to 1GB uploads

# === Custom Background Styling ===
page_bg = """
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #f4f6f9; /* clean light background */
    background-size: cover;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1e1e2f; /* dark sidebar */
}

/* Force ALL text in sidebar to white */
[data-testid="stSidebar"], 
[data-testid="stSidebar"] * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

/* Transparent header */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background-color: #2563eb;   /* blue button */
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.2em;
    font-weight: 600;
    cursor: pointer;
    transition: 0.3s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #1e40af;   /* darker blue on hover */
    color: #f1f5f9 !important;
}

/* === Sliders === */
.stSlider > div > div > div[data-testid="stTickBar"] {
    background: #374151; /* dark gray track */
}
.stSlider > div > div > div > div[data-testid="stThumbValue"] {
    color: #ffffff !important; /* white value label */
}
.stSlider > div > div > div > div[role="slider"] {
    background-color: #2563eb; /* blue knob */
    border: 2px solid #1e40af;
}
.stSlider > div > div > div > div[role="slider"]:hover {
    background-color: #1e40af; /* darker blue on hover */
}

/* === Radio buttons & checkboxes === */
.stRadio div[role="radiogroup"] > label > div:first-child,
.stCheckbox > label > div:first-child {
    border: 2px solid #2563eb !important;   /* blue border */
    background-color: #1e1e2f !important;   /* match sidebar */
}
.stRadio div[role="radiogroup"] > label > div[aria-checked="true"],
.stCheckbox > label > div[aria-checked="true"] {
    background-color: #2563eb !important;   /* filled blue when active */
    border: 2px solid #1e40af !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🚗 Intersection Vehicle Counter")

###############################################################################
# Auto-download YOLOv4-tiny (weights/cfg) + COCO labels on first run
###############################################################################
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

###############################################################################
# Load classes
###############################################################################
with open(FILES["names"], "r") as f:
    CLASSES = [c.strip() for c in f.readlines()]

VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle"}

###############################################################################
# Load YOLO network
###############################################################################
net = cv2.dnn.readNetFromDarknet(FILES["cfg"], FILES["weights"])
if net.empty():
    st.error("Failed to load YOLOv4-tiny network. Check weights/cfg files.")
    st.stop()

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

###############################################################################
# Tracker
###############################################################################
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

###############################################################################
# Vehicle detection
###############################################################################
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

###############################################################################
# Streamlit UI
###############################################################################
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio("Source", ["Upload Video", "Local File", "Webcam"], index=0)
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
local_video_path = None
cap = None

if source == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video (≤1GB)", type=["mp4", "mov", "avi", "mkv"])
elif source == "Local File":
    local_file = st.file_uploader("Pick a local video file", type=["mp4", "mov", "avi", "mkv"])
    if local_file is not None:
        local_video_path = local_file.name
        st.success(f"Selected: {local_video_path}")
else:
    cam_index = st.number_input("Webcam index", value=0, step=1, min_value=0)

start_btn = st.button("▶️ Start")

direction_counts = {"left_to_right":0, "right_to_left":0, "up_to_down":0, "down_to_up":0}
class_totals = {cls: 0 for cls in selected_classes}
events = []  # store all events

# === Sidebar live stats placeholders ===
stats_placeholder = st.sidebar.empty()
direction_placeholder = st.sidebar.empty()

if start_btn:
    if source == "Upload Video":
        if uploaded_video is None:
            st.warning("Please upload a video first.")
            st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

    elif source == "Local File":
        if not local_video_path or not os.path.exists(local_video_path):
            st.warning("Please provide a valid local video file.")
            st.stop()
        cap = cv2.VideoCapture(local_video_path)

    else:  # Webcam
        cap = cv2.VideoCapture(int(cam_index))

    if not cap.isOpened():
        st.error("Could not open video source.")
        st.stop()

    tracker = CentroidTracker(max_distance=max_distance, max_age=max_age)
    frame_holder = st.empty()
    fps_time = time.time()
    frame_idx = 0

    # safer loop
    while cap.isOpened():
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

            # Check crossings
            if len(tr.trace) >= 2:
                px, py = tr.trace[-2]
                dx = cx - px
                dy = cy - py
                event_time = time.strftime("%H:%M:%S", time.localtime())

                if use_h and not tr.counted_crossings["h"]:
                    if (py < h_line_y <= cy) or (py > h_line_y >= cy):
                        if dy > 0:
                            direction_counts["up_to_down"] += 1
                            events.append((tid, "up_to_down", tr.cls, frame_idx, event_time))
                        else:
                            direction_counts["down_to_up"] += 1
                            events.append((tid, "down_to_up", tr.cls, frame_idx, event_time))
                        class_totals[tr.cls] += 1
                        tr.counted_crossings["h"] = True

                if use_v and not tr.counted_crossings["v"]:
                    if (px < v_line_x <= cx) or (px > v_line_x >= cx):
                        if dx > 0:
                            direction_counts["left_to_right"] += 1
                            events.append((tid, "left_to_right", tr.cls, frame_idx, event_time))
                        else:
                            direction_counts["right_to_left"] += 1
                            events.append((tid, "right_to_left", tr.cls, frame_idx, event_time))
                        class_totals[tr.cls] += 1
                        tr.counted_crossings["v"] = True

        # === Overlay totals on video ===
        overlay_lines = []
        overlay_lines.append(" | ".join([f"{cls.capitalize()}: {cnt}" for cls, cnt in class_totals.items()]))
        overlay_lines.append(f"Total: {sum(class_totals.values())}")
        overlay_lines.append(
            f"L→R: {direction_counts['left_to_right']} | R→L: {direction_counts['right_to_left']}"
        )
        overlay_lines.append(
            f"U→D: {direction_counts['up_to_down']} | D→U: {direction_counts['down_to_up']}"
        )

        y0 = 40
        for i, line in enumerate(overlay_lines):
            y = y0 + i * 25
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)

        # === Live sidebar stats update ===
        stats_placeholder.write("### 🚘 Vehicle Class Counts")
        stats_placeholder.write(pd.DataFrame(list(class_totals.items()), columns=["Class", "Count"]))

        direction_placeholder.write("### 🧭 Direction Counts")
        direction_placeholder.write(pd.DataFrame([
            ["Left → Right", direction_counts["left_to_right"]],
            ["Right → Left", direction_counts["right_to_left"]],
            ["Up → Down", direction_counts["up_to_down"]],
            ["Down → Up", direction_counts["down_to_up"]],
        ], columns=["Direction", "Count"]))

        if fps_display:
            now = time.time()
            fps = 1.0 / (now - fps_time)
            fps_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        frame_holder.image(frame, channels="BGR")

    cap.release()
    st.success("Video processing completed.")

    df_events = pd.DataFrame(events, columns=["Track ID", "Direction", "Class", "Frame", "Time"])
    st.subheader("📊 All Events")
    st.dataframe(df_events)

    # download CSV
    csv = df_events.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", csv, "vehicle_events.csv", "text/csv")

    # summary table
    st.subheader("📈 Summary Totals")
    summary_data = []
    for cls, cnt in class_totals.items():
        summary_data.append([cls, cnt])
    summary_df = pd.DataFrame(summary_data, columns=["Class", "Total"])
    st.table(summary_df)

    st.subheader("🧭 Direction Totals")
    dir_df = pd.DataFrame([
        ["Left → Right", direction_counts["left_to_right"]],
        ["Right → Left", direction_counts["right_to_left"]],
        ["Up → Down", direction_counts["up_to_down"]],
        ["Down → Up", direction_counts["down_to_up"]],
    ], columns=["Direction", "Total"])
    st.table(dir_df)

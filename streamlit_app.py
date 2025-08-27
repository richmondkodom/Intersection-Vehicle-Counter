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
def detect_vehicles(frame, conf_thresh=0.3, nms_thresh=0.4, target_classes=None, input_size=416):
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
st.set_page_config(page_title="Vehicle Counter", layout="wide")
st.title("🚗 Intersection Vehicle Counter")

with st.sidebar:
    st.header("Settings")
    source = st.radio("Source", ["Upload Video", "Webcam"], index=0)
    conf_thresh = st.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
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

    st.markdown("---")
    st.caption("Tip: use a 720p clip for best live performance on CPU.")

uploaded_video = None
cap = None

if source == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
else:
    cam_index = st.number_input("Webcam index", value=0, step=1, min_value=0)

start_btn = st.button("▶️ Start")

direction_counts = {"left_to_right":0, "right_to_left":0, "up_to_down":0, "down_to_up":0}
class_totals = {cls: 0 for cls in selected_classes}
events = []  # store all events

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
    frame_holder = st.empty()
    stats_col1, stats_col2 = st.columns(2)
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

        # === Debug overlay: show number of detections YOLO made ===
        cv2.putText(frame, f"Detections: {len(dets)}", (10,90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

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
            cv2.putText(frame, label, (int(cx - bw/2), int(max(0,y-8))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,220,10), 2)

            # Check crossings
            if len(tr.trace) >= 2:
                px, py = tr.trace[-2]
                dx = cx - px
                dy = cy - py
                event_time = time.strftime("%H:%M:%S", time.localtime())

                # Horizontal line
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

                # Vertical line
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

        if fps_display:
            now = time.time()
            fps = 1.0 / max(1e-6, now - fps_time)
            fps_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,180,255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb, channels="RGB")

        # Update stats (filter zeros)
        stats_col1.write("### Directions")
        df_dir = pd.DataFrame(
            [(k, v) for k, v in direction_counts.items() if v > 0],
            columns=["Direction", "Count"]
        )
        if not df_dir.empty:
            stats_col1.dataframe(df_dir, use_container_width=True)
        else:
            stats_col1.info("No crossings yet.")

        stats_col2.write("### By Vehicle Class")
        if class_totals:
            df_classes = (
                pd.DataFrame.from_dict(class_totals, orient="index", columns=["Count"])
                .reset_index()
                .rename(columns={"index": "Class"})
            )
            df_classes = df_classes[df_classes["Count"] > 0]
            if not df_classes.empty:
                df_classes = df_classes.sort_values(by="Count", ascending=False)
                stats_col2.dataframe(df_classes, use_container_width=True)
            else:
                stats_col2.info("No vehicles counted yet.")
        else:
            stats_col2.info("No vehicles yet.")

    cap.release()
    st.success("Finished.")
    total = sum(direction_counts.values())
    st.metric("Grand Total", total)

    if events:
        df = pd.DataFrame(events, columns=["track_id","direction","class","frame","timestamp"])
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Log (CSV)", csv, file_name="vehicle_counts.csv", mime="text/csv")

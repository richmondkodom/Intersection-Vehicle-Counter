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

# 🔎 Debug: confirm classes
st.write("Loaded classes:", CLASSES[:15])

###############################################################################
# Load YOLO network
###############################################################################
net = cv2.dnn.readNetFromDarknet(FILES["cfg"], FILES["weights"])
if net.empty():
    st.error("Failed to load YOLOv4-tiny network. Check weights/cfg files.")
    st.stop()

# Force CPU backend (no CUDA needed)
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
st.title("🚗 Intersection Vehicle Counter (Debug Mode)")

with st.sidebar:
    st.header("Settings")
    source = st.radio("Source", ["Upload Video", "Webcam"], index=0)
    conf_thresh = st.slider("Detection confidence", 0.1, 0.9, 0.25, 0.05)
    nms_thresh = st.slider("NMS threshold", 0.1, 0.9, 0.45, 0.05)
    input_size = st.select_slider("Model input size", options=[320, 416, 512, 608], value=416)

    st.markdown("⚠️ Debug mode: detects ALL classes, not just vehicles.")

uploaded_video = None
cap = None

if source == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
else:
    cam_index = st.number_input("Webcam index", value=0, step=1, min_value=0)

start_btn = st.button("▶️ Start")

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

    frame_holder = st.empty()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # detect ALL classes (not filtering)
        dets = detect_vehicles(frame, conf_thresh, nms_thresh, None, input_size)

        # 🔎 Debug: show raw detections
        if dets:
            st.write([f"{cname} {conf:.2f}" for (_, _, _, _, cname, conf) in dets])
        else:
            st.write(f"Frame {frame_idx}: No detections")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Finished.")

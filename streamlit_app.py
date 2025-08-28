import os
import cv2
import time
import math
import urllib.request
import tempfile
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import deque

# ============================================================================
# App setup & style
# ============================================================================
st.set_page_config(page_title="🚦 Vehicle Counter Dashboard", layout="wide")
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #141e30, #243b55);
            color: #fff;
        }
        section[data-testid="stSidebar"] {
            background: #1e1e1e;
        }
        h1, h2, h3 {
            color: #f5f5f5 !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 15px;
            margin: 8px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚦 Smart Vehicle Counter Dashboard")

# ============================================================================
# Model files
# ============================================================================
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
    st.error("Failed to load YOLOv4-tiny network")
    st.stop()
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# ============================================================================
# Tracker
# ============================================================================
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
        assigned, out = set(), {}
        for det in detections:
            dcx, dcy, w, h, cname, conf = det
            best_id, best_dist = None, 1e9
            for tid, tr in self.tracks.items():
                if tid in assigned: continue
                dist = self._euclidean((dcx, dcy), tr.trace[-1])
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None and best_dist <= self.max_distance:
                tr = self.tracks[best_id]
                tr.trace.append((dcx, dcy))
                tr.last_seen = now
                if tr.cls is None: tr.cls = cname
                assigned.add(best_id)
                out[best_id] = (dcx, dcy, w, h, tr.cls or cname, conf)
            else:
                tid = self.next_id; self.next_id += 1
                tr = Track(tid, (dcx, dcy)); tr.cls = cname; tr.last_seen = now
                self.tracks[tid] = tr; assigned.add(tid)
                out[tid] = (dcx, dcy, w, h, cname, conf)
        return out

# ============================================================================
# Detection
# ============================================================================
def detect_vehicles(frame, conf_thresh=0.25, nms_thresh=0.4, target_classes=None, input_size=416):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    boxes, confs, class_ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_thresh:
                cx, cy = int(det[0] * w), int(det[1] * h)
                bw, bh = int(det[2] * w), int(det[3] * h)
                x, y = int(cx - bw / 2), int(cy - bh / 2)
                cname = CLASSES[class_id] if class_id < len(CLASSES) else str(class_id)
                if target_classes and cname not in target_classes: continue
                boxes.append([x, y, bw, bh]); confs.append(confidence); class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, nms_thresh)
    detections = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, bw, bh = boxes[i]
            cx, cy = x + bw // 2, y + bh // 2
            cname = CLASSES[class_ids[i]] if class_ids[i] < len(CLASSES) else str(class_ids[i])
            detections.append((cx, cy, bw, bh, cname, confs[i]))
    return detections

# ============================================================================
# Sidebar Settings
# ============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    source = st.radio("Video Source", ["Upload Video", "Webcam"], 0)
    conf_thresh = st.slider("Confidence", 0.1, 0.9, 0.25, 0.05)
    nms_thresh = st.slider("NMS threshold", 0.1, 0.9, 0.45, 0.05)
    input_size = st.select_slider("Model input", [320, 416, 512, 608], 416)
    max_distance = st.slider("Tracker distance", 10, 150, 60, 5)
    max_age = st.slider("Tracker age (s)", 1.0, 5.0, 2.0, 0.5)
    line_mode = st.selectbox("Count lines", ["Horizontal & Vertical", "Horizontal only", "Vertical only"], 0)
    h_ratio = st.slider("Horizontal line", 0.1, 0.9, 0.5, 0.05)
    v_ratio = st.slider("Vertical line", 0.1, 0.9, 0.5, 0.05)
    selected_classes = st.multiselect("Detect classes", sorted(list(VEHICLE_CLASSES)), list(VEHICLE_CLASSES))
    draw_boxes = st.checkbox("Draw boxes", True)
    show_ids = st.checkbox("Show track IDs", True)
    show_trace = st.checkbox("Draw trails", True)
    fps_display = st.checkbox("Show FPS", True)

start_btn = st.button("▶️ Start Counting", use_container_width=True)

# ============================================================================
# Dashboard Placeholders
# ============================================================================
col1, col2, col3, col4 = st.columns(4)
with col1: m1 = st.metric("➡️ Left → Right", 0)
with col2: m2 = st.metric("⬅️ Right → Left", 0)
with col3: m3 = st.metric("⬇️ Up → Down", 0)
with col4: m4 = st.metric("⬆️ Down → Up", 0)

st.markdown("### 🎥 Live Video Feed")
frame_holder = st.empty()

c1, c2 = st.columns([2, 1])
with c1: class_chart = st.empty()
with c2: direction_chart = st.empty()
log_table = st.empty()

# ============================================================================
# Main loop
# ============================================================================
if start_btn:
    # Video source
    if source == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
        if not uploaded_video: st.stop()
        tfile = tempfile.NamedTemporaryFile(delete=False); tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened(): st.error("Cannot open video"); st.stop()

    tracker = CentroidTracker(max_distance, max_age)
    direction_counts = {"left_to_right":0, "right_to_left":0, "up_to_down":0, "down_to_up":0}
    class_totals = {cls: 0 for cls in selected_classes}
    events, fps_time, frame_idx = [], time.time(), 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1; h, w = frame.shape[:2]
        h_line_y, v_line_x = int(h*h_ratio), int(w*v_ratio)
        use_h = line_mode in ("Horizontal & Vertical","Horizontal only")
        use_v = line_mode in ("Horizontal & Vertical","Vertical only")

        dets = detect_vehicles(frame, conf_thresh, nms_thresh, set(selected_classes), input_size)
        tracks = tracker.update(dets)

        if use_h: cv2.line(frame,(0,h_line_y),(w,h_line_y),(0,255,255),2)
        if use_v: cv2.line(frame,(v_line_x,0),(v_line_x,h),(255,255,0),2)

        for tid,(cx,cy,bw,bh,cname,conf) in tracks.items():
            tr = tracker.tracks[tid]
            if show_trace and len(tr.trace)>=2:
                for i in range(1,len(tr.trace)):
                    cv2.line(frame,tr.trace[i-1],tr.trace[i],(200,200,200),2)
            if draw_boxes:
                x,y=int(cx-bw/2),int(cy-bh/2)
                cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
            label=f"{cname} {int(conf*100)}%"; 
            if show_ids: label=f"ID {tid}|"+label
            cv2.putText(frame,label,(x,max(15,y-8)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(10,220,10),2)

            if len(tr.trace)>=2:
                px,py=tr.trace[-2]; dx,dy=cx-px,cy-py
                event_time=time.strftime("%H:%M:%S",time.localtime())
                if use_h and not tr.counted_crossings["h"]:
                    if (py<h_line_y<=cy) or (py>h_line_y>=cy):
                        if dy>0: direction="up_to_down"; direction_counts["up_to_down"]+=1
                        else: direction="down_to_up"; direction_counts["down_to_up"]+=1
                        class_totals[cname]+=1; tr.counted_crossings["h"]=True
                        events.append((tid,direction,cname,frame_idx,event_time))
                if use_v and not tr.counted_crossings["v"]:
                    if (px<v_line_x<=cx) or (px>v_line_x>=cx):
                        if dx>0: direction="left_to_right"; direction_counts["left_to_right"]+=1
                        else: direction="right_to_left"; direction_counts["right_to_left"]+=1
                        class_totals[cname]+=1; tr.counted_crossings["v"]=True
                        events.append((tid,direction,cname,frame_idx,event_time))

        # Metrics
        m1.metric("➡️ Left → Right", direction_counts["left_to_right"])
        m2.metric("⬅️ Right → Left", direction_counts["right_to_left"])
        m3.metric("⬇️ Up → Down", direction_counts["up_to_down"])
        m4.metric("⬆️ Down → Up", direction_counts["down_to_up"])

        # Charts
        df_classes=pd.DataFrame(list(class_totals.items()),columns=["Class","Count"])
        fig_bar=px.bar(df_classes,x="Class",y="Count",color="Class",text="Count")
        class_chart.plotly_chart(fig_bar,use_container_width=True)
        df_dirs=pd.DataFrame(list(direction_counts.items()),columns=["Direction","Count"])
        fig_pie=px.pie(df_dirs,values="Count",names="Direction",hole=0.4)
        direction_chart.plotly_chart(fig_pie,use_container_width=True)

        # Log
        if events:
            df_log=pd.DataFrame(events,columns=["Track ID","Direction","Class","Frame","Time"])
            log_table.dataframe(df_log.tail(20),use_container_width=True)

        if fps_display:
            now=time.time(); fps=1.0/max(1e-6,now-fps_time); fps_time=now
            cv2.putText(frame,f"FPS:{fps:.1f}",(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(50,180,255),2)
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb,channels="RGB")
    cap.release()
    st.success("✅ Finished")
    if events:
        csv=pd.DataFrame(events,columns=["Track ID","Direction","Class","Frame","Time"]).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV",csv,"vehicle_counts.csv","text/csv")

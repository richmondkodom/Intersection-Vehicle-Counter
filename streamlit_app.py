import os
import cv2
import time
import math
import urllib.request
import tempfile
import numpy as np
import streamlit as st
import pandas as pd
import io
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
# Load network + classes
###############################################################################
with open(FILES["names"], "r") as f:
    CLASSES = [c.strip() for c in f.readlines()]

# Vehicle-like classes in COCO
VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle"}

net = cv2.dnn.readNetFromDarknet(FILES["cfg"], FILES["weights"])
try:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
except Exception:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

###############################################################################
# Simple Centroid Tracker
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
# Detection function
###############################################################################
def detect_vehicles(frame, conf_thresh=0.3, nms_thresh=0.4, target_classes=None, input_size=416):
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
st.set_page_config(page_title="Vehicle Counter (Streamlit)", layout="wide")
st.title("🚗 Vehicle Detector & Direction Counter")

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
    selected_classes = st.multiselect(
        "Vehicle classes to detect",
        sorted(list(VEHICLE_CLASSES)),
        default=["car", "truck", "bus", "motorbike", "bicycle"]
    )

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

# Counts
direction_counts = {"left_to_right": 0, "right_to_left": 0, "up_to_down": 0, "down_to_up": 0}
class_totals = defaultdict(int)

# For CSV report
events = []

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

        if use_h:
            cv2.line(frame, (0, h_line_y), (w, h_line_y), (0, 255, 255), 2)
        if use_v:
            cv2.line(frame, (v_line_x, 0), (v_line_x, h), (255, 255, 0), 2)

        for tid, (cx, cy, bw, bh, cname, conf) in tracks.items():
            tr = tracker.tracks[tid]
            if show_trace and len(tr.trace) >= 2:
                for i in range(1, len(tr.trace)):
                    cv2.line(frame, tr.trace[i-1], tr.trace[i], (200, 200, 200), 2)

            if draw_boxes:
                x = int(cx - bw/2); y = int(cy - bh/2)
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

            label = f"{cname} {int(conf*100)}%"
            if show_ids:
                label = f"ID {tid} | " + label
            cv2.putText(frame, label, (int(cx - bw/2), int(max(0, y-8))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10,220,10), 2)

            if len(tr.trace) >= 2:
                (px, py) = tr.trace[-2]
                dx = cx - px
                dy = cy - py

                if use_h and not tr.counted_crossings["h"]:
                    if (py < h_line_y <= cy) or (py > h_line_y >= cy):
                        if dy > 0:
                            direction_counts["up_to_down"] += 1
                            class_totals[cname] += 1
                            events.append({"frame": frame_idx, "track_id": tid, "class": cname, "direction": "up_to_down"})
                        else:
                            direction_counts["down_to_up"] += 1
                            class_totals[cname] += 1
                            events.append({"frame": frame_idx, "track_id": tid, "class": cname, "direction": "down_to_up"})
                        tr.counted_crossings["h"] = True

                if use_v and not tr.counted_crossings["v"]:
                    if (px < v_line_x <= cx) or (px > v_line_x >= cx):
                        if dx > 0:
                            direction_counts["left_to_right"] += 1
                            class_totals[cname] += 1
                            events.append({"frame": frame_idx, "track_id": tid, "class": cname, "direction": "left_to_right"})
                        else:
                            direction_counts["right_to_left"] += 1
                            class_totals[cname] += 1
                            events.append({"frame": frame_idx, "track_id": tid, "class": cname, "direction": "right_to_left"})
                        tr.counted_crossings["v"] = True

        if fps_display:
            now = time.time()
            fps = 1.0 / max(1e-6, (now - fps_time))
            fps_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,180,255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame_rgb, channels="RGB")

        with stats_col1:
            st.subheader("Direction Counts")
            st.write(pd.DataFrame([direction_counts]))
        with stats_col2:
            st.subheader("By Vehicle Class")
            st.write(pd.DataFrame([class_totals]))

        if st.button("⏹ Stop", key=f"stop_{frame_idx}"):
            break

    cap.release()

    st.success("Finished.")
    total = sum(direction_counts.values())
    st.metric("Grand Total", total)

    if events:
        df = pd.DataFrame(events)
        dir_df = pd.DataFrame([direction_counts])
        class_df = pd.DataFrame(list(class_totals.items()), columns=["class", "count"])
        class_df["Percentage"] = (class_df["count"] / class_df["count"].sum() * 100).round(2)
        pivot_df = pd.pivot_table(
            df,
            values="track_id",
            index="class",
            columns="direction",
            aggfunc="count",
            fill_value=0,
            margins=True,
            margins_name="Total"
        )

        # Excel export with formatting + charts
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # Write sheets
            df.to_excel(writer, index=False, sheet_name="Raw Events Log")
            dir_df.to_excel(writer, index=False, sheet_name="Direction Summary")
            class_df.to_excel(writer, index=False, sheet_name="Class Summary")
            pivot_df.to_excel(writer, sheet_name="Class×Direction Summary")

            workbook = writer.book
            header_fmt = workbook.add_format({"bold": True, "bg_color": "#DCE6F1", "border": 1, "align": "center"})
            total_fmt  = workbook.add_format({"bold": True, "bg_color": "#FFE699", "border": 1, "align": "center"})
            cell_fmt   = workbook.add_format({"border": 1})
            percent_fmt = workbook.add_format({"num_format": "0.00%", "border": 1})

            # Conditional formatting for Class Summary
            class_ws = writer.sheets["Class Summary"]
            class_ws.set_row(0, None, header_fmt)
            class_ws.set_column(0, 2, 15, cell_fmt)
            class_ws.conditional_format(1, 1, len(class_df), 1, {
                "type": "3_color_scale",
                "min_color": "#F8696B",
                "mid_color": "#FFEB84",
                "max_color": "#63BE7B"
            })
            class_ws.set_column(2, 2, 15, percent_fmt)

            # Conditional formatting for Class×Direction Summary (heatmap excl. totals)
            pivot_ws = writer.sheets["Class×Direction Summary"]
            nrows, ncols = pivot_df.shape
            pivot_ws.set_row(0, None, header_fmt)
            pivot_ws.set_column(0, ncols, 15, cell_fmt)
            if nrows >= 2 and ncols >= 2:
                pivot_ws.conditional_format(1, 1, nrows-2, ncols-2, {
                    "type": "3_color_scale",
                    "min_color": "#F8696B",
                    "mid_color": "#FFEB84",
                    "max_color": "#63BE7B"
                })
                # highlight totals row/col
                pivot_ws.set_row(nrows-1, None, total_fmt)
                pivot_ws.set_column(ncols-1, ncols-1, 15, total_fmt)

            # Grand Summary with charts
            summary_ws = workbook.add_worksheet("Grand Summary")

            # --- Direction Totals block ---
            summary_ws.write(0, 0, "Direction Totals", header_fmt)
            for col, val in enumerate(dir_df.columns):
                summary_ws.write(1, col, val, header_fmt)
            for col, val in enumerate(dir_df.iloc[0].tolist()):
                summary_ws.write(2, col, val, cell_fmt)

            # --- Class Totals block ---
            start_row = 5
            summary_ws.write(start_row, 0, "Vehicle Class Totals", header_fmt)
            for col, val in enumerate(class_df.columns):
                summary_ws.write(start_row+1, col, val, header_fmt)
            for r in range(len(class_df)):
                for c, v in enumerate(class_df.iloc[r].tolist()):
                    if c == 2:  # Percentage column
                        summary_ws.write(start_row+2+r, c, v/100.0, percent_fmt)
                    else:
                        summary_ws.write(start_row+2+r, c, v, cell_fmt)

            # --- Charts ---
            # Chart 1: Vehicles by Class (column)
            chart1 = workbook.add_chart({"type": "column"})
            chart1.add_series({
                "name": "Vehicles by Class",
                "categories": f"'Class Summary'!A2:A{len(class_df)+1}",
                "values":     f"'Class Summary'!B2:B{len(class_df)+1}",
            })
            chart1.set_title({"name": "Vehicles by Class"})
            chart1.set_x_axis({"name": "Class"})
            chart1.set_y_axis({"name": "Count"})
            chart1.set_style(11)

            # Chart 2: Directions by Class (stacked)
            chart2 = workbook.add_chart({"type": "column", "subtype": "stacked"})
            directions = list(pivot_df.columns[:-1])  # exclude Total col
            # Exclude the last row (Total) for categories/values
            for i in range(len(directions)):
                chart2.add_series({
                    "name":       ["Class×Direction Summary", 0, i+1],
                    "categories": ["Class×Direction Summary", 1, 0, len(pivot_df)-2, 0],
                    "values":     ["Class×Direction Summary", 1, i+1, len(pivot_df)-2, i+1],
                })
            chart2.set_title({"name": "Directions by Class"})
            chart2.set_x_axis({"name": "Class"})
            chart2.set_y_axis({"name": "Count"})
            chart2.set_style(12)

            # Chart 3: Direction Distribution (pie)
            chart3 = workbook.add_chart({"type": "pie"})
            chart3.add_series({
                "name": "Direction Distribution",
                "categories": f"'Direction Summary'!A1:D1",
                "values":     f"'Direction Summary'!A2:D2",
                "data_labels": {"percentage": True}
            })
            chart3.set_title({"name": "Direction Distribution"})

            # Chart 4: Class Distribution (pie)
            chart4 = workbook.add_chart({"type": "pie"})
            chart4.add_series({
                "name": "Class Distribution",
                "categories": f"'Class Summary'!A2:A{len(class_df)+1}",
                "values":     f"'Class Summary'!C2:C{len(class_df)+1}",
                "data_labels": {"percentage": True}
            })
            chart4.set_title({"name": "Class Distribution"})

            # Insert charts
            summary_ws.insert_chart(2, 4, chart1, {"x_offset": 20, "y_offset": 10})
            summary_ws.insert_chart(20, 4, chart2, {"x_offset": 20, "y_offset": 10})
            summary_ws.insert_chart(2, 9, chart3, {"x_offset": 20, "y_offset": 10})
            summary_ws.insert_chart(20, 9, chart4, {"x_offset": 20, "y_offset": 10})

        excel_data = output.getvalue()
        st.download_button(
            "⬇️ Download All Summaries (Excel Dashboard)",
            excel_data,
            file_name="vehicle_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No crossing events were detected, so there is nothing to export.")

###############################################################################
# Lightweight Smoke Tests (run locally by setting env RUN_SMOKE_TESTS=1)
###############################################################################
def _run_smoke_tests():
    # 1) detect_vehicles should return a list on a blank frame
    blank = np.zeros((416, 416, 3), dtype=np.uint8)
    out = detect_vehicles(blank, conf_thresh=0.9)  # high threshold => almost certainly empty
    assert isinstance(out, list), "detect_vehicles must return a list"

    # 2) CentroidTracker should create tracks for new detections
    trk = CentroidTracker(max_distance=50, max_age=2.0)
    dets = [(100, 100, 40, 20, "car", 0.8), (300, 120, 30, 15, "truck", 0.85)]
    tracks = trk.update(dets)
    assert len(tracks) == 2, "Tracker should create two tracks for two detections"

    # 3) Update with nearby detections should keep same IDs
    dets2 = [(105, 102, 40, 20, "car", 0.82), (295, 118, 30, 15, "truck", 0.83)]
    tracks2 = trk.update(dets2)
    assert set(tracks.keys()) == set(tracks2.keys()), "Track IDs should persist across small movements"

if os.environ.get("RUN_SMOKE_TESTS") == "1":
    _run_smoke_tests()

# 2D-to-3D Vehicle Location System
> Real-time traffic camera feed → YOLO detection → GPS map projection

![Pipeline](https://img.shields.io/badge/Computer%20Vision-YOLOv8-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---
<img width="956" height="560" alt="image" src="https://github.com/user-attachments/assets/b5349f3e-d2cb-4060-b285-e491eb33b25f" />

## What this project does

Standard traffic cameras produce video — but video alone doesn't tell you *where* vehicles are in the real world. This project bridges that gap by taking an ordinary roadside camera feed and transforming it into a live spatial map of vehicle locations, complete with speed estimates, vehicle classification, and congestion awareness.

The result: a single window showing the raw camera feed, the YOLO detection overlay, and a real-time map with color-coded vehicle dots — all running at 7+ FPS on a CPU-only machine.

---

## Why this matters

Cities and highway authorities already have thousands of roadside cameras installed. The problem is that most of those cameras are only used for human monitoring — someone watches a screen. They produce no structured data, no spatial awareness, and no analytics.

This project shows how existing camera infrastructure can be upgraded with computer vision to produce:

- **Real-time vehicle location** on a map without GPS hardware in the vehicles
- **Speed estimation** per vehicle using frame-to-frame position change
- **Traffic density and congestion metrics** derived purely from video
- **Vehicle type classification** (car, truck, bus, motorcycle)

Applications include smart city traffic management, highway digital twins, incident detection, infrastructure planning, and autonomous vehicle testing environments.

---

## How it works

### Stage 1 — Video ingestion
OpenCV reads the camera feed frame by frame. Each frame is resized to 640×360 for efficient processing while maintaining enough resolution for accurate detection.

### Stage 2 — YOLO detection
YOLOv8n (the nano variant of Ultralytics YOLOv8) runs inference on each frame, producing bounding boxes for vehicles. Only vehicle classes are requested: car (2), motorcycle (3), bus (5), truck (7).

**Why YOLOv8 specifically?**
YOLOv8 was chosen over newer versions (v9, v10, v11) for three reasons: it has the most mature documentation and community support, it runs efficiently on CPU without a dedicated GPU, and the Ultralytics package allows upgrading to newer versions by changing a single line of code. The accuracy difference on this task is negligible compared to the ecosystem advantages.

### Stage 3 — ByteTrack multi-object tracking
Detection alone gives boxes per frame with no identity continuity — a car detected in frame 100 has no connection to the same car in frame 101. ByteTrack solves this by assigning persistent IDs to each vehicle across frames. This is essential for speed estimation and trajectory analysis.

### Stage 4 — Homography projection (pixel → GPS)
This is the core mathematical step. By selecting 4 corresponding point pairs — known pixel locations in the camera image and their real GPS coordinates on the ground — OpenCV computes a homography matrix H using `cv2.findHomography()`.

Once H is computed, any pixel coordinate (x, y) in the frame can be projected to a real-world GPS coordinate (lat, lon) using `cv2.perspectiveTransform()`. This is applied to the bottom-center of each vehicle bounding box, which approximates the vehicle's ground contact point.

### Stage 5 — Speed estimation
Speed is estimated from the change in GPS position between consecutive frames divided by elapsed time. The GPS delta is converted to meters using the standard approximation:

```
Δlat_meters = Δlat × 111,320
Δlon_meters = Δlon × 111,320 × cos(latitude)
speed_ms    = sqrt(Δlat_m² + Δlon_m²) / Δt
speed_kmh   = speed_ms × 3.6
```

### Stage 6 — Map rendering and analytics
Vehicle positions are rendered on a dark map panel using their GPS coordinates projected back to map pixels. Dots are color-coded by vehicle type. A heatmap trail shows historical positions. A label bar displays live FPS, vehicle count, average speed, and congestion level (LOW / MED / HIGH).

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Ultralytics YOLOv8 | Object detection |
| OpenCV | Video processing, homography, rendering |
| ByteTrack | Multi-object tracking |
| NumPy | Matrix math, coordinate transforms |
| Folium | HTML map output (optional) |

---

## Setup

```bash
# Install dependencies
pip install ultralytics opencv-python folium numpy

# Install CPU-only PyTorch (no GPU needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Run
python traffic_pipeline.py
```

---

## Calibration

The homography requires 4 point pairs: pixel coordinates in the camera frame and their matching GPS coordinates on Google Maps. To recalibrate for a different camera:

1. Run `read_frame.py` to save a calibration frame
2. Run the clicking script to select 4 road points and get pixel coordinates
3. Open Google Maps satellite view, find the same road points, right-click → "What's here?" for GPS
4. Paste both sets of coordinates into `traffic_pipeline.py`

Accuracy improves when the 4 points are spread as wide as possible across the visible road surface.

---

## Results

- 7–13 FPS on Intel Iris Xe (integrated GPU, no CUDA)
- Vehicles detected with 65–87% confidence on side-angle highway footage
- Speed estimates within ~15% of actual speed (limited by calibration accuracy)
- Full pipeline running end-to-end on a standard laptop

---

## Next steps

- [ ] Improve calibration accuracy using lane marking GPS ground truth
- [ ] Add lane-level vehicle assignment
- [ ] Streamlit dashboard for interactive monitoring
- [ ] Export annotated output video for sharing
- [ ] Test on live RTSP camera stream
- [ ] Upgrade to YOLOv11 for improved small-vehicle detection

---

## Author

Built as a computer vision learning project, starting from zero CV knowledge and building a full detection-to-map pipeline in a single session.

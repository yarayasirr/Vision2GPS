import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import math
import webbrowser
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

VIDEO_PATH = r"C:\Users\yaray\Desktop\traffic-cv\video\traffic.mp4"

PIXEL_POINTS = np.float32([
    [181, 354],
    [523, 353],
    [371,  97],
    [419, 102],
])

GPS_POINTS = np.float32([
    [40.8451, -73.9108],
    [40.8451, -73.9098],
    [40.8461, -73.9103],
    [40.8461, -73.9100],
])

CONF_THRESHOLD  = 0.25
VEHICLE_CLASSES = [2, 3, 5, 7]
CLASS_NAMES     = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
CLASS_COLORS_CV = {          # BGR for OpenCV
    2: (0, 220, 0),
    3: (0, 165, 255),
    5: (255, 100, 0),
    7: (0, 60, 255),
}
CLASS_COLORS_HEX = {         # hex for Folium
    2: "#00dc00",
    3: "#ffa500",
    5: "#6464ff",
    7: "#ff3c00",
}

MAP_CENTER   = [40.8456, -73.9103]
MAP_HTML     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_map.html")
UPDATE_EVERY = 12            # frames between map saves
LAT_M        = 111320
LON_M        = 111320 * math.cos(math.radians(MAP_CENTER[0]))

# ─────────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────────

model    = YOLO("yolov8n.pt")
cap      = cv2.VideoCapture(VIDEO_PATH)
H_mat, _ = cv2.findHomography(PIXEL_POINTS, GPS_POINTS)

def pixel_to_gps(px, py):
    pt  = np.float32([[px, py]]).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pt, H_mat)
    return float(out[0][0][0]), float(out[0][0][1])

track_history   = defaultdict(list)   # id → [(lat, lon, t), ...]
speed_cache     = {}                  # id → kmh
cls_cache       = {}                  # id → class int
heatmap_pts     = []                  # [(lat, lon, cls), ...]

# ─────────────────────────────────────────────
#  MAP BUILDER
# ─────────────────────────────────────────────

def build_map(current_positions, stats):
    """Write a self-refreshing Folium HTML map."""
    # Build HTML manually so we can inject auto-refresh + dashboard header
    # without importing folium at every frame (folium is slow to import)
    markers_js = ""
    for tid, (lat, lon, cls) in current_positions.items():
        col   = CLASS_COLORS_HEX.get(cls, "#00dc00")
        name  = CLASS_NAMES.get(cls, "Vehicle")
        speed = speed_cache.get(tid, 0)
        spd_s = f"{int(speed)} km/h" if speed > 2 else "—"
        popup  = f"{name} #{tid} | {spd_s}"
        markers_js += f"""
        L.circleMarker([{lat}, {lon}], {{
            radius: 9, color: 'white', weight: 1.5,
            fillColor: '{col}', fillOpacity: 0.92
        }}).bindPopup('{popup}').addTo(map);
"""

    # Heatmap layer (last 400 points)
    heat_data = ""
    for (lat, lon, cls) in heatmap_pts[-400:]:
        heat_data += f"[{lat}, {lon}, 0.4],"

    fps_val  = stats.get("fps", 0)
    veh_val  = stats.get("vehicles", 0)
    spd_val  = stats.get("avg_speed", 0)
    cong_val = stats.get("congestion", "LOW")
    cong_col = {"HIGH": "#ff4444", "MED": "#ffaa00", "LOW": "#44ff44"}[cong_val]

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="2"/>
  <title>Traffic Monitor</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.heat/dist/leaflet-heat.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, sans-serif; background: #111; color: #eee; }}

    #dashboard {{
      display: flex; align-items: center; gap: 24px;
      padding: 10px 20px; background: #1a1a1a;
      border-bottom: 1px solid #333;
    }}
    .stat {{
      display: flex; flex-direction: column; align-items: center;
    }}
    .stat-val {{
      font-size: 22px; font-weight: 600; color: #fff;
      line-height: 1.1;
    }}
    .stat-label {{
      font-size: 11px; color: #888; margin-top: 2px; text-transform: uppercase;
    }}
    .divider {{ width: 1px; height: 36px; background: #333; }}
    #title {{
      font-size: 15px; font-weight: 500; color: #aaa;
      margin-right: auto;
    }}
    #cong {{
      font-size: 13px; font-weight: 600;
      padding: 4px 14px; border-radius: 20px;
      background: {cong_col}22; color: {cong_col};
      border: 1px solid {cong_col}66;
    }}
    #map {{ height: calc(100vh - 64px); width: 100%; }}

    /* Legend */
    .legend {{
      position: absolute; bottom: 28px; left: 12px; z-index: 1000;
      background: rgba(20,20,20,0.88); border-radius: 8px;
      padding: 10px 14px; font-size: 12px; color: #ccc;
      border: 1px solid #333;
    }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; border: 1.5px solid white; }}
    .refresh-note {{
      position: absolute; top: 74px; right: 12px; z-index: 1000;
      background: rgba(20,20,20,0.75); border-radius: 6px;
      padding: 5px 10px; font-size: 11px; color: #666;
      border: 1px solid #333;
    }}
  </style>
</head>
<body>
  <div id="dashboard">
    <span id="title">2D-to-3D Vehicle Location Monitor</span>
    <div class="stat">
      <span class="stat-val">{veh_val}</span>
      <span class="stat-label">Vehicles</span>
    </div>
    <div class="divider"></div>
    <div class="stat">
      <span class="stat-val">{spd_val:.0f} <span style="font-size:13px;color:#888">km/h</span></span>
      <span class="stat-label">Avg Speed</span>
    </div>
    <div class="divider"></div>
    <div class="stat">
      <span class="stat-val">{fps_val:.1f}</span>
      <span class="stat-label">FPS</span>
    </div>
    <div class="divider"></div>
    <span id="cong">{cong_val} CONGESTION</span>
  </div>

  <div id="map"></div>
  <div class="refresh-note">Auto-refreshes every 2s</div>

  <div class="legend">
    <div style="font-size:11px;color:#888;margin-bottom:6px;font-weight:500;">VEHICLE TYPE</div>
    <div class="legend-row"><div class="dot" style="background:#00dc00"></div> Car</div>
    <div class="legend-row"><div class="dot" style="background:#ffa500"></div> Motorcycle</div>
    <div class="legend-row"><div class="dot" style="background:#6464ff"></div> Bus</div>
    <div class="legend-row"><div class="dot" style="background:#ff3c00"></div> Truck</div>
  </div>

  <script>
    var map = L.map('map').setView({MAP_CENTER}, 17);

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19
    }}).addTo(map);

    // Heatmap layer
    var heatData = [{heat_data}];
    if (heatData.length > 0) {{
      L.heatLayer(heatData, {{radius: 20, blur: 18, maxZoom: 17,
                               gradient: {{0.3:'blue', 0.6:'lime', 1.0:'red'}}
      }}).addTo(map);
    }}

    // Vehicle markers
    {markers_js}
  </script>
</body>
</html>""".replace("{MAP_CENTER}", f"[{MAP_CENTER[0]}, {MAP_CENTER[1]}]")

    with open(MAP_HTML, "w", encoding="utf-8") as f:
        f.write(html)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def estimate_speed(tid, lat, lon, now):
    h = track_history[tid]
    if len(h) < 2:
        return 0.0
    pl, plo, pt = h[-1]
    dt = now - pt
    if dt <= 0:
        return 0.0
    dist = math.sqrt(((lat-pl)*LAT_M)**2 + ((lon-plo)*LON_M)**2)
    return min((dist/dt)*3.6, 180.0)   # cap at 180 km/h

fps_disp    = 0.0
t_prev      = time.time()
frame_count = 0
map_opened  = False

# Write empty map and open browser
build_map({}, {"fps":0,"vehicles":0,"avg_speed":0,"congestion":"LOW"})
webbrowser.open(f"file:///{MAP_HTML.replace(os.sep, '/')}")
print(f"Map opened in browser: {MAP_HTML}")
print("Running pipeline — press Q on the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        track_history.clear()
        continue

    frame_count += 1
    small    = cv2.resize(frame, (640, 360))
    original = small.copy()

    results = model.track(small, classes=VEHICLE_CLASSES, conf=CONF_THRESHOLD,
                          persist=True, tracker="bytetrack.yaml", verbose=False)

    annotated         = results[0].plot(line_width=1)
    current_frame_ids = set()
    current_positions = {}
    now               = time.time()

    if results[0].boxes.id is not None:
        for box, tid_t, cls_t in zip(results[0].boxes.xyxy,
                                      results[0].boxes.id,
                                      results[0].boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            tid = int(tid_t)
            cls = int(cls_t)
            current_frame_ids.add(tid)
            cls_cache[tid] = cls

            cx, cy = int((x1+x2)/2), int(y2)
            lat, lon = pixel_to_gps(cx, cy)

            speed = estimate_speed(tid, lat, lon, now)
            speed_cache[tid] = speed

            track_history[tid].append((lat, lon, now))
            if len(track_history[tid]) > 30:
                track_history[tid].pop(0)

            current_positions[tid] = (lat, lon, cls)
            heatmap_pts.append((lat, lon, cls))

            # Speed on detection box
            col = CLASS_COLORS_CV.get(cls, (0,220,0))
            lbl = f"{int(speed)}km/h" if speed > 2 else CLASS_NAMES.get(cls,"")
            cv2.putText(annotated, lbl, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

    # FPS
    t_now    = time.time()
    fps_disp = 0.9*fps_disp + 0.1*(1.0/max(t_now-t_prev, 0.001))
    t_prev   = t_now

    active    = len(current_frame_ids)
    avg_speed = (sum(speed_cache.get(t,0) for t in current_frame_ids)/active
                 if active else 0)
    congestion = "HIGH" if active>10 else "MED" if active>5 else "LOW"
    cong_col   = {"HIGH":(0,0,255),"MED":(0,165,255),"LOW":(0,220,0)}[congestion]

    # Stats overlay
    cv2.rectangle(annotated, (0,0), (235,90), (0,0,0), -1)
    cv2.putText(annotated, f"FPS: {fps_disp:.1f}",          (8,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,0), 1)
    cv2.putText(annotated, f"Vehicles: {active}",            (8,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 1)
    cv2.putText(annotated, f"Avg: {avg_speed:.0f} km/h",    (8,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,220,220), 1)
    cv2.putText(annotated, f"Congestion: {congestion}",      (8,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, cong_col, 1)

    # Side-by-side display
    top  = np.hstack([original, annotated])
    bar  = np.zeros((28, top.shape[1], 3), dtype=np.uint8)
    bar[:] = (18, 18, 22)
    cv2.putText(bar,
                f"Approximate Google Map Location  |  FPS {fps_disp:.1f}  |  "
                f"Vehicles: {active}  |  Avg Speed: {avg_speed:.0f} km/h  |  "
                f"Congestion: {congestion}",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    display = np.vstack([top, bar])
    cv2.imshow("2D-to-3D Vehicle Location", display)

    # Update map every N frames
    if frame_count % UPDATE_EVERY == 0:
        stats = {"fps": fps_disp, "vehicles": active,
                 "avg_speed": avg_speed, "congestion": congestion}
        build_map(current_positions, stats)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Final map save
stats = {"fps": fps_disp, "vehicles": 0, "avg_speed": 0, "congestion": "LOW"}
build_map({}, stats)
print("Done!")
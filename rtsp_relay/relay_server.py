"""
RTSP Relay Microservice
========================
Reads RTSP streams from local IP cameras and re-serves them as
MJPEG HTTP streams accessible by any remote server (production included).

Expose publicly using ngrok or any tunnel. The production app then
uses the public MJPEG URL instead of the raw RTSP URL.

Usage:
    python relay_server.py                    # Uses config.json
    python relay_server.py --ngrok            # Auto-starts ngrok tunnel
    python relay_server.py --port 6001        # Custom port

Endpoints:
    GET /                          → Health dashboard (HTML)
    GET /health                    → JSON health status
    GET /streams                   → JSON list of all stream configs
    GET /stream/<cam_id>           → MJPEG live stream
    GET /snapshot/<cam_id>         → Single JPEG snapshot
    POST /api/streams/add          → Add a new stream at runtime
    POST /api/streams/remove       → Remove a stream at runtime
    GET /tunnel                    → Current ngrok public URL (if active)
"""

import cv2
import time
import json
import threading
import argparse
import logging
import os
import sys
from datetime import datetime
from flask import Flask, Response, jsonify, request, render_template_string
import numpy as np

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("relay.log", encoding="utf-8"),
    ],
)
# Force UTF-8 on Windows stdout to prevent cp1252 encoding errors
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
log = logging.getLogger("rtsp_relay")

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")
DEFAULT_CONFIG = {
    "port": 6001,
    "host": "0.0.0.0",
    "auth_token": "",          # Optional Bearer token for stream protection
    "reconnect_delay": 3,      # Seconds before retrying a dead stream
    "jpeg_quality": 70,        # 1-100, higher = better quality / more bandwidth
    "target_fps": 15,          # Max FPS to relay
    "streams": [
        {
            "id": 1,
            "name": "IP Camera 1",
            "rtsp_url": "rtsp://admin:admin@192.168.1.20:1935",
            "enabled": True
        }
    ]
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            log.info(f"Config loaded from {CONFIG_FILE}")
            return cfg
        except Exception as e:
            log.warning(f"Could not read config.json ({e}), using defaults.")
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        log.error(f"Could not save config: {e}")

config = load_config()

# ── Stream State ───────────────────────────────────────────────────────────────
# { stream_id: {"frame": np.ndarray | None, "status": str, "fps": float,
#               "frames": int, "last_frame_ts": float, "lock": Lock } }
stream_state: dict = {}
stream_threads: dict = {}          # { stream_id: threading.Thread }
stream_stop_events: dict = {}      # { stream_id: threading.Event }

ngrok_public_url: str = ""         # Filled when ngrok is active


# ── Placeholder frame generator ───────────────────────────────────────────────
def _make_placeholder(text: str, width=640, height=360) -> np.ndarray:
    """Return a grey frame with centered text."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
    x = (width - tw) // 2
    y = (height + th) // 2
    cv2.putText(frame, text, (x, y), font, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (10, height - 10), font, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
    return frame


# ── Capture thread ─────────────────────────────────────────────────────────────
def _capture_thread(stream_cfg: dict, stop_event: threading.Event):
    sid = stream_cfg["id"]
    url = stream_cfg["rtsp_url"]
    reconnect_delay = config.get("reconnect_delay", 3)
    target_fps = config.get("target_fps", 15)
    frame_interval = 1.0 / max(target_fps, 1)
    lock = stream_state[sid]["lock"]

    log.info(f"[Cam {sid}] Capture thread started -> {url}")

    def _open():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Short timeouts so we detect dead streams quickly
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        return cap

    cap = _open()
    if cap.isOpened():
        with lock:
            stream_state[sid]["status"] = "connected"
        log.info(f"[Cam {sid}] Connected.")
    else:
        with lock:
            stream_state[sid]["status"] = "offline"
        log.warning(f"[Cam {sid}] Could not open stream: {url}")

    last_time = time.time()

    consecutive_failures = 0
    while not stop_event.is_set():
        if not cap.isOpened():
            with lock:
                stream_state[sid]["status"] = "reconnecting"
                stream_state[sid]["frame"] = None
            log.warning(f"[Cam {sid}] Reconnecting in {reconnect_delay}s...")
            time.sleep(reconnect_delay)
            cap = _open()
            consecutive_failures = 0
            if cap.isOpened():
                with lock:
                    stream_state[sid]["status"] = "connected"
                log.info(f"[Cam {sid}] Reconnected.")
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                log.warning(f"[Cam {sid}] Bad frame 10 consecutive times — reconnecting.")
                cap.release()
                cap = cv2.VideoCapture.__new__(cv2.VideoCapture)  # blank cap
                with lock:
                    stream_state[sid]["status"] = "reconnecting"
                    stream_state[sid]["frame"] = None
                time.sleep(reconnect_delay)
                cap = _open()
                consecutive_failures = 0
            else:
                time.sleep(0.05)
            continue

        consecutive_failures = 0

        now = time.time()
        elapsed = now - last_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        last_time = time.time()
        with lock:
            stream_state[sid]["frame"] = frame
            stream_state[sid]["status"] = "connected"
            stream_state[sid]["frames"] += 1
            stream_state[sid]["last_frame_ts"] = last_time
            # Rolling FPS estimate
            stream_state[sid]["fps"] = round(1.0 / max(elapsed, 0.001), 1)

    cap.release()
    log.info(f"[Cam {sid}] Capture thread stopped.")


# ── Stream management helpers ──────────────────────────────────────────────────
def _ensure_state(sid: int):
    if sid not in stream_state:
        stream_state[sid] = {
            "frame": None,
            "status": "initializing",
            "fps": 0.0,
            "frames": 0,
            "last_frame_ts": 0.0,
            "lock": threading.Lock(),
        }

def start_stream(stream_cfg: dict):
    sid = stream_cfg["id"]
    _ensure_state(sid)
    if sid in stream_threads and stream_threads[sid].is_alive():
        log.info(f"[Cam {sid}] Already running.")
        return
    stop_event = threading.Event()
    stream_stop_events[sid] = stop_event
    t = threading.Thread(
        target=_capture_thread,
        args=(stream_cfg, stop_event),
        daemon=True,
        name=f"relay-cam-{sid}",
    )
    t.start()
    stream_threads[sid] = t
    log.info(f"[Cam {sid}] Thread started.")

def stop_stream(sid: int):
    if sid in stream_stop_events:
        stream_stop_events[sid].set()
    if sid in stream_threads:
        stream_threads[sid].join(timeout=5)
        del stream_threads[sid]
    stream_stop_events.pop(sid, None)
    log.info(f"[Cam {sid}] Stopped.")


# ── MJPEG generator ────────────────────────────────────────────────────────────
def _mjpeg_generator(sid: int):
    quality = config.get("jpeg_quality", 70)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _ensure_state(sid)

    while True:
        frame = None
        status = "offline"
        lock = stream_state[sid]["lock"]
        with lock:
            frame_src = stream_state[sid].get("frame")
            status = stream_state[sid].get("status", "offline")
            if frame_src is not None:
                frame = frame_src.copy()

        if frame is None:
            frame = _make_placeholder(f"Camera {sid} — {status.upper()}")

        # Stamp timestamp on frame
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 1, cv2.LINE_AA)
        # Stamp stream ID
        cv2.putText(frame, f"Relay Cam {sid}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            time.sleep(0.05)
            continue

        jpg_bytes = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpg_bytes)).encode() + b"\r\n"
            b"\r\n" + jpg_bytes + b"\r\n"
        )

        target_fps = config.get("target_fps", 15)
        time.sleep(1.0 / max(target_fps, 1))


# ── Auth helper ────────────────────────────────────────────────────────────────
def _check_auth(req) -> bool:
    token = config.get("auth_token", "")
    if not token:
        return True  # No auth configured
    auth_header = req.headers.get("Authorization", "")
    return auth_header == f"Bearer {token}"


# ── Routes ─────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RTSP Relay Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 24px; }
    h1 { font-size: 1.6rem; font-weight: 700; color: #60a5fa; margin-bottom: 6px; }
    .subtitle { color: #64748b; font-size: 0.85rem; margin-bottom: 28px; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .badge-green { background: #14532d; color: #4ade80; }
    .badge-yellow { background: #713f12; color: #fde68a; }
    .badge-red { background: #7f1d1d; color: #fca5a5; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; }
    .card { background: #1e2330; border: 1px solid #2d3748; border-radius: 12px; padding: 18px; }
    .card h2 { font-size: 1rem; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .stream-img { width: 100%; border-radius: 8px; border: 1px solid #2d3748; background: #111; }
    .info { margin-top: 10px; font-size: 0.78rem; color: #94a3b8; }
    .url-box { background: #0f1117; border: 1px solid #334155; border-radius: 6px; padding: 8px 12px;
               font-family: monospace; font-size: 0.8rem; word-break: break-all; margin-top: 8px;
               color: #7dd3fc; }
    .tunnel-box { background: #1e3a5f; border: 1px solid #3b82f6; border-radius: 10px; padding: 14px 18px; margin-bottom: 24px; }
    .tunnel-box .label { font-size: 0.75rem; color: #93c5fd; margin-bottom: 4px; }
    .tunnel-box .url { font-size: 0.9rem; font-family: monospace; color: #bfdbfe; word-break: break-all; }
    footer { margin-top: 36px; color: #475569; font-size: 0.75rem; }
  </style>
</head>
<body>
  <h1>🎥 RTSP Relay Microservice</h1>
  <div class="subtitle">Forwarding local RTSP streams as public MJPEG endpoints</div>

  {% if tunnel_url %}
  <div class="tunnel-box">
    <div class="label">🌐 Active ngrok Tunnel</div>
    <div class="url">{{ tunnel_url }}</div>
  </div>
  {% endif %}

  <div class="grid">
    {% for s in streams %}
    <div class="card">
      <h2>
        {{ s.name }}
        <span class="badge {% if s.status == 'connected' %}badge-green{% elif s.status == 'reconnecting' %}badge-yellow{% else %}badge-red{% endif %}">
          {{ s.status }}
        </span>
      </h2>
      <img class="stream-img" src="/stream/{{ s.id }}" alt="Stream {{ s.id }}">
      <div class="info">
        FPS: {{ s.fps }} &nbsp;|&nbsp; Frames: {{ s.frames }}
        <div class="url-box">MJPEG → {{ base_url }}/stream/{{ s.id }}</div>
        {% if tunnel_url %}
        <div class="url-box">Public → {{ tunnel_url }}/stream/{{ s.id }}</div>
        {% endif %}
      </div>
    </div>
    {% endfor %}
  </div>

  <footer>Smart AI Monitoring — RTSP Relay &nbsp;|&nbsp; Port {{ port }}</footer>

  <script>
    // Auto-refresh status every 5s without reloading streams
    setInterval(() => {
      fetch('/streams').then(r => r.json()).then(data => {
        console.log('Status refreshed', data);
      });
    }, 5000);
  </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    streams_data = []
    for s in config.get("streams", []):
        sid = s["id"]
        _ensure_state(sid)
        with stream_state[sid]["lock"]:
            st = stream_state[sid]
            streams_data.append({
                "id": sid,
                "name": s.get("name", f"Camera {sid}"),
                "status": st["status"],
                "fps": st["fps"],
                "frames": st["frames"],
            })
    base = request.host_url.rstrip("/")
    return render_template_string(
        DASHBOARD_HTML,
        streams=streams_data,
        tunnel_url=ngrok_public_url,
        base_url=base,
        port=config.get("port", 6001),
    )


@app.route("/health")
def health():
    statuses = {}
    for sid, st in stream_state.items():
        with st["lock"]:
            statuses[str(sid)] = {
                "status": st["status"],
                "fps": st["fps"],
                "frames": st["frames"],
                "last_frame_age_s": round(time.time() - st["last_frame_ts"], 1) if st["last_frame_ts"] else None,
            }
    return jsonify({
        "ok": True,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tunnel_url": ngrok_public_url or None,
        "streams": statuses,
    })


@app.route("/streams")
def list_streams():
    result = []
    for s in config.get("streams", []):
        sid = s["id"]
        _ensure_state(sid)
        with stream_state[sid]["lock"]:
            st = stream_state[sid]
        result.append({
            "id": sid,
            "name": s.get("name", f"Camera {sid}"),
            "rtsp_url": s.get("rtsp_url"),
            "enabled": s.get("enabled", True),
            "status": st["status"],
            "fps": st["fps"],
            "frames": st["frames"],
            "mjpeg_url": f"{request.host_url}stream/{sid}",
            "public_mjpeg_url": f"{ngrok_public_url}/stream/{sid}" if ngrok_public_url else None,
        })
    return jsonify({"streams": result, "tunnel_url": ngrok_public_url or None})


@app.route("/stream/<int:cam_id>")
def stream(cam_id):
    if not _check_auth(request):
        return Response("Unauthorized", status=401,
                        headers={"WWW-Authenticate": "Bearer"})
    _ensure_state(cam_id)
    return Response(
        _mjpeg_generator(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/snapshot/<int:cam_id>")
def snapshot(cam_id):
    if not _check_auth(request):
        return Response("Unauthorized", status=401)
    _ensure_state(cam_id)
    frame = None
    with stream_state[cam_id]["lock"]:
        f = stream_state[cam_id].get("frame")
        if f is not None:
            frame = f.copy()
    if frame is None:
        frame = _make_placeholder(f"Camera {cam_id} — offline")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/tunnel")
def tunnel_info():
    return jsonify({
        "active": bool(ngrok_public_url),
        "url": ngrok_public_url or None,
    })


@app.route("/api/streams/add", methods=["POST"])
def add_stream():
    data = request.get_json() or {}
    sid = int(time.time())  # unique id
    new_stream = {
        "id": sid,
        "name": data.get("name", f"Camera {sid}"),
        "rtsp_url": data.get("rtsp_url", ""),
        "enabled": True,
    }
    if not new_stream["rtsp_url"]:
        return jsonify({"error": "rtsp_url is required"}), 400
    config["streams"].append(new_stream)
    save_config(config)
    start_stream(new_stream)
    return jsonify({"status": "added", "stream": new_stream})


@app.route("/api/streams/remove", methods=["POST"])
def remove_stream():
    data = request.get_json() or {}
    sid = int(data.get("id", -1))
    config["streams"] = [s for s in config["streams"] if s["id"] != sid]
    save_config(config)
    stop_stream(sid)
    return jsonify({"status": "removed", "id": sid})


# ── ngrok tunnel ───────────────────────────────────────────────────────────────
def _start_ngrok(port: int):
    """Attempts to start an ngrok HTTP tunnel and stores the public URL."""
    global ngrok_public_url
    try:
        from pyngrok import ngrok as pyngrok, conf
        conf.get_default().log_level = "WARNING"

        # Set authtoken from env var if provided
        authtoken = os.environ.get("NGROK_AUTHTOKEN", "").strip()
        if authtoken:
            pyngrok.set_auth_token(authtoken)
            log.info("[ngrok] Authtoken configured from NGROK_AUTHTOKEN env var.")
        else:
            log.warning("[ngrok] No NGROK_AUTHTOKEN set. Set it with:")
            log.warning("  $env:NGROK_AUTHTOKEN='your_token'  (PowerShell)")
            log.warning("  OR pass --authtoken <token> flag")
            log.warning("  Get token at: https://dashboard.ngrok.com/get-started/your-authtoken")

        tunnel = pyngrok.connect(port, "http")
        ngrok_public_url = tunnel.public_url.replace("http://", "https://")
        log.info(f"[ngrok] Tunnel active: {ngrok_public_url}")

        # Write public URL to a shared file so app.py can pick it up
        out_path = os.path.join(os.path.dirname(__file__), "tunnel_url.txt")
        with open(out_path, "w") as f:
            # Write per-stream public MJPEG URLs
            lines = [ngrok_public_url]
            for s in config.get("streams", []):
                lines.append(f"cam{s['id']}={ngrok_public_url}/stream/{s['id']}")
            f.write("\n".join(lines))
        log.info(f"[ngrok] Tunnel URL saved to {out_path}")

    except ImportError:
        log.error("[ngrok] pyngrok not installed. Run: pip install pyngrok")
    except Exception as e:
        log.error(f"[ngrok] Failed to start tunnel: {e}")


# ── Startup ────────────────────────────────────────────────────────────────────
def start_all_streams():
    for s in config.get("streams", []):
        if s.get("enabled", True):
            start_stream(s)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTSP Relay Microservice")
    parser.add_argument("--port", type=int, default=config.get("port", 6001),
                        help="Port to run the relay server on")
    parser.add_argument("--ngrok", action="store_true",
                        help="Start an ngrok tunnel automatically")
    parser.add_argument("--authtoken", type=str, default="",
                        help="ngrok authtoken (or set NGROK_AUTHTOKEN env var)")
    parser.add_argument("--host", type=str, default=config.get("host", "0.0.0.0"),
                        help="Host to bind on")
    args = parser.parse_args()

    config["port"] = args.port
    config["host"] = args.host

    log.info("=" * 60)
    log.info("  RTSP Relay Microservice starting...")
    log.info(f"  Streams: {len(config.get('streams', []))}")
    log.info(f"  Binding: {args.host}:{args.port}")
    log.info("=" * 60)

    start_all_streams()

    # Allow --authtoken flag to override env var
    if args.authtoken:
        os.environ["NGROK_AUTHTOKEN"] = args.authtoken

    if args.ngrok:
        ngrok_thread = threading.Thread(
            target=_start_ngrok, args=(args.port,), daemon=True
        )
        ngrok_thread.start()
        time.sleep(2)  # Let ngrok settle

    app.run(host=args.host, port=args.port, debug=False, threaded=True)

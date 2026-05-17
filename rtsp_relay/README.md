# RTSP Relay Microservice

Bridges your **local RTSP IP camera** to production servers via a public MJPEG HTTP stream.

## Problem it solves

```
Production Server ❌──────────────> rtsp://192.168.1.20:1935  (private LAN, unreachable)
Production Server ✅──> ngrok ──── Relay (localhost:6001) ──> rtsp://192.168.1.20:1935
```

## Files

| File | Purpose |
|------|---------|
| `relay_server.py` | The Flask microservice |
| `config.json` | Camera stream configuration |
| `start_relay.bat` | One-click startup script (Windows) |
| `requirements.txt` | Python dependencies |

## Quick Start

### 1. Run locally (LAN only)
```bash
python relay_server.py
```
Access at: `http://localhost:6001`

### 2. Run with public ngrok tunnel (for production)
```bash
python relay_server.py --ngrok
```
Or double-click **`start_relay.bat`**

This will:
- Start the relay on port 6001
- Create a public ngrok tunnel (e.g. `https://abc123.ngrok.io`)
- Write `tunnel_url.txt` so `app.py` auto-picks it up

## Configuration (`config.json`)

```json
{
  "port": 6001,
  "jpeg_quality": 70,
  "target_fps": 15,
  "reconnect_delay": 3,
  "auth_token": "",
  "streams": [
    {
      "id": 1,
      "name": "IP Camera 1",
      "rtsp_url": "rtsp://admin:admin@192.168.1.20:1935",
      "enabled": true
    }
  ]
}
```

Add more cameras by adding entries to `streams[]`.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | HTML dashboard with live previews |
| `GET /health` | JSON health status |
| `GET /streams` | JSON list of streams + public URLs |
| `GET /stream/<id>` | MJPEG live stream |
| `GET /snapshot/<id>` | Single JPEG frame |
| `GET /tunnel` | Current ngrok URL |
| `POST /api/streams/add` | Add stream at runtime |
| `POST /api/streams/remove` | Remove stream |

## How `app.py` auto-switches

When `rtsp_relay/tunnel_url.txt` exists, `app.py` reads it at startup
and automatically replaces the RTSP source with the public MJPEG URL:

```
tunnel_url.txt content:
  https://abc123.ngrok.io
  cam1=https://abc123.ngrok.io/stream/1
```

No manual changes needed in `app.py` — it's automatic.

## ngrok Setup (first time only)

1. Sign up at https://ngrok.com (free)
2. Copy your authtoken from the dashboard
3. Run once: `ngrok config add-authtoken <YOUR_TOKEN>`

Free plan supports 1 tunnel, which is enough for this use case.

## Protect the stream (optional)

Set `auth_token` in `config.json`:
```json
{ "auth_token": "mysecrettoken123" }
```

Then access streams with:
```
Authorization: Bearer mysecrettoken123
```

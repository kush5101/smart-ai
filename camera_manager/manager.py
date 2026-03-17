import cv2
import threading
import time

class CameraManager:
    def __init__(self):
        self.sources = {}
        self.caps = {}
        self.frames = {}
        self.locks = {}
        self._stop_event = threading.Event()

    def add_source(self, cam_id, source_url, name):
        self.sources[cam_id] = {"url": source_url, "name": name}
        self.locks[cam_id] = threading.Lock()
        
        # Start a dedicated thread for this camera
        t = threading.Thread(target=self._update_loop, args=(cam_id,), daemon=True)
        t.start()

    def _update_loop(self, cam_id):
        url = self.sources[cam_id]["url"]
        while not self._stop_event.is_set():
            if cam_id not in self.caps:
                print(f"[Cam {cam_id}] Attempting to connect to: {url}")
                self.caps[cam_id] = cv2.VideoCapture(url)
                if not self.caps[cam_id].isOpened():
                    print(f"[Cam {cam_id}] FAILED to open source.")
                    time.sleep(5)
                    continue
                # Optimize for RTSP
                self.caps[cam_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f"[Cam {cam_id}] Connected successfully.")
            
            success, frame = self.caps[cam_id].read()
            if success:
                with self.locks[cam_id]:
                    self.frames[cam_id] = frame
            else:
                print(f"[Cam {cam_id}] Connection lost or no frame. Retrying...")
                if cam_id in self.caps:
                    self.caps[cam_id].release()
                    del self.caps[cam_id]
                time.sleep(2)

    def get_frame(self, cam_id):
        if cam_id in self.frames:
            with self.locks[cam_id]:
                return self.frames[cam_id].copy()
        return None

    def list_cameras(self):
        return [{"id": k, "name": v["name"], "url": v["url"]} for k,v in self.sources.items()]

    def stop(self):
        self._stop_event.set()
        for cap in self.caps.values():
            cap.release()

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
        
        is_local = isinstance(source_url, int) or (isinstance(source_url, str) and str(source_url).isdigit())
        target_loop = self._webcam_update_loop if is_local else self._rtsp_update_loop
        
        # Start a dedicated thread for this camera
        t = threading.Thread(target=target_loop, args=(cam_id,), daemon=True)
        t.start()

    def _webcam_init(self, url):
        c = cv2.VideoCapture(int(url), cv2.CAP_DSHOW)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            c.set(cv2.CAP_PROP_FPS, 30)
        return c

    def _webcam_update_loop(self, cam_id):
        url = self.sources[cam_id]["url"]
        import os
        if os.name == 'nt':
            import ctypes
            ctypes.windll.ole32.CoInitialize(0)
            
        while not self._stop_event.is_set():
            if cam_id not in self.caps:
                print(f"[CamManager {cam_id} - WEBCAM] Attempting to connect to: {url}")
                self.caps[cam_id] = self._webcam_init(url)
                if not self.caps[cam_id].isOpened():
                    print(f"[CamManager {cam_id}] FAILED to open source.")
                    time.sleep(5)
                    continue
                print(f"[CamManager {cam_id}] Connected successfully.")
            
            success, frame = self.caps[cam_id].read()
            if success and frame is not None and frame.size > 0:
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_clone = frame.copy()
                    with self.locks[cam_id]:
                        self.frames[cam_id] = frame_clone
            else:
                print(f"[CamManager {cam_id}] Connection lost or no frame. Retrying...")
                if cam_id in self.caps:
                    self.caps[cam_id].release()
                    del self.caps[cam_id]
                time.sleep(2)
            time.sleep(0.01)

    def _rtsp_init(self, url, is_rtsp):
        if is_rtsp:
            c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if c.isOpened():
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                c.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                c.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            return c
        return cv2.VideoCapture(url)

    def _rtsp_update_loop(self, cam_id):
        url = self.sources[cam_id]["url"]
        is_rtsp = isinstance(url, str) and url.startswith('rtsp')
        
        while not self._stop_event.is_set():
            if cam_id not in self.caps:
                print(f"[CamManager {cam_id} - NETWORK] Attempting to connect to: {url}")
                self.caps[cam_id] = self._rtsp_init(url, is_rtsp)
                if not self.caps[cam_id].isOpened():
                    print(f"[CamManager {cam_id}] FAILED to open source.")
                    time.sleep(5)
                    continue
                print(f"[CamManager {cam_id}] Connected successfully.")
            
            success, frame = self.caps[cam_id].read()
            if success and frame is not None and frame.size > 0:
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame_clone = frame.copy()
                    with self.locks[cam_id]:
                        self.frames[cam_id] = frame_clone
            else:
                print(f"[CamManager {cam_id}] Connection lost or no frame. Retrying...")
                if cam_id in self.caps:
                    self.caps[cam_id].release()
                    del self.caps[cam_id]
                time.sleep(2)
            time.sleep(0.01)

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

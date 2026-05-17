import cv2
import time

def test_rtsp(url):
    print(f"Testing RTSP URL: {url}")
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return False
    
    print("Stream opened successfully. Reading first frame...")
    ret, frame = cap.read()
    if ret:
        print(f"Success! Frame resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("Error: Could not read frame from stream.")
    
    cap.release()
    return ret

if __name__ == "__main__":
    # A public RTSP stream for testing
    test_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov"
    test_rtsp(test_url)

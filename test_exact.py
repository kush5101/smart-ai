import cv2
import time
import os

def test_exact():
    print("Testing Cam 0 with exact app.py sequence...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open.")
        return
    
    # Exact sequence in my latest fix for app.py
    r1 = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    r2 = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    r3 = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    r4 = cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"Set Results: MJPG:{r1}, W:{r2}, H:{r3}, FPS:{r4}")
    
    # Warmup exactly like app.py (20 frames)
    for i in range(20):
        success, frame = cap.read()
        if success:
            print(f"Warmup {i} success, mean: {frame.mean()}")
        else:
            print(f"Warmup {i} failed.")
        time.sleep(0.01)
    
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"Final status: shape={frame.shape}, mean={frame.mean()}")
        cv2.imwrite("exact_test_result.jpg", frame)
        print("Saved exact_test_result.jpg")
    else:
        print("Final read failed.")
        
    cap.release()

if __name__ == "__main__":
    test_exact()

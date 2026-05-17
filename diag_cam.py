import cv2
import time
import os

def diag():
    print("Testing Cam 0 with DSHOW...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Failed to open Cam 0 with DSHOW. Trying default...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open Cam 0.")
        return

    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Warmup...")
    for _ in range(30):
        cap.read()
    
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"Captured frame of shape {frame.shape}")
        # Check if frame is mostly black or noise
        mean = frame.mean()
        print(f"Mean pixel value: {mean}")
        cv2.imwrite("diag_frame_default.jpg", frame)
        print("Saved diag_frame_default.jpg")
    else:
        print("Failed to capture frame.")

    # Try with MJPG
    print("Testing with MJPG...")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    for _ in range(30):
        cap.read()
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"Captured MJPG frame of shape {frame.shape}, mean: {frame.mean()}")
        cv2.imwrite("diag_frame_mjpg.jpg", frame)
        print("Saved diag_frame_mjpg.jpg")

    cap.release()

if __name__ == "__main__":
    diag()

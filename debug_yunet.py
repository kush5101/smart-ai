import cv2
import os

model_dir = os.path.join(os.getcwd(), "models")
detect_model_path = os.path.join(model_dir, "face_detection_yunet.onnx")

print(f"Testing FaceDetectorYN with model: {detect_model_path}")
try:
    # Standard signature
    detector = cv2.FaceDetectorYN.create(
        model=detect_model_path,
        config="",
        inputSize=(640, 480),
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=5000
    )
    print("Creation successful with named arguments.")
except Exception as e:
    print(f"Error with named arguments: {e}")

try:
    # Positional arguments
    detector = cv2.FaceDetectorYN.create(
        detect_model_path,
        "",
        (640, 480),
        0.85,
        0.3,
        5000
    )
    print("Creation successful with positional arguments.")
except Exception as e:
    print(f"Error with positional arguments: {e}")

# Check help
print("\nHelp for cv2.FaceDetectorYN.create:")
help(cv2.FaceDetectorYN.create)

import cv2
import os
import sys

model_dir = "models"
detect_model_path = os.path.join(model_dir, "face_detection_yunet.onnx")
recog_model_path = os.path.join(model_dir, "face_recognition_sface.onnx")

print("Initializing models...")
try:
    detector = cv2.FaceDetectorYN.create(detect_model_path, "", (640, 480))
    recognizer = cv2.FaceRecognizerSF.create(recog_model_path, "")
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(known_faces_dir, filename)
        print(f"Processing {filename}...")
        img = cv2.imread(filepath)
        if img is None:
            print(f"Failed to read {filename}")
            continue
        
        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)
        
        if faces is not None:
            print(f"Detected {len(faces)} faces in {filename}")
            aligned_face = recognizer.alignCrop(img, faces[0])
            feature = recognizer.feature(aligned_face)
            print(f"Extracted feature for {filename}")
        else:
            print(f"No faces detected in {filename}")
print("Diagnostics complete.")

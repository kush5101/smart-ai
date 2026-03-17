import sys
sys.path.append('.')
from face_attendance import FaceAttendanceSystem
import os

fs = FaceAttendanceSystem()
print(f"Files in known_faces: {os.listdir('known_faces')}")
print(f"Cache keys: {list(fs.face_images.keys())}")
print(f"Known face names: {fs.known_face_names}")
if not fs.face_images:
    print("CRITICAL: Cache is EMPTY!")
else:
    print("Cache has content.")

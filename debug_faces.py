import cv2
import os
import sys

# Mocking the name cleaning logic
def clean_name(raw_input):
    if not raw_input: return ""
    name = os.path.splitext(raw_input)[0] 
    name = name.replace("_", " ")
    name = "".join([c for c in name if not c.isdigit()])
    return name.strip().title()

faces_dir = r'c:\Users\sawan\Desktop\fire\smart_ai_monitoring\known_faces'
print(f"Scanning {faces_dir}...")

filenames = sorted(os.listdir(faces_dir))
name_counts = {}

for f in filenames:
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(faces_dir, f)
        img = cv2.imread(path)
        name = clean_name(f)
        status = "OK" if img is not None else "FAILED TO READ"
        shape = img.shape if img is not None else "N/A"
        print(f"File: {f} | Name: {name} | Status: {status} | Shape: {shape}")
        
        if img is not None:
            name_counts[name] = name_counts.get(name, 0) + 1

print("\nFinal Stats:")
for name, count in name_counts.items():
    print(f"- {name}: {count} samples")

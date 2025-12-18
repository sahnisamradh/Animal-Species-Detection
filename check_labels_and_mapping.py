import os
from collections import Counter
from ultralytics import YOLO
import cv2
import numpy as np

# CONFIG
LABELS_ROOT = "yolo_dataset_balanced/labels"  # or yolo_dataset/labels if you used that
IMAGES_ROOT = "yolo_dataset_balanced/images"
ANIMALS_YAML = "animals.yaml"  # for reference
SAMPLE_IMAGES_TO_SHOW = 5

# 1. gather class id stats across train/val/test
counts = Counter()
min_id, max_id = None, None
total_files = 0
for split in ["train","val","test"]:
    d = os.path.join(LABELS_ROOT, split)
    if not os.path.exists(d):
        continue
    for f in os.listdir(d):
        if not f.endswith(".txt"): continue
        total_files += 1
        with open(os.path.join(d,f)) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                counts[cid] += 1
                if min_id is None or cid < min_id: min_id = cid
                if max_id is None or cid > max_id: max_id = cid

print("Total label files scanned:", total_files)
print("Class id counts (sample):")
for k,v in sorted(counts.items()):
    print(f"  id {k}: {v}")
print(f"Min id: {min_id}, Max id: {max_id}")
print()

# 2. check animals.yaml names length
if os.path.exists(ANIMALS_YAML):
    with open(ANIMALS_YAML) as fh:
        yaml_txt = fh.read()
    print("animals.yaml preview (first 200 chars):")
    print(yaml_txt[:200].replace("\n"," | "))
    print()
else:
    print("animals.yaml not found at", ANIMALS_YAML)

# 3. issue detection
nc_declared = None
try:
    import yaml
    y = yaml.safe_load(open(ANIMALS_YAML))
    nc_declared = y.get("nc")
    names = y.get("names")
    print("Declared nc:", nc_declared)
    if names:
        print("Declared number of names:", len(names))
except Exception:
    pass

if min_id is not None:
    if min_id == 1:
        print("⚠️ WARNING: label IDs start at 1. YOLO expects 0..nc-1. This will shift classes by +1.")
    if max_id is not None and nc_declared is not None and max_id >= nc_declared:
        print(f"⚠️ WARNING: Found class id {max_id} >= declared nc {nc_declared}. That's invalid.")
print()

# 4. sample visual check of ground truth boxes (draw GT in red)
print("Saving sample GT visualizations to ./label_checks/")
os.makedirs("label_checks", exist_ok=True)
sample = 0
for split in ["train","val","test"]:
    imgdir = os.path.join(IMAGES_ROOT, split)
    lbldir = os.path.join(LABELS_ROOT, split)
    if not os.path.exists(imgdir) or not os.path.exists(lbldir):
        continue
    for f in os.listdir(lbldir):
        if sample >= SAMPLE_IMAGES_TO_SHOW: break
        if not f.endswith(".txt"): continue
        img_name = os.path.splitext(f)[0]
        found = None
        for ext in [".jpg",".jpeg",".png"]:
            p = os.path.join(imgdir, img_name+ext)
            if os.path.exists(p):
                found = p; break
        if not found: continue
        img = cv2.imread(found)
        h,w = img.shape[:2]
        with open(os.path.join(lbldir,f)) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) != 5: continue
                cid = int(float(parts[0])); a,b,c,d = map(float, parts[1:])
                # detect whether it's normalized (values <=1) or absolute pixels (big numbers)
                if a<=1 and b<=1 and c<=1 and d<=1:
                    # YOLO format: x_center y_center w h (normalized)
                    xc,yc,ww,hh = a,b,c,d
                    x1 = int((xc-ww/2)*w); y1 = int((yc-hh/2)*h)
                    x2 = int((xc+ww/2)*w); y2 = int((yc+hh/2)*h)
                else:
                    # assume x_min y_min x_max y_max in pixels
                    x1,y1,x2,y2 = int(a),int(b),int(c),int(d)
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(img,str(cid),(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
        outp = os.path.join("label_checks", f.replace(".txt",".jpg"))
        cv2.imwrite(outp, img)
        sample += 1
    if sample>=SAMPLE_IMAGES_TO_SHOW:
        break

print("Done. Open the images in ./label_checks to visually inspect GT boxes.")
print("If boxes are clearly around animals but the model predicts a different class, the likely issue is 0-based vs 1-based class IDs or mismatch with animals.yaml order.")

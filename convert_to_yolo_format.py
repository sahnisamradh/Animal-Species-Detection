import os
import cv2

# Root of YOLO dataset
root = "yolo_dataset"

# Go through each split
for split in ["train", "val", "test"]:
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)

    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        # Match image by same name
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            path = os.path.join(img_dir, lbl_file.replace(".txt", ext))
            if os.path.exists(path):
                image_path = path
                break

        if not image_path:
            print(f"⚠️ No matching image for {lbl_file}")
            continue

        # Read image to get size
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not read image {image_path}")
            continue
        h, w = img.shape[:2]

        # Read label lines
        with open(os.path.join(lbl_dir, lbl_file), "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x_min, y_min, x_max, y_max = map(float, parts)

            # Convert to YOLO normalized format
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            box_w = (x_max - x_min) / w
            box_h = (y_max - y_min) / h

            # Clamp to [0,1]
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            box_w = min(max(box_w, 0), 1)
            box_h = min(max(box_h, 0), 1)

            new_lines.append(f"{int(cls)} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

        # Overwrite file
        with open(os.path.join(lbl_dir, lbl_file), "w") as f:
            f.writelines(new_lines)

print("\n✅ All labels converted to YOLO normalized format!")

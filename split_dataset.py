import os, shutil, random
from collections import defaultdict

# ===== CONFIG =====
BASE_DIR = "yolo_dataset"
IMAGES_BASE = os.path.join(BASE_DIR, "images")
LABELS_BASE = os.path.join(BASE_DIR, "labels")
OUTPUT_BASE = "yolo_dataset_balanced"
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
# ==================

def build_file_map():
    """Group images by class ID across train/val/test."""
    class_to_files = defaultdict(list)

    for split in ["train", "val", "test"]:
        labels_dir = os.path.join(LABELS_BASE, split)
        if not os.path.exists(labels_dir):
            continue

        for lbl in os.listdir(labels_dir):
            if not lbl.endswith(".txt"):
                continue
            lbl_path = os.path.join(labels_dir, lbl)
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            if not lines:
                continue

            classes = {line.split()[0] for line in lines}
            for cls in classes:
                class_to_files[cls].append((split, lbl))
    return class_to_files

def split_dataset(class_to_files):
    """Create balanced train/val/test sets."""
    train_files, val_files, test_files = set(), set(), set()
    for cls, file_list in class_to_files.items():
        random.shuffle(file_list)
        n_total = len(file_list)
        n_val = max(1, int(n_total * VAL_RATIO))
        n_test = max(1, int(n_total * TEST_RATIO))

        val_files.update(file_list[:n_val])
        test_files.update(file_list[n_val:n_val+n_test])
        train_files.update(file_list[n_val+n_test:])
    return train_files, val_files, test_files

def copy_split(file_tuples, split_name):
    """Copy images and labels into the new split folders."""
    img_out = os.path.join(OUTPUT_BASE, "images", split_name)
    lbl_out = os.path.join(OUTPUT_BASE, "labels", split_name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for split, lbl_file in file_tuples:
        src_lbl = os.path.join(LABELS_BASE, split, lbl_file)
        shutil.copy(src_lbl, lbl_out)
        img_name = os.path.splitext(lbl_file)[0]
        for ext in [".jpg", ".jpeg", ".png"]:
            src_img = os.path.join(IMAGES_BASE, split, img_name + ext)
            if os.path.exists(src_img):
                shutil.copy(src_img, img_out)
                break

def main():
    mapping = build_file_map()
    train_files, val_files, test_files = split_dataset(mapping)

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")

    copy_split(train_files, "train")
    copy_split(val_files, "val")
    copy_split(test_files, "test")

if __name__ == "__main__":
    main()

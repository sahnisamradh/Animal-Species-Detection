import os

LABELS_ROOT = "yolo_dataset_balanced/labels"  # adjust if needed
for split in ["train","val","test"]:
    lbl_dir = os.path.join(LABELS_ROOT, split)
    if not os.path.exists(lbl_dir): continue
    for f in os.listdir(lbl_dir):
        if not f.endswith(".txt"): continue
        path = os.path.join(lbl_dir,f)
        with open(path) as fh:
            lines = fh.readlines()
        new = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            if cid >= 0:
                cid_new = cid - 1  # shift down
                if cid_new < 0:
                    raise ValueError(f"Negative id for {path}")
                new.append(f"{cid_new} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                changed = True
            else:
                new.append(line)
        if changed:
            with open(path,"w") as fh:
                fh.writelines(new)
print("Done. All labels remapped to 0-based indices (cid-1).")

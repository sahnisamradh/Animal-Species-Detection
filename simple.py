# evaluate_model.py

from ultralytics import YOLO
import torch

# ==============================================================
# 1️⃣ AUTO DEVICE DETECTION
# ==============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n🚀 Evaluating on device: {device.upper()}\n")

# ==============================================================
# 2️⃣ LOAD TRAINED MODEL
# ==============================================================
MODEL_PATH = "animal_training_fast_final/yolov8n_fast_clean_mapped/weights/best.pt"
model = YOLO(MODEL_PATH)

# ==============================================================
# 3️⃣ RUN VALIDATION
# ==============================================================
results = model.val(
    data="animals.yaml",
    imgsz=512,
    batch=16,
    device=device,
    conf=0.25,
    iou=0.6,
    save_json=True,
    verbose=True
)

# ==============================================================
# 4️⃣ SUMMARY METRICS (F1 FIXED)
# ==============================================================
precision = float(results.box.mp)
recall = float(results.box.mr)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

print("\n================== 📊 MODEL PERFORMANCE SUMMARY ==================\n")
print(f"✔️ Precision:    {precision:.4f}")
print(f"✔️ Recall:       {recall:.4f}")
print(f"✔️ F1-Score:     {f1_score:.4f}")
print(f"✔️ mAP@50:       {results.box.map50:.4f}")
print(f"✔️ mAP@50-95:    {results.box.map:.4f}\n")

# ==============================================================
# 5️⃣ CLASS-WISE RESULTS (SAFE FOR MISSING CLASSES)
# ==============================================================
print("==================== 📌 CLASS-WISE METRICS =======================\n")

ap50_list = results.box.ap50
ap_list = results.box.ap

for cls_id, cls_name in results.names.items():

    # Safe AP50 & AP (handle missing classes)
    ap50 = ap50_list[cls_id] if cls_id < len(ap50_list) else 0.0
    ap = ap_list[cls_id] if cls_id < len(ap_list) else 0.0

    print(f"Class: {cls_name}")
    print(f" - AP50:      {ap50:.4f}")
    print(f" - AP50-95:   {ap:.4f}")
    print("--------------------------------------------------")

# ==============================================================
# 6️⃣ OUTPUT PATHS
# ==============================================================
print("\n📁 Confusion Matrix saved at:")
print("   runs/detect/val/confusion_matrix.png")

print("\n📁 Detailed COCO-style results saved at:")
print("   runs/detect/val/coco_eval.json")

print("\n🎯 Evaluation complete!\n")

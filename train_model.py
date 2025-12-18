from ultralytics import YOLO
import torch

# ==============================================================
# 1️⃣ AUTO DEVICE DETECTION
# ==============================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Training on device: {device.upper()}")

# ==============================================================
# 2️⃣ LOAD MODEL
# ==============================================================
# Using YOLOv8 Nano (fastest)
# You can switch to 'yolov8n6.pt' (slightly larger) for +3% accuracy
model = YOLO('yolov8n.pt')

# ==============================================================
# 3️⃣ TRAINING CONFIGURATION (optimized for CPU/GPU speed)
# ==============================================================
model.train(
    data='animals.yaml',          # Path to dataset YAML
    epochs=45,                    # Slightly longer for smoother convergence
    imgsz=512,                    # Image size (keeps training fast)
    batch=16,                     # Larger batch for efficiency
    lr0=0.0015,                   # Balanced LR (avoids overshoot)
    lrf=0.01,                     # Final LR multiplier (for smoother decay)
    optimizer='SGD',              # Faster on small models
    momentum=0.937,               # Default YOLO momentum
    weight_decay=0.0005,          # Regularization
    pretrained=True,              # Start from COCO pretrained
    cache=True,                   # Cache images for faster epochs
    patience=10,                  # Early stop if no improvement
    device=device,

    # ==== AUGMENTATIONS (balanced for Nano model) ====
    augment=True,
    hsv_h=0.04, hsv_s=0.6, hsv_v=0.4,    # color augmentations
    degrees=8, translate=0.15, scale=0.5, shear=0.05,  # geometric augments
    mosaic=0.7, mixup=0.15,              # strong but safe
    flipud=0.3, fliplr=0.5,              # random flips
    perspective=0.0005,                  # light perspective shift
    copy_paste=0.05,                     # adds object-level variety

    # ==== PROJECT INFO ====
    project='animal_training_fast_final',
    name='yolov8n_fast_clean_mapped',
    verbose=True
)

# ==============================================================
# 4️⃣ VALIDATION METRICS
# ==============================================================
metrics = model.val()
print("\n📊 Final Validation Metrics:")
print(metrics)

# ==============================================================
# 5️⃣ EXPORT BEST MODEL
# ==============================================================
best_model = 'animal_training_fast_final/yolov8n_fast_clean_mapped/weights/best.pt'
print(f"\n✅ Training complete! Best model saved at:\n{best_model}")

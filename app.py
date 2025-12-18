import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import numpy as np
import skfuzzy as fuzz
import json

# ======================================================
# Load trained YOLOv8 model
# ======================================================
MODEL_PATH = "animal_training_fast_final/yolov8n_fast_clean_mapped/weights/best.pt"
model = YOLO(MODEL_PATH)

# ======================================================
# Load Animal Knowledge Base (JSON)
# ======================================================
with open("animal_data.json", "r") as f:
    animal_data = json.load(f)

# ======================================================
# Fuzzy Logic Setup
# ======================================================
x_danger = np.arange(0, 101, 1)
low = fuzz.trimf(x_danger, [0, 0, 40])
medium = fuzz.trimf(x_danger, [30, 50, 70])
high = fuzz.trapmf(x_danger, [60, 75, 100, 100])

animal_categories = {
    'Zebra': 'medium', 'Tiger': 'high', 'Rhinoceros': 'high', 'Ostrich': 'medium',
    'Lion': 'high', 'Leopard': 'high', 'Horse': 'medium', 'Jaguar': 'high',
    'Harbor seal': 'medium', 'Goat': 'low', 'Giraffe': 'medium', 'Fox': 'medium',
    'Elephant': 'high', 'Eagle': 'medium', 'Deer': 'low', 'Crab': 'low',
    'Chicken': 'low', 'Caterpillar': 'low', 'Cheetah': 'high', 'Butterfly': 'low'
}

def compute_danger_level(animal_name: str) -> float:
    """Compute crisp danger level using fuzzy sets."""
    animal_name = animal_name.strip().title()
    if animal_name not in animal_categories:
        return 0.0
    category = animal_categories[animal_name]
    if category == 'low':
        crisp_value = 25
    elif category == 'medium':
        crisp_value = 55
    elif category == 'high':
        crisp_value = 85
    else:
        crisp_value = 0
    return round(crisp_value, 2)

# ======================================================
# CSP-Like Knowledge Constraint Satisfaction
# ======================================================
def infer_animal_details(animal_name):
    """
    Fetch data from knowledge base and satisfy info constraints:
    Each detected animal must yield a consistent info tuple:
    (name, habitat, diet, conservation_status, danger, interesting_fact)
    """
    animal_name = animal_name.strip().title()
    if animal_name not in animal_data:
        return {
            "error": f"No data found for {animal_name}",
            "inferred": False
        }

    data = animal_data[animal_name]
    danger = compute_danger_level(animal_name)

    # CSP constraint satisfaction: All key facts must exist
    constraints_satisfied = all(
        key in data for key in [
            "scientific_name", "habitat", "diet", "conservation_status", "behavior"
        ]
    )

    return {
        "name": animal_name,
        "scientific_name": data.get("scientific_name", "Unknown"),
        "habitat": data.get("habitat", "Unknown"),
        "diet": data.get("diet", "Unknown"),
        "lifespan": data.get("average_lifespan", "Unknown"),
        "behavior": data.get("behavior", "Unknown"),
        "status": data.get("conservation_status", "Unknown"),
        "fact": data.get("interesting_fact", "N/A"),
        "danger_level": danger,
        "constraints_satisfied": constraints_satisfied
    }

# ======================================================
# Streamlit App UI
# ======================================================
st.set_page_config(page_title="Animal Species Detector (Fuzzy CSP)", layout="wide")
st.title("🐾 Intelligent Animal Species Detection (Fuzzy + CSP)")
st.write("Upload an **image** or **video**, and the system will detect animals, infer knowledge, and reason about their danger levels using fuzzy logic.")

# Sidebar
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.35, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (Overlap)", 0.1, 1.0, 0.45, 0.05)

uploaded_file = st.file_uploader("📁 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

# ======================================================
# Display Animal Knowledge Card
# ======================================================
def display_animal_card(animal_info):
    st.markdown(f"### 🦓 **{animal_info['name']}**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Scientific Name:** {animal_info['scientific_name']}")
        st.write(f"**Habitat:** {animal_info['habitat']}")
        st.write(f"**Diet:** {animal_info['diet']}")
        st.write(f"**Lifespan:** {animal_info['lifespan']}")
        st.write(f"**Behavior:** {animal_info['behavior']}")
    with col2:
        st.write(f"**Conservation Status:** {animal_info['status']}")
        st.progress(int(animal_info['danger_level']))
        st.write(f"🧠 **Danger Level:** {animal_info['danger_level']}%")
        if animal_info['danger_level'] >= 70:
            st.warning("⚠️ High Risk Animal – Maintain Distance!")
        elif animal_info['danger_level'] >= 40:
            st.info("🟠 Medium Risk Animal – Exercise Caution.")
        else:
            st.success("🟢 Low Risk Animal – Generally Harmless.")
    st.info(f"**Fun Fact:** {animal_info['fact']}")
    st.markdown("---")

# ======================================================
# Function: process and display image
# ======================================================
def process_image(image_path):
    st.subheader("🔍 Detection Result (Image)")
    results = model.predict(source=image_path, conf=conf_threshold, iou=iou_threshold, save=False)
    annotated_frame = results[0].plot()

    detected_animals = []
    if results and len(results[0].boxes) > 0:
        for c in results[0].boxes.cls:
            label = model.names[int(c)]
            detected_animals.append(label)

    st.image(annotated_frame, caption="Detected Animals", use_container_width=True)

    if detected_animals:
        st.subheader("🧩 Knowledge Inference (Fuzzy + CSP)")
        for animal in detected_animals:
            info = infer_animal_details(animal)
            display_animal_card(info)
    else:
        st.warning("No animals detected in the image.")

# ======================================================
# Function: process and display video
# ======================================================
def process_video(video_path):
    st.subheader("🎬 Processing Video... Please wait")

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_out.name

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    detected_species = set()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
        annotated_frame = results[0].plot()

        if results and len(results[0].boxes) > 0:
            for c in results[0].boxes.cls:
                label = model.names[int(c)]
                detected_species.add(label)

        out.write(annotated_frame)
        frame_idx += 1
        progress.progress(min(frame_idx / frame_count, 1.0))

    cap.release()
    out.release()
    st.success("✅ Video processing complete!")
    st.video(output_path)

    if detected_species:
        st.subheader("🧩 Knowledge Inference (Fuzzy + CSP)")
        for animal in detected_species:
            info = infer_animal_details(animal)
            display_animal_card(info)
    else:
        st.warning("No animals detected in the video.")

    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download Processed Video", data=f, file_name="detected_animals.mp4")

# ======================================================
# Main Logic
# ======================================================
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
    temp_input.write(uploaded_file.read())
    temp_input.close()

    if file_ext in ["jpg", "jpeg", "png"]:
        process_image(temp_input.name)
    elif file_ext in ["mp4", "mov", "avi"]:
        process_video(temp_input.name)
    else:
        st.error("Unsupported file type! Please upload JPG, PNG, or MP4 video.")

import streamlit as st
from ultralytics import YOLO
from PIL import Image

# ----------------------------------------
# Streamlit page title
# ----------------------------------------
st.title("🚨 Drowsiness / Phone-Use Detection (Image Upload)")

# ----------------------------------------
# Load YOLOv8 model (cached for speed)
# ----------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # keep best.pt in the same folder
model = load_model()
class_names = model.names

# ----------------------------------------
# File uploader
# ----------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Display the uploaded image first
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ----------------------------------------
    # Run YOLO inference
    # ----------------------------------------
    with st.spinner("Detecting..."):
        results = model(img)

    # ----------------------------------------
    # Show annotated image
    # ----------------------------------------
    st.subheader("Detections")
    st.image(results[0].plot(), caption="Detected Objects", use_column_width=True)

    # ----------------------------------------
    # Detailed detection info & simple alerts
    # ----------------------------------------
    detected_labels = []
    for r in results:
        for c in r.boxes.cls:
            label = class_names[int(c)]
            detected_labels.append(label)

    st.write("**Detected classes:**", ", ".join(detected_labels) if detected_labels else "None")

    # Optional alerts similar to your webcam logic
    if "drowsy" in [lbl.lower() for lbl in detected_labels]:
        st.error("😴 Drowsiness detected!")
    if "phone" in [lbl.lower() for lbl in detected_labels]:
        st.warning("📱 Phone use detected!")
    if not detected_labels:
        st.info("No relevant detections found.")

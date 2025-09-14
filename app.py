import streamlit as st
import cv2
import time
from ultralytics import YOLO
from pathlib import Path

# ---------------------------
# Load YOLO model once
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
class_names = model.names

# ---------------------------
# UI: Title and Start Button
# ---------------------------
st.title("🚨 Drowsiness / Distraction Detector")
start_cam = st.checkbox("Start Webcam")

# Thresholds
DROWSY_FRAME_THRESHOLD = 30        # 30 frames of drowsy
PHONE_TIME_THRESHOLD = 5           # 5 seconds of phone use
DISTRACTION_FRAME_THRESHOLD = 300  # 300 frames no face

# ---------------------------
# Session Counters
# ---------------------------
if "sleep_counter" not in st.session_state:
    st.session_state.sleep_counter = 0
if "phone_start_time" not in st.session_state:
    st.session_state.phone_start_time = None
if "distraction_counter" not in st.session_state:
    st.session_state.distraction_counter = 0

FRAME_WINDOW = st.image([])        # webcam display
alert_placeholder = st.empty()     # alert messages

# ---------------------------
# Webcam Loop
# ---------------------------
if start_cam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam.")
    else:
        while True:
            success, frame = cap.read()
            if not success:
                st.error("Failed to capture image.")
                break

            # YOLO detection
            results = model(frame)
            phone_detected = drowsy_detected = awake_detected = False

            for r in results:
                filtered_boxes = []
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    label = class_names[class_id].lower()
                    if label == "smoking":
                        continue  # skip smoking
                    filtered_boxes.append(box)

                    conf = float(box.conf[0])
                    if label == "drowsy" and conf > 0.4:
                        drowsy_detected = True
                    elif label == "awake" and conf > 0.5:
                        awake_detected = True
                    elif label == "phone" and conf > 0.3:
                        phone_detected = True
                r.boxes = filtered_boxes
                annotated_frame = r.plot()

            # --- Drowsiness ---
            if drowsy_detected:
                st.session_state.sleep_counter += 1
                if st.session_state.sleep_counter >= DROWSY_FRAME_THRESHOLD:
                    alert_placeholder.error("😴 Drowsy for 30 frames! Alarm 1")
                    st.audio(str(Path("alarm_sounds/Alarm_drowsy.mp3")))
                    time.sleep(4)
            else:
                st.session_state.sleep_counter = 0

            # --- Phone usage ---
            if phone_detected:
                if st.session_state.phone_start_time is None:
                    st.session_state.phone_start_time = time.time()
                elif time.time() - st.session_state.phone_start_time >= PHONE_TIME_THRESHOLD:
                    alert_placeholder.error("📱 Phone detected for 5+ sec! Alarm 2")
                    st.audio(str(Path("alarm_sounds/Alarm_phone.wav")))
                    st.session_state.phone_start_time = None
            else:
                st.session_state.phone_start_time = None

            # --- Distraction ---
            if (st.session_state.phone_start_time is None
                and not awake_detected and not drowsy_detected):
                st.session_state.distraction_counter += 1
                if st.session_state.distraction_counter >= DISTRACTION_FRAME_THRESHOLD:
                    alert_placeholder.error("🙈 Distracted for 5 sec! Alarm 3")
                    st.audio(str(Path("alarm_sounds/Alarm_distracted.wav")))
                    st.session_state.distraction_counter = 0
            else:
                st.session_state.distraction_counter = 0

            # Display live annotated video in Streamlit
            FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

else:
    st.info("✅ Tick **Start Webcam** to begin detection.")

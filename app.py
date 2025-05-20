import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import pandas as pd
import tempfile
import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import threading

# App configuration
st.set_page_config(
    page_title="AI Surveillance: Animal & Object Detection",
    page_icon="🐾",
    layout="wide"
)

# Load YOLOv8 model (cached)
@st.cache_resource
def load_model():
    try:
        model = YOLO('yolov8n.pt')  # Load pretrained model
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Email alert function
def send_email_alert(detected_objects, recipient):
    sender = "your_email@example.com"
    password = "your_email_password"
    
    msg = MIMEText(f"ALERT: The following objects were detected: {', '.join(detected_objects)}")
    msg['Subject'] = "Security Alert: Object Detection"
    msg['From'] = sender
    msg['To'] = recipient
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.send_message(msg)
        st.success("Email alert sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Detection function
def run_detection(model, frame, confidence_threshold, classes_to_detect):
    results = model(frame, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            if conf >= confidence_threshold and (label in classes_to_detect or not classes_to_detect):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, detections

# Main application
def main():
    st.title("🐾 AI-Powered Animal & Object Detection")
    st.markdown("Real-time detection with alert system")
    
    # Initialize session state
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'alert_triggered' not in st.session_state:
        st.session_state.alert_triggered = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("Configuration")
        
        # Input source selection
        input_source = st.radio(
            "Input Source",
            ["Webcam", "Video File", "RTSP Stream"],
            index=0
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1, max_value=1.0, value=0.5, step=0.05
        )
        
        # Classes to detect (common animals and objects)
        class_options = [
            'person', 'bicycle', 'car', 'motorcycle', 'dog', 'cat', 
            'bird', 'horse', 'sheep', 'cow', 'bear', 'elephant', 
            'zebra', 'giraffe'
        ]
        classes_to_detect = st.multiselect(
            "Classes to Detect (empty for all)",
            class_options,
            default=['dog', 'cat', 'person']
        )
        
        # Alert settings
        st.subheader("Alert Settings")
        enable_visual_alert = st.checkbox("Visual Alert", True)
        enable_sound_alert = st.checkbox("Sound Alert", False)
        enable_email_alert = st.checkbox("Email Alert", False)
        
        if enable_email_alert:
            email_recipient = st.text_input("Recipient Email")
            email_threshold = st.slider(
                "Email Alert Confidence Threshold",
                min_value=0.1, max_value=1.0, value=0.7, step=0.05
            )
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Detection")
        frame_placeholder = st.empty()
        
        # Start/stop button
        if st.button("Start Detection" if not st.session_state.detection_active else "Stop Detection"):
            st.session_state.detection_active = not st.session_state.detection_active
        
        # Input source handling
        if input_source == "Webcam":
            video_capture = cv2.VideoCapture(0)
        elif input_source == "Video File":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
            if uploaded_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                video_capture = cv2.VideoCapture(tfile.name)
            else:
                video_capture = None
        else:  # RTSP Stream
            rtsp_url = st.text_input("RTSP Stream URL", "rtsp://username:password@ip_address:port")
            video_capture = cv2.VideoCapture(rtsp_url)
        
        # Detection loop
        if st.session_state.detection_active and video_capture is not None:
            while st.session_state.detection_active and video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Run detection
                processed_frame, detections = run_detection(
                    model, frame, confidence_threshold, classes_to_detect
                )
                
                # Update detection history
                for detection in detections:
                    st.session_state.detection_history.append(detection)
                
                # Trigger alerts if any detections
                if detections:
                    st.session_state.alert_triggered = True
                    
                    # Visual alert
                    if enable_visual_alert:
                        processed_frame = cv2.copyMakeBorder(
                            processed_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 255]
                        )
                    
                    # Sound alert (browser-based)
                    if enable_sound_alert:
                        st.write('<audio autoplay><source src="https://www.soundjay.com/buttons/sounds/beep-07.mp3" type="audio/mpeg"></audio>', 
                                unsafe_allow_html=True)
                    
                    # Email alert for high confidence detections
                    if enable_email_alert and email_recipient:
                        high_conf_detections = [
                            d['label'] for d in detections 
                            if d['confidence'] >= email_threshold
                        ]
                        if high_conf_detections:
                            threading.Thread(
                                target=send_email_alert,
                                args=(high_conf_detections, email_recipient)
                            ).start()
                
                # Display processed frame
                frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                
                # Small delay to prevent high CPU usage
                time.sleep(0.05)
            
            video_capture.release()
    
    with col2:
        st.subheader("Detection Log")
        
        if st.session_state.detection_history:
            # Convert to DataFrame for nice display
            df = pd.DataFrame(st.session_state.detection_history)
            st.dataframe(df.sort_values("time", ascending=False))
            
            # Alert status
            if st.session_state.alert_triggered:
                st.error("ALERT: Objects detected!")
            else:
                st.success("No recent alerts")
            
            # Clear log button
            if st.button("Clear Log"):
                st.session_state.detection_history = []
                st.session_state.alert_triggered = False
                st.rerun()
        else:
            st.info("No detections yet")

if __name__ == "__main__":
    main()
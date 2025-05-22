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
from streamlit.components.v1 import html
from streamlit import toast

# App configuration
st.set_page_config(
    page_title="AI Surveillance: Animal & Object Detection",
    page_icon="🐾",
    layout="wide"
)

# ================== DEFINITIONS ==================

class AlertSystem:
    """
    Handles all alert types (visual, audio, email) for the detection system.
    
    Attributes:
        visual_enabled (bool): Toggle for visual alerts
        audio_enabled (bool): Toggle for audio alerts
        email_enabled (bool): Toggle for email alerts
        email_recipient (str): Email address to receive alerts
        email_threshold (float): Confidence threshold for email alerts
        last_email_time (float): Timestamp of last email sent
        email_cooldown (int): Minimum seconds between emails
    """
    
    def __init__(self):
        self.visual_enabled = True
        self.audio_enabled = False
        self.email_enabled = False
        self.email_recipient = ""
        self.email_threshold = 0.7
        self.last_email_time = 0
        self.email_cooldown = 60  # 1 minute cooldown
    
    def visual_alert(self, frame):
        """Adds red border to frame for visual alert"""
        return cv2.copyMakeBorder(frame, 10, 10, 10, 10, 
                                 cv2.BORDER_CONSTANT, value=[0, 0, 255])
    
    def audio_alert(self):
        """Triggers browser-based audio alert"""
        html_str = """
        <audio autoplay>
            <source src="https://www.soundjay.com/buttons/sounds/beep-07.mp3" type="audio/mpeg">
        </audio>
        """
        html(html_str, height=0)
    
    def email_alert(self, detections):
        """
        Sends email alert if conditions are met.
        
        Args:
            detections (list): List of detection dictionaries
            
        Returns:
            bool: True if email was sent, False otherwise
        """
        if not self.email_enabled or not self.email_recipient:
            return False
            
        current_time = time.time()
        if current_time - self.last_email_time < self.email_cooldown:
            return False
            
        # Filter high confidence detections
        high_conf_detections = [
            f"{d['label']} ({d['confidence']:.0%})" 
            for d in detections 
            if d['confidence'] >= self.email_threshold
        ]
        
        if not high_conf_detections:
            return False
            
        if send_email_alert(high_conf_detections, self.email_recipient):
            self.last_email_time = current_time
            return True
        return False

# ================== MAIN CODE ==================

# Load YOLOv8 model (cached)
@st.cache_resource
def load_model():
    """Loads and caches the YOLOv8 model"""
    try:
        model = YOLO('yolov8n.pt')  # Load pretrained model
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def send_email_alert(detected_objects, recipient):
    """Sends email alert with detection details"""
    try:
        # Get credentials from Streamlit secrets
        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
        smtp_server = st.secrets["email"].get("smtp_server", "smtp.gmail.com")
        smtp_port = st.secrets["email"].get("smtp_port", 465)
        
        # Create message
        msg = MIMEText(
            f"ALERT: The following objects were detected:\n\n"
            f"{chr(10).join(detected_objects)}\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        msg['Subject'] = "🚨 Security Alert: Objects Detected"
        msg['From'] = sender
        msg['To'] = recipient
        
        # Send email
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender, password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {str(e)}")
        return False

def run_detection(model, frame, confidence_threshold, classes_to_detect):
    """
    Runs object detection on a frame.
    
    Args:
        model: YOLOv8 model
        frame: Input image frame
        confidence_threshold: Minimum confidence score
        classes_to_detect: List of class names to detect
        
    Returns:
        tuple: (processed_frame, detections)
    """
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

def main():
    """Main application function"""
    st.title("🐾 AI-Powered Animal & Object Detection")
    st.markdown("Real-time detection with alert system")
    
    # Initialize session state
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'alert_triggered' not in st.session_state:
        st.session_state.alert_triggered = False
    
    # Initialize Alert System
    alert_system = AlertSystem()
    
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
        
        # Classes to detect
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
        alert_system.visual_enabled = st.checkbox("Visual Alert", True)
        alert_system.audio_enabled = st.checkbox("Sound Alert", False)
        alert_system.email_enabled = st.checkbox("Email Alert", False)
        
        if alert_system.email_enabled:
            alert_system.email_recipient = st.text_input("Recipient Email")
            alert_system.email_threshold = st.slider(
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
                    if alert_system.visual_enabled:
                        processed_frame = alert_system.visual_alert(processed_frame)
                    
                    # Audio alert
                    if alert_system.audio_enabled:
                        alert_system.audio_alert()
                    
                    # Email alert
                    if alert_system.email_alert(detections):
                        st.toast("Email alert sent!", icon="✉️")
                
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

import os
import requests
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time

# Configuration
SLEEP_TIMEOUT = 300  # 5 minutes in seconds
CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "coco.names"
WEIGHTS_URL = "https://github.com/KarunakarMalkagalla/RealTimeObjectDetection/releases/download/v1.0.0/yolov3.weights"

# Initialize session state
if 'last_activity_time' not in st.session_state:
    st.session_state.last_activity_time = time.time()
if 'is_asleep' not in st.session_state:
    st.session_state.is_asleep = False

# Function to download YOLO weights
@st.cache_resource
def download_yolo_weights():
    if not os.path.exists(WEIGHTS_PATH):
        with st.spinner("Downloading YOLO weights (200+ MB, this may take a while)..."):
            response = requests.get(WEIGHTS_URL, stream=True)
            response.raise_for_status()
            with open(WEIGHTS_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

# Model loading function with error handling
@st.cache_resource(ttl=3600)  # Refresh every hour
def load_yolo_model():
    try:
        net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(CLASSES_PATH, 'r') as f:
            classes = f.read().strip().split('\n')
        return net, classes
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

# Download weights first
download_yolo_weights()

# Sleep mode management
current_time = time.time()
if not st.session_state.is_asleep and (current_time - st.session_state.last_activity_time) > SLEEP_TIMEOUT:
    st.session_state.is_asleep = True

if st.session_state.is_asleep:
    # Clear resources
    if 'net' in st.session_state:
        del st.session_state.net
    if 'classes' in st.session_state:
        del st.session_state.classes
    
    # Sleep mode UI
    st.markdown('<div style="text-align: center; font-size: 24px; color: gray;">💤 App in Sleep Mode</div>', unsafe_allow_html=True)
    st.write("_To conserve resources, the app is sleeping. Click below to wake up._")
    
    if st.button("Wake Up"):
        st.session_state.last_activity_time = time.time()
        st.session_state.is_asleep = False
        st.rerun()
    
    st.stop()
else:
    # Update activity time
    st.session_state.last_activity_time = current_time
    # Load model
    try:
        net, classes = load_yolo_model()
        st.session_state.net = net
        st.session_state.classes = classes
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.stop()

# Detection function with error handling
def detect_objects(image, confidence_threshold=0.5, nms_threshold=0.4):
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(image_bgr, 1/255, (416, 416), (0, 0, 0), True, crop=False)
        st.session_state.net.setInput(blob)
        
        layer_names = st.session_state.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in st.session_state.net.getUnconnectedOutLayers()]
        detections = st.session_state.net.forward(output_layers)
        
        height, width = image.shape[:2]
        boxes, confidences, class_ids = [], [], []
        
        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        results = []
        if indices is not None:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                results.append({
                    'box': [x, y, w, h],
                    'confidence': confidences[i],
                    'class': st.session_state.classes[class_ids[i]]
                })
        return results
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return []

# UI Components
st.set_page_config(page_title="Object Detection", page_icon=":eye:", layout="centered")

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #2c3e50;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(145deg, #f0f4f8, #dfe6ec);
    }
    .footer {
        text-align: center;
        margin-top: 2em;
        color: #7f8c8d;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🔍 Smart Object Detector</div>', unsafe_allow_html=True)

# Controls
col1, col2 = st.columns(2)
with col1:
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
with col2:
    nms_threshold = st.slider("NMS Threshold", 0.1, 1.0, 0.4, 0.05)

# Image processing
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        with st.spinner("Analyzing image..."):
            detections = detect_objects(img_array, confidence_threshold, nms_threshold)
            
        if detections:
            # Draw bounding boxes
            output_image = img_array.copy()
            for obj in detections:
                x, y, w, h = obj['box']
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{obj['class']}: {obj['confidence']:.2f}"
                cv2.putText(output_image, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            st.image(output_image, caption="Detection Results", use_column_width=True)
            
            # Display summary
            counts = {}
            for obj in detections:
                counts[obj['class']] = counts.get(obj['class'], 0) + 1
            summary = "Detected: " + ", ".join([f"{v} {k}s" for k, v in counts.items()])
            st.success(f"✅ {summary}")
        else:
            st.warning("No objects detected with current settings")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Health status
last_active = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.last_activity_time))
st.markdown(f'<div class="footer">Last activity: {last_active}</div>', unsafe_allow_html=True)

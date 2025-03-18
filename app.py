import os
import requests
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import time  # For activity tracking

# Streamlit app configuration
st.set_page_config(
    page_title="Real-Time Object Detection App",
    page_icon=":guardsman:",
    layout="centered"
)

# Constants
SLEEP_TIMEOUT = 300  # 5 minutes in seconds
CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "coco.names"
WEIGHTS_URL = "https://github.com/KarunakarMalkagalla/RealTimeObjectDetection/releases/download/v1.0.0/yolov3.weights"

# Initialize session state variables
if 'last_activity_time' not in st.session_state:
    st.session_state.last_activity_time = time.time()
if 'is_asleep' not in st.session_state:
    st.session_state.is_asleep = False

# Function to download the file
def download_file(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        f.write(response.content)

# Function to load YOLO model
def load_yolo_model(config_path, weights_path, classes_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return net, classes

# Function to detect objects using YOLO
def detect_objects(image, net, classes, confidence_threshold=0.7, nms_threshold=0.4):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(image_bgr, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    detections = net.forward(output_layers)
    height, width = image.shape[:2]

    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    result = []
    if indices is not None:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            result.append({
                'box': [x, y, w, h],
                'confidence': confidences[i],
                'class_id': class_ids[i],
                'class_name': classes[class_ids[i]]
            })
    return result, confidences

# Function to generate description
def generate_description(detections, confidences):
    object_count = {}
    for detection in detections:
        class_name = detection['class_name']
        object_count[class_name] = object_count.get(class_name, 0) + 1

    description = "The image contains the following objects: "
    for class_name, count in object_count.items():
        description += f"{count} {class_name}(s), "

    if confidences:
        overall_confidence = np.mean(confidences)
    else:
        overall_confidence = 0

    description = description.rstrip(", ") + "."
    description += f" Overall confidence of detection: {overall_confidence * 100:.2f}%."
    return description

# Download weights if not present
if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("Downloading YOLO weights (this may take several minutes)..."):
        download_file(WEIGHTS_URL, WEIGHTS_PATH)

# Check sleep mode status
current_time = time.time()
if not st.session_state.is_asleep and (current_time - st.session_state.last_activity_time) > SLEEP_TIMEOUT:
    st.session_state.is_asleep = True

# Handle sleep mode display and wake up logic
if st.session_state.is_asleep:
    # Clear model from memory
    if 'net' in st.session_state:
        del st.session_state.net
    if 'classes' in st.session_state:
        del st.session_state.classes

    # Sleep mode interface
    st.markdown('<div style="text-align: center; font-size: 24px; color: gray;">💤 App is in Sleep Mode</div>', unsafe_allow_html=True)
    st.write("_To conserve resources, the app has entered sleep mode. Click the button below to wake it up._")
    
    if st.button("Wake Up"):
        st.session_state.last_activity_time = time.time()
        st.session_state.is_asleep = False
        st.rerun()
    
    st.stop()
else:
    # Update last activity time
    st.session_state.last_activity_time = current_time

    # Load model if not loaded
    if 'net' not in st.session_state or 'classes' not in st.session_state:
        with st.spinner("Loading object detection model..."):
            net, classes = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH, CLASSES_PATH)
            st.session_state.net = net
            st.session_state.classes = classes

# Main application interface
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: black;
        font-weight: bold;
        background-color: #90EE90;
        padding: 20px;
        border-radius: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        color: gray;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Real-Time Object Detection App</div>', unsafe_allow_html=True)
st.write("Upload an image to detect objects using YOLOv3. Below is the object detection result.")

# Add user control sliders
confidence_threshold = st.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Adjust the minimum confidence required for object detection"
)

nms_threshold = st.slider(
    "NMS Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Adjust the threshold for non-maximum suppression"
)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Analyzing image..."):
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display uploaded image
        st.image(image_np, caption="Uploaded Image", use_container_width=True)

        # Detect objects
        detections, confidences = detect_objects(
            image_np,
            st.session_state.net,
            st.session_state.classes,
            confidence_threshold,
            nms_threshold
        )

        # Generate and display description
        if detections:
            description = generate_description(detections, confidences)
            st.subheader("Detection Results")
            st.success(description)
            
            # Draw bounding boxes
            img_with_boxes = image_np.copy()
            for detection in detections:
                x, y, w, h = detection['box']
                cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{detection['class_name']}: {detection['confidence']:.2f}"
                cv2.putText(img_with_boxes, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            st.image(img_with_boxes, caption="Detected Objects", use_container_width=True)
        else:
            st.warning("No objects detected with current confidence threshold.")

# Add footer with activity status
last_active = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.last_activity_time))
st.markdown(f'<div class="footer">Last activity: {last_active}</div>', unsafe_allow_html=True)

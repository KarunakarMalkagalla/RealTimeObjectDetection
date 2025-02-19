import os
import requests
import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Streamlit app styling (moved to the correct position)
st.set_page_config(page_title="Real-Time Object Detection App", page_icon=":guardsman:", layout="centered")

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

# File paths
CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "coco.names"
# Download weights if not present
WEIGHTS_URL = "https://github.com/KarunakarMalkagalla/RealTimeObjectDetection/releases/download/v1.0.0/yolov3.weights"
if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("Downloading YOLO weights..."):
        download_file(WEIGHTS_URL, WEIGHTS_PATH)

# Load model and classes
net, classes = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH, CLASSES_PATH)

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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Real-Time Object Detection App</div>', unsafe_allow_html=True)
st.write("Upload an image to detect objects using YOLOv3. Below is the object detection result.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Detecting objects..."):
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display the uploaded image
        st.image(image_np, caption="Uploaded Image", use_container_width=True)

        # Detect objects
        detections, confidences = detect_objects(image_np, net, classes)

        # Generate and display description
        description = generate_description(detections, confidences)
        st.subheader("Description of the Image")
        st.write(description)

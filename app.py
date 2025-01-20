import os
import requests
import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Streamlit app styling
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

# Optimized function to detect objects using YOLO
def detect_objects(image, net, classes, confidence_threshold=0.5, nms_threshold=0.3):
    # Convert image to BGR for OpenCV processing
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(image_bgr, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass to get predictions
    detections = net.forward(output_layers)
    height, width = image.shape[:2]

    # Initialize lists for storing detection results
    boxes, confidences, class_ids = [], [], []
    for output in detections:
        for detection in output:
            scores = detection[5:]  # Class probabilities
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                # Get bounding box coordinates
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
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

# Function to generate perception-based description
def generate_perception_description(detections, image):
    description = "The scene appears to have the following elements: "

    object_count = {}
    for detection in detections:
        class_name = detection['class_name']
        object_count[class_name] = object_count.get(class_name, 0) + 1

    if object_count:
        for class_name, count in object_count.items():
            description += f"{count} {class_name}(s), "
        description = description.rstrip(", ") + ". "
    else:
        description += "No specific objects were detected. "

    # Add image properties
    height, width, _ = image.shape
    if height > width:
        description += "The image appears to have a portrait orientation. "
    else:
        description += "The image appears to have a landscape orientation. "
    description += "Lighting conditions appear consistent. "

    return description

# File paths
CONFIG_PATH = "yolov3.cfg"
WEIGHTS_PATH = "yolov3.weights"
CLASSES_PATH = "coco.names"
WEIGHTS_URL = "https://github.com/KarunakarMalkagalla/RealTimeObjectDetection/releases/download/v1.0.0/yolov3.weights"

# Download weights if not present
if not os.path.exists(WEIGHTS_PATH):
    with st.spinner("Downloading YOLO weights..."):
        download_file(WEIGHTS_URL, WEIGHTS_PATH)

# Load model and classes
net, classes = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH, CLASSES_PATH)

# Streamlit app layout
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
st.write("Upload an image to get a description of the scene.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Analyzing the image..."):
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Display the uploaded image
        st.image(image_np, caption="Uploaded Image", use_container_width=True)

        # Detect objects
        detections, confidences = detect_objects(image_np, net, classes)

        # Generate and display perception-based description
        perception_description = generate_perception_description(detections, image_np)
        st.subheader("Description of the Image")
        st.write(perception_description)

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

# Function to detect objects using YOLOv5 ONNX
def detect_objects(image, net, classes, confidence_threshold=0.7, nms_threshold=0.4):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(image_bgr, 1/255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers_names)
    height, width = image.shape[:2]

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
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

# File paths for YOLOv5
CONFIG_PATH = "yolov5s.onnx"
WEIGHTS_PATH = ""
CLASSES_PATH = "coco.names"

# Download YOLOv5 ONNX model if not present
ONNX_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx"
if not os.path.exists(CONFIG_PATH):
    with st.spinner("Downloading YOLOv5 ONNX model..."):
        download_file(ONNX_URL, CONFIG_PATH)

print(f"Checking if ONNX file exists: {os.path.exists(CONFIG_PATH)}") # Added check

# Download COCO class names if not present
CLASSES_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names"
if not os.path.exists(CLASSES_PATH):
    with st.spinner("Downloading COCO class names..."):
        download_file(CLASSES_URL, CLASSES_PATH)

# Load model and classes
net, classes = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH, CLASSES_PATH)

st.markdown(...)
st.markdown(...)
st.write(...)

uploaded_file = st.file_uploader(...)

if uploaded_file:
    with st.spinner("Detecting objects..."):
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        st.image(...)

        detections, confidences = detect_objects(image_np, net, classes)

        description = generate_description(detections, confidences)
        st.subheader(...)
        st.write(description)

<<<<<<< HEAD
import cv2

def load_yolo_model(config_path, weights_path, classes_path):
    """
    Load the YOLO model, configuration, and class names.
    
    Args:
        config_path (str): Path to the YOLO configuration file.
        weights_path (str): Path to the YOLO weights file.
        classes_path (str): Path to the file containing class names.
    
    Returns:
        tuple: Loaded YOLO model and class names.
    """
    # Load YOLO model
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Load class names
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    
    return net, classes
=======
import cv2

def load_yolo_model(config_path, weights_path, classes_path):
    """
    Load the YOLO model, configuration, and class names.
    
    Args:
        config_path (str): Path to the YOLO configuration file.
        weights_path (str): Path to the YOLO weights file.
        classes_path (str): Path to the file containing class names.
    
    Returns:
        tuple: Loaded YOLO model and class names.
    """
    # Load YOLO model
    net = cv2.dnn.readNet(weights_path, config_path)
    
    # Load class names
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    
    return net, classes
>>>>>>> 7ead8ed (Initial commit)

import cv2
import numpy as np
from collections import Counter

def detect(img):
    # Load Yolo model files
    yolo_weight = "yolov3.weights"
    yolo_config = "yolov3.cfg"
    coco_labels = "coco.names"
    net = cv2.dnn.readNet(yolo_weight, yolo_config)
    
    # Load coco object names file
    classes = []
    with open(coco_labels, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # # Defining desired shape
    fWidth = 1280
    fHeight = 1280
    
    old_height, old_width, old_channels = img.shape
    # Resize image in opencv
    # cv2.imwrite("sample_image.png", img)
    img = cv2.resize(img, (fWidth, fHeight))



    # TODO: Ajouter un bouton pour demander à l'utilisateur si il veut masquer la zone éloigné.
    if True:
        pts = np.array([[0, 0], [0, 685], [790, 400], [1280, 615], [1280, 0]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # img = cv2.polylines(img, [pts], True, (0, 0, 0), 2)
        img = cv2.fillPoly(img, [pts], (50, 50, 50))

    
    height, width, channels = img.shape

    # Convert image to Blob
    blob = cv2.dnn.blobFromImage(img, 1/255, (fWidth, fHeight), (0, 0, 0), True, crop=False)
    # Set input for YOLO object detection
    net.setInput(blob)

    # Find names of all layers
    layer_names = net.getLayerNames()


    # Find names of three output layers
    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Send blob data to forward pass
    outs = net.forward(output_layers)

    # Generating random color for all 80 classes
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Extract information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            # Extract score value
            scores = detection[5:]
            # Object id
            class_id = np.argmax(scores)
            # Confidence score for each object ID
            confidence = scores[class_id]
            # if confidence > 0.5 and class_id == 0:
            if confidence > 0.5:
                # Extract values to draw bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding box with text for each object
    font = cv2.FONT_HERSHEY_DUPLEX

    labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labels.append(label)
            confidence_label = int(confidences[i] * 100)
            color = colors[i] if i < len(colors) else colors[0]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{label, confidence_label}', (x-25, y + 75), font, 1, color, 1)

    labels = Counter(labels)

    # Show image
    img = cv2.resize(img, (old_width, old_height))
    # cv2.imwrite("sample_image1.png", img)
    return img, labels

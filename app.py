from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
# from keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.utils import img_to_array, array_to_img

app = Flask(__name__)


confidenceThreshold = 0.5
NMSThreshold = 0.3

modelConfiguration = r'model/yolov3_custom.cfg'
modelWeights = r'model/yolov3_custom_last.weights'

# labelsPath = r"model/classes.names"
# labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
# COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i - 1] for i in net.getUnconnectedOutLayers()]


# # Load YOLO
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("model/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(frame):
    # Resize the frame, normalize, and forward pass through the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputLayer)

    # Process outputs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_objects(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/object_detection')
def object_detection():
    return render_template('object_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
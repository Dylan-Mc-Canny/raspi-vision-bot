import cv2
import numpy as np
import requests
import time

# Replace with your Pi's IP address (alert endpoint)
pi_alert_url = "#alert sent to pi url"

cap = cv2.VideoCapture("#url + port of pi")

# Load YOLO as before (adjust paths if needed)
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outs = net.getUnconnectedOutLayers()
output_layers = []
for i in outs:
    if isinstance(i, (list, tuple, np.ndarray)):
        output_layers.append(layer_names[i[0] - 1])
    else:
        output_layers.append(layer_names[i - 1])

colors = np.random.uniform(0, 255, size=(len(classes), 3))

last_alert_time = 0
alert_cooldown = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    person_detected = False

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label == "person":
                person_detected = True

    # Send alert to Pi if person detected and cooldown passed
    if person_detected and (time.time() - last_alert_time) > alert_cooldown:
        try:
            requests.post(pi_alert_url)
            print("Alert sent to Pi!")
            last_alert_time = time.time()
        except requests.exceptions.RequestException as e:
            print(f"Failed to send alert: {e}")

    cv2.imshow("YOLO on PC (stream from Pi)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

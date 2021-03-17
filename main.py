from flask import Flask, jsonify, request, render_template
#from detect import yolo
import os
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import time


def yolo(weights, cfg, names, image):

    net = cv2.dnn.readNet(weights, cfg)
    classes = []
    with open(names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    # Loading image
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image
    heigh, widt = img.shape[:2]
    img = cv2.resize(img, (round(0.3 * widt), round(0.3 * heigh)), interpolation=cv2.INTER_CUBIC)
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()
    # Showing informations on the screen
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
    #print(class_ids)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.51, 0.15)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            if x<0:
                x=0
            elif y<0:
                y=0

            if class_ids[i]==0:
                color = (100, 50, 200)
            elif class_ids[i]==1:
                color = (0, 255, 0)
            label = str(classes[class_ids[i]])

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label+' '+str(format(confidences[i], '.2f')), (x+30, y + 50), font, 2, color, 3)
    return img, label


app = Flask(__name__)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'static/input/'
SAVE_FOLDER = 'static/output/'
app.config['SAVE_FOLDER'] = SAVE_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
weights = 'model/yolov4.weights'
cfg = 'model/yolov4.cfg'
names = 'model/yolo_dcl.names'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('tem.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    j = 0
    if request.method == 'POST':
        j += 1
        file = request.files['file']
        filename = secure_filename(file.filename)
        ima = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(ima)
        img, _ = yolo(weights, cfg, names, ima)
        cv2.imwrite(os.path.join(app.config['SAVE_FOLDER'], str(j)+filename), img)

        return render_template("predict.html", user_image=
        os.path.join(app.config['SAVE_FOLDER'], str(j) + filename))
    pass


if __name__ == '__main__':
    app.run(debug=True)

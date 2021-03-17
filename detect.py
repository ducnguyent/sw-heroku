import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
# Load Yolo
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
    print(-start + end)
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
    A=[]
    #print(indexes)
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
            # if boxes != []:
            #     img0 = img[y:y+h, x:x+w]
            #     cv2.imwrite("OUT/test/"+label+str(i)+aaa, img0)
            A.append(boxes[i]); A.append(label+' '+str(format(confidences[i], '.2f')))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label+' '+str(format(confidences[i], '.2f')), (x+30, y + 50), font, 2, color, 3)
    print(A)
    return img, class_ids, boxes

#nhap thong so cua yolo
weights = "yolodcl_cheo.weights"
cfg = "yolodcl_cheo.cfg"
names = "yolo.names"
path = 'E:/PycharmProjects/mobile robot/abc/Test xeo/'
images = os.listdir(path)


for image in images:
    if "txt" not in image:
        img, id, box = yolo(weights, cfg, names, path+image)
        #img = cv2.resize(img, (720, 480))
        #print(image)
        #cv2.imwrite("OUT/OUT_TEST_XEO/"+image, img)
        cv2.imshow(image, img)
        cv2.waitKey(0)


        #if id[0] == 2:
        #    img, __ = yolo(weights1, cfg1, names1, img1, (0, 0, 255))
        #img0 = img[y:y+h, x:x+w]
        #cv2.imwrite("OUT/out_sang/"+image, img)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         if images.index(image)<4:
#             plt.figure(1)
#             #plot 49 tam thi dung 2 subplot 5x5. Neu plot 60 thì dùng 4 cai 4x4 cho de nhin. Code duoi chi cho toi da 64 anh
#             plt.subplot(2, 2, images.index(image) + 1), plt.imshow(img)
#         elif 4<=images.index(image)<8:
#             plt.figure(2)
#             plt.subplot(2, 2, images.index(image)-3), plt.imshow(img)
#         elif 8<=images.index(image)<24:
#             plt.figure(3)
#             plt.subplot(4, 4, images.index(image) - 7), plt.imshow(img)
#         else:
#             plt.figure(4)
#             plt.subplot(5, 5, images.index(image) - 23), plt.imshow(img)
# plt.show()

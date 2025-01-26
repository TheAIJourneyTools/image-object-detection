import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

class YOLODetector:
    def __init__(self, names_path, weights_path, config_path, pro_min=0.5, threshold=0.3):
        self.names = open(names_path).read().strip().split("\n")
        self.weights_path = weights_path
        self.configuration_path = config_path
        self.pro_min = pro_min
        self.threshold = threshold
        self.net = cv2.dnn.readNetFromDarknet(self.configuration_path, self.weights_path)
        self.layers = self.net.getLayerNames()
        self.output_layers = [self.layers[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
        self.colours = np.random.randint(0, 255, size=(len(self.names), 3), dtype='uint8')

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (416, 416))
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        return image, blob

    def detect_objects(self, blob):
        self.net.setInput(blob)
        t1 = time.time()
        output = self.net.forward(self.output_layers)
        t2 = time.time()
        print('YOLO took {:.5f} seconds'.format(t2 - t1))
        return output

    def process_detections(self, output, image):
        Height, Width = image.shape[:2]
        boxes, confidences, classes = [], [], []

        for out in output:
            for res in out:
                scores = res[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > self.pro_min:
                    box = res[0:4] * np.array([Width, Height, Width, Height])
                    x, y, w, h = box.astype('int')
                    x = int(x - (w / 2))
                    y = int(y - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence_current))
                    classes.append(class_current)

        results = cv2.dnn.NMSBoxes(boxes, confidences, self.pro_min, self.threshold)
        return results, boxes, confidences, classes

    def draw_boxes(self, image, results, boxes, confidences, classes):
        if len(results) > 0:
            for i in results.flatten():
                x, y = boxes[i][0], boxes[i][1]
                width, height = boxes[i][2], boxes[i][3]
                colour_box_current = [int(j) for j in self.colours[classes[i]]]
                cv2.rectangle(image, (x, y), (x + width, y + height), colour_box_current, 2)
                text_box_current = '{}: {:.4f}'.format(self.names[int(classes[i])], confidences[i])
                cv2.putText(image, text_box_current, (x + 2, y + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))

    def show_image(self, image):
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()

    def save_image(self, image, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, 'output.png'), image)

# Usage
""" detector = YOLODetector('kaggle/input/coco.names', 'kaggle/input/yolov3.weights', 'kaggle/input/yolov3.cfg')
image, blob = detector.preprocess_image('kaggle/input/img/image.png')
output = detector.detect_objects(blob)
results, boxes, confidences, classes = detector.process_detections(output, image)
detector.draw_boxes(image, results, boxes, confidences, classes)
#detector.show_image(image)
detector.save_image(image, 'image-output') """
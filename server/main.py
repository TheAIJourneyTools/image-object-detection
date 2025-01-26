from flask import Flask, request, send_file
from flask_cors import CORS
from YOLODetector import YOLODetector

import os

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS

detector = YOLODetector('kaggle/input/coco.names', 'kaggle/input/yolov3.weights', 'kaggle/input/yolov3.cfg')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return "No image file found", 400

    image_file = request.files['image']
    image_path = os.path.join('temp', image_file.filename)
    image_file.save(image_path)

    image, blob = detector.preprocess_image(image_path)
    output = detector.detect_objects(blob)
    results, boxes, confidences, classes = detector.process_detections(output, image)
    detector.draw_boxes(image, results, boxes, confidences, classes)

    output_path = os.path.join('temp', 'output.png')
    detector.save_image(image, 'temp')

    return send_file(output_path, mimetype='image/png')

if __name__ == '__main__':
    if not os.path.exists('temp'):
        os.makedirs('temp')
    app.run(host='0.0.0.0', port=5000)

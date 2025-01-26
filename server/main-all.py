import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import time

""" for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) """

names=open("kaggle/input/coco.names").read()
""" print(names.strip().split("\n")) """
names=names.strip().split("\n")

weights_path = 'kaggle/input/yolov3.weights'
configuration_path = 'kaggle/input/yolov3.cfg'

pro_min = 0.5 # Setting minimum probability to eliminate weak predictions
threshold = 0.3 # Setting threshold for non maximum suppression

net = cv2.dnn.readNetFromDarknet(configuration_path,weights_path)

# Getting names of all layers
layers = net.getLayerNames()  # list of layers' names

""" # # Check point
print(layers) """

""" for i in net.getUnconnectedOutLayers().flatten():
    print(layers[i-1])
 """

output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()] 
""" print(output_layers) """

image=cv2.imread("kaggle/input/img/image.png")
image = cv2.resize(image, (416, 416))  # Resize image to 416x416
""" print(image.shape) """

""" plt.rcParams['figure.figsize'] = (8,8)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show() """

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
""" print(image.shape)  
print(blob.shape)   """

blob_to_show = blob[0, :, :, :].transpose(1, 2,0)
""" print(blob_to_show.shape)  """

""" plt.rcParams['figure.figsize'] = (5, 5)
plt.imshow(blob_to_show)
plt.show()
 """

""" net.setInput(blob) # giving blob as input to our YOLO Network.
t1=time.time()
output = net.forward(output_layers)
t2 = time.time()
 """
# Showing spent time for forward pass
""" print('YOLO took {:.5f} seconds'.format(t2-t1))
 """
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416,416), swapRB=True, crop=False)
""" print(image.shape)  
print(blob.shape)   """

blob_to_show = blob[0, :, :, :].transpose(1, 2,0)
""" print(blob_to_show.shape) 

plt.rcParams['figure.figsize'] = (5, 5)
plt.imshow(blob_to_show)
plt.show() """

net.setInput(blob) # giving blob as input to our YOLO Network.
t1=time.time()
output = net.forward(output_layers)
t2 = time.time()

# Showing spent time for forward pass
""" print('YOLO took {:.2f} seconds'.format(t2-t1))
print(output)
print(output[0][0])
 """
""" a=np.array([1,2,3,4,5,6,7]) """
""" print(a[2:])

print(a[0:4])
 """
colours = np.random.randint(0, 255, size=(len(names), 3), dtype='uint8') 
""" print(colours.shape)
print(len(colours))
print(colours[0])   """

classes = []
confidences = []
boxes = []

Height = image.shape[0]
Width = image.shape[1]

""" print(Width,Height) """

for out in output:
    for res in out:
        
        scores = res[5:]
        class_current = np.argmax(scores) # returning indices with max score and that would be our class as that will be 1 and rest will be 0

        # Getting the probability for current object by accessing the indices returned by argmax.
        confidence_current = scores[class_current]

        # Eliminating the weak predictions that is with minimum probability and this loop will only be encountered when an object will be there
        if confidence_current > 0.5:
            
            # Scaling bounding box coordinates to the initial image size
            # YOLO data format just keeps center of detected box and its width and height
            #that is why we are multiplying them elemwnt wise by width and height
            box = res[0:4] * np.array([Width, Height, Width, Height])  #In the first 4 indices only contains 
            #the output consisting of the coordinates.
            #print(res[0:4])
            #print(box)

            # From current box with YOLO format getting top left corner coordinates
            # that are x and y
            x, y, w, h = box.astype('int')
            x = int(x - (w / 2))
            y = int(y - (h / 2))
            
            # Adding results into the lists
            boxes.append([x, y, int(w), int(h)]) ## appending all the boxes.
            confidences.append(float(confidence_current)) ## appending all the confidences
            classes.append(class_current) ## appending all the classes        

results = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.4)

# Showing labels of the detected objects
""" for i in range(len(classes)):
    print(names[int(classes[i])]) """

results.flatten()

if len(results) > 0:

    for i in results.flatten():
        
        # Getting current bounding box coordinates
        x, y = boxes[i][0],boxes[i][1]
        width, height = boxes[i][2], boxes[i][3]
        
        colour_box_current = [int(j) for j in colours[classes[i]]]

        # Drawing bounding box on the original image
        cv2.rectangle(image, (x, y), (x + width, y + height),
                      colour_box_current, 2)

        # Preparing text with label and confidence 
        text_box_current = '{}: {:.4f}'.format(names[int(classes[i])], confidences[i])

        # Putting text with label and confidence
        cv2.putText(image, text_box_current, (x+2, y+20), cv2.FONT_HERSHEY_DUPLEX, 0.5,(0,0,0))

plt.rcParams['figure.figsize'] = (10,10)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
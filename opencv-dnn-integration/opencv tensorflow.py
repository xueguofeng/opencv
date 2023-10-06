import cv2
import numpy as np


filename = "images/beer.png"
#filename = "images/monitor.png"

frame = cv2.imread(filename)

# GoogLeNet, different files from Caffe
weightFile = "tensorflow_inception_graph.pb"  # model architeccture and weights
classFile = "imagenet_comp_graph_label_strings.txt"  # # 1000 labels

classes = None
with open(classFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

frame = cv2.imread(filename)

inHeight = 224
inWidth = 224
swap_rgb = True
mean = [117, 117, 117]
scale = 1.0

net = cv2.dnn.readNetFromTensorflow(weightFile)

blob = cv2.dnn.blobFromImage(frame, scale, (inWidth, inHeight), mean, swap_rgb, crop=False)

net.setInput(blob)

out = net.forward()

out = out.flatten()

classId = np.argmax(out)

className = classes[classId]
confidence = out[classId]

label = "Predicted Class = {}, Confidence = {:.3f}".format(className, confidence)

cv2.putText(frame, label, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
cv2.imshow("Image", frame)
cv2.waitKey(0)

print("Class ID : {}".format(classId))
print(label)




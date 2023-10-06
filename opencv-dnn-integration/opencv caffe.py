
import cv2
import numpy as np

import utils_paths

rows = open("synset_words.txt").read().strip().split("\n") # 1000 labels
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# GoogLeNet
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt","bvlc_googlenet.caffemodel") # architecture and weights

# 图像路径
imagePaths = sorted(list(utils_paths.list_images("images/")))

# 图像数据预处理
image = cv2.imread(imagePaths[0])
resized = cv2.resize(image, (224, 224))
# image scalefactor size mean swapRB

a_blob = cv2.dnn.blobFromImage(resized, 1, (224, 224), (104, 117, 123)) # blob - binary large object
print("First Blob: {}".format(a_blob.shape)) # a batch with 1 sample

# 得到预测结果
net.setInput(a_blob)
preds = net.forward()

# 排序，取分类可能性最大的
idx = np.argsort(preds[0])[::-1][0]
text = "Label: {}, {:.2f}%".format(classes[idx],preds[0][idx] * 100)
cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)

# 显示
cv2.imshow("Image", image)
cv2.waitKey(0)


many_images = []

# 方法一样，数据是一个batch
for p in imagePaths[1:]:
	image = cv2.imread(p)
	image = cv2.resize(image, (224, 224))
	many_images.append(image)


many_blobs = cv2.dnn.blobFromImages(many_images, 1, (224, 224), (104, 117, 123))
print("Second Blob: {}".format(many_blobs.shape)) # a batch with many samples


net.setInput(many_blobs)
preds = net.forward()

for (i, p) in enumerate(imagePaths[1:]):
	image = cv2.imread(p)
	idx = np.argsort(preds[i])[::-1][0]
	text = "Label: {}, {:.2f}%".format(classes[idx],
		preds[i][idx] * 100)
	cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
		0.7, (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
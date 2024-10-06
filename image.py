import cv2
import matplotlib.pyplot as plt

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"
class_labels_file = 'labels.txt'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

img = cv2.imread("car.jpeg")
classLabels = []
with open(class_labels_file, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(360, 360)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

Class_Index, confidence, bbox = model.detect(img, confThreshold=0.5)
print(Class_Index)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

for Class_Ind, conf, boxes in zip(Class_Index.flatten(), confidence.flatten(), bbox):
    cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[0] + boxes[2], boxes[1] + boxes[3]), (255, 0, 0), 2)
    cv2.putText(img, classLabels[Class_Ind - 1], (boxes[0] + 10, boxes[1] + 40), font,
                fontScale=font_scale, color=(0, 255, 0), thickness=3)

cv2.imshow('object detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

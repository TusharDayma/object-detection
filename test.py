import cv2
import matplotlib.pyplot as plt

class ObjectDetector:
    def __init__(self, config_file, frozen_model, class_labels_file):
        self.model = cv2.dnn_DetectionModel(frozen_model, config_file)
        self.classLabels = []
        with open(class_labels_file, 'rt') as fpt:
            self.classLabels = fpt.read().rstrip('\n').split('\n')

        self.model.setInputSize(360, 360)
        self.model.setInputScale(1.0 / 127.5)
        self.model.setInputMean((127.5, 127.5, 127.5))
        self.model.setInputSwapRB(True)

    def detect_objects(self, image_path, confThreshold=0.5):
        img = cv2.imread(image_path)

        Class_Index, confidence, bbox = self.model.detect(img, confThreshold=confThreshold)

        for Class_Ind, conf, boxes in zip(Class_Index.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[0] + boxes[2], boxes[1] + boxes[3]), (255, 0, 0), 2)
            cv2.putText(img, self.classLabels[Class_Ind - 1], (boxes[0] + 10, boxes[1] + 40),
                        cv2.FONT_HERSHEY_PLAIN, fontScale=3, color=(0, 255, 0), thickness=3)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

def main():
    config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    frozen_model = "frozen_inference_graph.pb"
    class_labels_file = 'labels.txt'
    image_path = " "

    detector = ObjectDetector(config_file, frozen_model, class_labels_file)
    detector.detect_objects(image_path)

if __name__ == "__main__":
    main()

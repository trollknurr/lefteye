from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np

from lefteye.classes import CLASSES


class Detector:
    def __init__(
            self,
            cfg_path: Optional[Path] = None,
            weights_path: Optional[Path] = None,
            confidence_level: float = 0.5
    ):
        """
        :param cfg_path: Path to neural net cfg file
        :param weights_path: Path to neural net weights file
        :param confidence_level: Base required confidence level
        """
        if not cfg_path or not weights_path:
            data_path = Path(__file__).parent.absolute() / "data"
            self.cfg_path = data_path / "yolov4.cfg"
            self.weights_path = data_path / "yolov4.weights"
        else:
            self.cfg_path = cfg_path
            self.weights_path = weights_path

        self.net = cv2.dnn.readNetFromDarknet(str(self.cfg_path), str(self.weights_path))
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.confidence_level = confidence_level

    def detect_by_image(self, image) -> List[Dict]:
        """
        Detect on PIL Image. Image converted to OpenCV format
        :param image: PIL Image
        """
        _image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return self.detect_by_matrix(_image)

    def detect_by_matrix(self, image_matrix: np.ndarray) -> List[Dict]:
        """
        Detect on OpenCV style image matrix (BGR)
        :param image_matrix: numpy ndarray shape (h, w, 3)
        """
        blob = cv2.dnn.blobFromImage(image_matrix, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.ln)

        boxes = []
        confidences = []
        class_ids = []
        h, w = image_matrix.shape[:2]
        results = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_level:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_level, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'class': CLASSES[class_ids[i]],
                    'proba': confidences[i],
                    'x': boxes[i][0],
                    'y': boxes[i][1],
                    'width': boxes[i][2],
                    'height': boxes[i][3],
                })

        return results

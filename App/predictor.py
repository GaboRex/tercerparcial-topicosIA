from typing import Any
import numpy as np
from ultralytics import YOLO
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#FACE_MODEL_PATH = "/Users/hp/TopicosIA/parcial1TIA/deploy/blaze_face_short_range.tflite"
MODEL_PATH = "/Users/hp/TopicosIA/tercerparcial-topicosIA/App/best.pt"

class CursoPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        print("Creando predictor...")
        self.model = YOLO(model_path)
    def predict_file(self, file_path: str):
        results = self.model([file_path])
        pred_data = []
        for i, res in enumerate(results):
            pred_data.append(
                {
                    "category": res.names[res.probs.top1],
                    "confidence":res.probs.data[res.probs.top1].item()
                }
            )
        return pred_data
    def predict_image(self, image_array: np.ndarray):
        results = self.model(image_array)[0]
        return {
            "class": results.names[results.probs.top1],
            "confidence": results.probs.data[results.probs.top1].item()
        }
    


if __name__ == "__main__":
    image_file = "/Users/hp/TopicosIA/parcial1TIA/deploy/gabriel.jpg"
    img = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
    predictor = CursoPredictor()
    prediction = predictor.predict_image(img)
    print(prediction)
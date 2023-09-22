import json

import torch
from label_studio_ml.model import LabelStudioMLBase
import requests, os
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

LS_URL = os.environ['LABEL_STUDIO_BASEURL']
LS_API_TOKEN = os.environ['LABEL_STUDIO_API_TOKEN']


class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.id_to_label = {}
        with open('categories.json') as f:
            categories = json.loads(f.read()).get('categories')
            for category in categories:
                self.id_to_label[category.get('id')] = category.get('name')

        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("best.pt").to(device)

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns 
            the list of predictions based on input list of tasks 
        """
        task = tasks[0]

        predictions = []
        score = 0
        header = {
            "Authorization": "Token " + LS_API_TOKEN}
        image = Image.open(BytesIO(requests.get(
            LS_URL + task['data']['url'], headers=header).content))
        original_width, original_height = image.size
        results = self.model.predict(image, iou=0, agnostic_nms=True)

        i = 0
        for result in results:
            for i, prediction in enumerate(result.boxes.cpu().numpy()):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append({
                    "id": str(i),
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "score": prediction.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": xyxy[0] / original_width * 100,
                        "y": xyxy[1] / original_height * 100,
                        "width": (xyxy[2] - xyxy[0]) / original_width * 100,
                        "height": (xyxy[3] - xyxy[1]) / original_height * 100,
                        "rectanglelabels": [self.id_to_label.get(str(int(prediction.cls.item())), 'unknown')]
                    }
                })
                score += prediction.conf.item()

        return [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8n",  # all predictions will be differentiated by model version
        }]

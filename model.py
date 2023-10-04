import json
from io import BytesIO
import os
import requests
from PIL import Image
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

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
        with open('notes.json') as f:
            categories = json.loads(f.read()).get('categories')
            for category in categories:
                self.id_to_label[category.get('id')] = category.get('name')

        device: str = 'cpu'
        self.model = YOLO('best.pt').to(device)
        self.model_updated = YOLO('best_updated.pt').to(device)

    def _predict(self, model_type, image):
        if model_type == 'current':
            model = self.model
        elif model_type == 'updated':
            model = self.model_updated
        else:
            raise Exception('Invalid model type')

        original_width, original_height = image.size
        results = model.predict(image, iou=0, agnostic_nms=True)
        predictions = []
        for result in results:
            for i, prediction in enumerate(result.boxes.cpu().numpy()):
                xyxy = prediction.xyxy[0].tolist()
                predictions.append({
                    "id": model_type + '_' + str(i),
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
                        "rectanglelabels": [self.id_to_label.get(str(int(prediction.cls.item())), 'unknown')],
                        "xyxy": xyxy
                    }
                })
                print(f"{int(prediction.cls.item())} - {self.id_to_label.get(int(prediction.cls.item()), 'unknown')}")

        return predictions

    def _box_overlap(self, box_1, box_2):
        if box_1[2] <= box_2[0] or box_2[2] <= box_1[0]:
            return False

        if box_1[3] <= box_2[1] or box_2[3] <= box_1[1]:
            return False

        return True

    def predict(self, tasks, **kwargs):
        """ This is where inference happens: model returns
            the list of predictions based on input list of tasks
        """
        task = tasks[0]

        header = {"Authorization": "Token " + LS_API_TOKEN}
        image = Image.open(BytesIO(requests.get(LS_URL + task['data']['url'], headers=header).content))

        predictions = self._predict('current', image)
        predictions_updated = self._predict('updated', image)

        duplicated_prediction = set([])
        for prediction in predictions:
            for prediction_updated in predictions_updated:
                box = prediction['value']['xyxy']
                box_updated = prediction_updated['value']['xyxy']

                if self._box_overlap(box, box_updated):
                    if prediction['score'] > prediction_updated['score']:
                        duplicated_prediction.add(prediction_updated['id'])
                    else:
                        duplicated_prediction.add(prediction['id'])

        return [{
            "result": [prediction for prediction in predictions + predictions_updated if
                       prediction['id'] not in duplicated_prediction],
            "score": 1,
            "model_version": 'v0',
        }]

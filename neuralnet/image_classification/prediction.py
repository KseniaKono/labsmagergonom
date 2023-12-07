import torch
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights


class ResNetImagePredictor:
    def __init__(self):
        self.weights = ResNet50_Weights.DEFAULT # Загрузка предобученных весов ResNet-50
        self.model = resnet50(weights=self.weights) # Создание модели ResNet-50 с загруженными весами
        self.model.eval() # Перевод модели в режим оценки (без обучения)
        self.preprocess = self.weights.transforms() # Получение значений для входных изображений на основе предобученных весов

    def predict(self, img_path):
        with torch.no_grad():
            img = read_image(img_path)
            batch = self.preprocess(img).unsqueeze(0) # Применение preprocess к изображению, добавление размера пакета
            prediction = self.model(batch).squeeze(0).softmax(0) # Передача изображения в модель, получение предсказаний
            results = [f"{self.weights.meta['categories'][i]}: " + f"{100 * v:.1f}%"
                       for v, i in zip(*torch.topk(prediction, 3))] # Форматирование топ-3 результатов # zip - совмещение списков в один # topk возвращает k крупнейших (или наименьших) элементов по заданному измерению.

            results = " | ".join(results)

            return results

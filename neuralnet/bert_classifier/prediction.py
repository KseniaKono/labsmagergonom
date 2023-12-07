import torch
from transformers import AutoTokenizer, BertForSequenceClassification


class BertClassificationPredictor:
    def __init__(self, pretrained_model="cointegrated/rubert-tiny2-cedr-emotion-detection"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model) # Создает объект токенизатора с использованием предобученной модели BERT, указанной в параметре pretrained_model.
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model) #Создает объект модели для классификации последовательностей с использованием предобученной модели BERT, указанной в параметре pretrained_model.

    def predict(self, text): # принимает текст в качестве входного аргумента
        tokenized_text = self.tokenizer(text, return_tensors='pt') #  Текст токенизируется
        with torch.no_grad(): #значит что модель на этом не учится
            pred = self.model(input_ids=tokenized_text.input_ids, #передача входных данных через модель BERT для получения предсказаний, представляет идентификаторы токенов, которые были сгенерированы токенизатором для входного текста. Каждое слово или подслово в тексте заменено уникальным числовым идентификатором, который соответствует токену в словаре BERT.
                              attention_mask=tokenized_text.attention_mask, #представляет собой маску внимания, которая указывает модели, какие токены во входной последовательности следует учитывать, а какие игнорировать.
                              token_type_ids=tokenized_text.token_type_ids) #представляет идентификаторы типов токенов. В случае BERT, который используется для задачи классификации, этот аргумент обычно не используется, поскольку тип токенов не играет роли.

            prediction_labels = ['no_emotion', 'joy', 'sadness', 'surprise', 'fear', 'anger'] #Список меток эмоций
            prediction_values = torch.softmax(pred.logits, -1).cpu().numpy()[0] #Вероятности предсказаний, преобразованные в массив numpy.
            prediction_sorted = sorted([x for x in zip(prediction_labels, prediction_values)], #Сортировка предсказаний в порядке убывания вероятности
                                       key=lambda x: x[1], reverse=True)
            prediction_output = "  |  ".join([f"{k}:".ljust(12, ' ') + f"{v:.5f}" # Строка, представляющая топ-3 предсказанных эмоций с их вероятностями.
                                              for k, v in prediction_sorted[:3]])

        return prediction_output

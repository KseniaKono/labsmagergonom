import torch
from collections import deque
from transformers import AutoTokenizer, AutoModelWithLMHead


class DialogBotRuGPTSmall:
    def __init__(self, pretrained_model='tinkoff-ai/ruDialoGPT-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model) # Создает объект токенизатора с использованием предобученной модели BERT, указанной в параметре pretrained_model.
        self.model = AutoModelWithLMHead.from_pretrained(pretrained_model) #Создает объект модели для классификации последовательностей с использованием предобученной модели BERT, указанной в параметре pretrained_model.
        self.context = deque([], maxlen=4)  # 2 inputs, 2 answers # Сохраняем предыдущие запросы к боту (2 вопроса и 2 ответа)
        self.rus_alphabet = "йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ" # Символы для ответов, разбитые по языкам и специальным символам
        self.en_alphabet = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
        self.symbols = "0123456789., !?\':;@#$&*()-=+"

    def predict(self, text):
        if len(self.context) == self.context.maxlen: # Если поле диалога переполнено, удаляем два самых старых элемента (вопрос и ответ)
            _ = self.context.popleft()
            _ = self.context.popleft()
        self.context.append(text) # Добавляем в поле с диалогом новый вопрос пользователя
        input_text = [f"@@ПЕРВЫЙ@@ {t}" if not i % 2 else f"@@ВТОРОЙ@@ {t}" # Строим входной текст, разделяя его на части для первого и второго участника диалога (мы и бот)
                      for i, t in enumerate(self.context)]
        input_text = " ".join(input_text) + " @@ВТОРОЙ@@ "

        tokenized_text = self.tokenizer(input_text, return_tensors='pt') # Токенизируем текст В данном случае, 'pt' означает, что токенизатор должен вернуть тензоры PyTorch (PyTorch Tensor).
        with torch.no_grad(): # Генерация ответа, модель на этом не обучается
            generated_token_ids = self.model.generate(
                **tokenized_text, # Распаковка ключевых слов  tokenized_text содержит токенизированный текст, который представлен в виде словаря с различными ключами (например, 'input_ids', 'attention_mask' и др.)
                top_k=10, # Количество токенов, рассматриваемых на каждом шаге генерации, основываясь на их вероятности.
                top_p=0.95, # Параметр, отсекающий токены с низкой вероятностью на каждом шаге. Токены, чья суммарная вероятность превышает top_p, остаются в рассмотрении.
                num_beams=3, # Количество лучей (beams), используемых при генерации текста. Больше лучей может привести к разнообразию результатов
                num_return_sequences=1, # Количество возвращаемых последовательностей (гипотез) генерации.
                do_sample=True, # Определяет, следует ли использовать сэмплирование при генерации текста.
                no_repeat_ngram_size=2, #  Параметр, предотвращающий повторение последовательностей токенов размера no_repeat_ngram_size
                temperature=1.2, # Параметр температуры, который влияет на разнообразие генерируемого текста. Большие значения могут увеличить разнообразие.
                repetition_penalty=1.2, # Штраф за повторение. Уменьшает вероятность генерации повторяющихся токенов.
                length_penalty=1.0, # Штраф за длину. Контролирует, насколько предпочтительными считаются более короткие или более длинные последовательности.
                eos_token_id=50257, # Идентификатор токена конца последовательности (End-Of-Sequence), который указывает на завершение генерации.
                max_new_tokens=40 # Максимальное количество новых токенов, которые можно сгенерировать.
            )

            context_with_response = [self.tokenizer.decode(sample_token_ids) for sample_token_ids in
                                     generated_token_ids] # Переводим результат в текст
            answer = context_with_response[0].split("@@ВТОРОЙ@@")[-1].replace("@@ПЕРВЫЙ@@", "") # Извлекаем ответ из сгенерированного текста
            filtered_answer = [letter for letter in answer if letter in self.rus_alphabet
                               or letter in self.en_alphabet or letter in self.symbols] # Фильтруем ответ, оставляя только символы трех наборов алфавитов выше
            filtered_answer = "".join(filtered_answer).strip()
            self.context.append(filtered_answer) # Добавляем отфильтрованный ответ в поле диалога

        return answer

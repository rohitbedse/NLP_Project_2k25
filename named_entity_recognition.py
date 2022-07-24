from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


class NamedEntityRecognition():
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
        self.nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

    def get_annotation(self, preds, text):
        splits = [0]
        entities = {}
        for i in preds:
            splits.append(i['start'])
            splits.append(i['end'])
            entities[i['word']] = i['entity_group']

        # Exclude bad preds
        exclude = ['', '.', '. ', ' ']
        for x in exclude:
            if x in entities.keys():
                entities.pop(x)

        parts = [text[i:j] for i, j in zip(splits, splits[1:] + [None])]

        final_annotation = [(x, entities[x], "") if x in entities.keys() else x for x in parts]

        return final_annotation

    def classify(self, text):
        preds = self.nlp(text)
        ner_annotation = self.get_annotation(preds, text)
        return preds, ner_annotation
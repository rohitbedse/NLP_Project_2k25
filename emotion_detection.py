from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers_interpret import SequenceClassificationExplainer
import torch
import pandas as pd


class EmotionDetection:
    """ This class is an example

    Attributes:
        class_attribute (str): (class attribute) The class attribute
        instance_attribute (str): The instance attribute
    """

    def __init__(self):
        hub_location = 'cardiffnlp/twitter-roberta-base-emotion'
        self.tokenizer = AutoTokenizer.from_pretrained(hub_location)
        self.model = AutoModelForSequenceClassification.from_pretrained(hub_location)
        self.explainer = SequenceClassificationExplainer(self.model, self.tokenizer)

    def justify(self, text):
        """
        The function to add two Complex Numbers.

        Parameters:
            num (ComplexNumber): The complex number to be added.

        Returns:
            ComplexNumber: A complex number which contains the sum.
        """

        word_attributions = self.explainer(text)
        html = self.explainer.visualize("example.html")

        return html

    def classify(self, text):
        """
        The function to add two Complex Numbers.

        Parameters:
            num (ComplexNumber): The complex number to be added.

        Returns:
            ComplexNumber: A complex number which contains the sum.
        """

        tokens = self.tokenizer.encode_plus(text, add_special_tokens=False, return_tensors='pt')
        outputs = self.model(**tokens)
        probs = torch.nn.functional.softmax(outputs[0], dim=-1)
        probs = probs.mean(dim=0).detach().numpy()
        labels = list(self.model.config.id2label.values())
        preds = pd.Series(probs, index=labels, name='Predicted Probability')

        return preds

    def run(self, text):
        """
        The function to add two Complex Numbers.

        Parameters:
            num (ComplexNumber): The complex number to be added.

        Returns:
            ComplexNumber: A complex number which contains the sum.
        """

        preds = self.classify(text)
        html = self.justify(text)

        return preds, html
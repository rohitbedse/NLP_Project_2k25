import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class POSTagging:
    """Part of Speech Tagging on text data"""

    def __init__(self):
       pass

    def classify(self, text):
        """
        The function to add two Complex Numbers.

        Parameters:
            num (ComplexNumber): The complex number to be added.

        Returns:
            ComplexNumber: A complex number which contains the sum.
        """

        text = word_tokenize(text)
        preds = nltk.pos_tag(text)
        return preds
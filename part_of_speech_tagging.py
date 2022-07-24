import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class POSTagging():
    def __init__(self):
       pass

    def classify(self, text):
        text = word_tokenize(text)
        preds = nltk.pos_tag(text)
        return preds
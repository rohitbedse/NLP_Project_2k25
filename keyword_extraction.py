import spacy
import pytextrank
import re
from operator import itemgetter


class KeywordExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")

    def get_keywords(self, text, max_keywords):
        doc = self.nlp(text)

        kws = [i.text for i in doc._.phrases[:max_keywords]]

        return kws

    def get_keyword_indicies(self, string_list, text):
        out = []
        for s in string_list:
            indicies = [[m.start(), m.end()] for m in re.finditer(re.escape(s), text)]
            out.extend(indicies)

        return out

    def merge_overlapping_indicies(self, indicies):
        # Sort the array on the basis of start values of intervals.
        indicies.sort()
        stack = []
        # insert first interval into stack
        stack.append(indicies[0])
        for i in indicies[1:]:
            # Check for overlapping interval,
            # if interval overlap
            if (stack[-1][0] <= i[0] <= stack[-1][-1]) or (stack[-1][-1] == i[0]-1):
                stack[-1][-1] = max(stack[-1][-1], i[-1])
            else:
                stack.append(i)
        return stack

    def merge_until_finished(self, indicies):
        len_indicies = 0
        while True:
            merged = self.merge_overlapping_indicies(indicies)
            if len_indicies == len(merged):
                out_indicies = sorted(merged, key=itemgetter(0))
                return out_indicies
            else:
                len_indicies = len(merged)

    def get_annotation(self, text, indicies, kws):

        # Convert indicies to list
        # kws = kws + [i.lower() for i in kws]

        arr = list(text)
        for idx in sorted(indicies, reverse=True):
            arr.insert(idx[0], "<kw>")
            arr.insert(idx[1]+1, "XXXxxxXXXxxxXXX <kw>")
        annotation = ''.join(arr)
        split = annotation.split('<kw>')
        final_annotation = [(x.replace('XXXxxxXXXxxxXXX ', ''), "KEY", "#26aaef") if "XXXxxxXXXxxxXXX" in x else x for x in split]

        kws_check = []
        for i in final_annotation:
            if type(i) is tuple:
                kws_check.append(i[0])

        return final_annotation

    def generate(self, text, max_keywords):

        kws = self.get_keywords(text, max_keywords)

        indicies = list(self.get_keyword_indicies(kws, text))
        if indicies:
            indicies_merged = self.merge_until_finished(indicies)
            annotation = self.get_annotation(text, indicies_merged, kws)
        else:
            annotation = None

        return annotation, kws


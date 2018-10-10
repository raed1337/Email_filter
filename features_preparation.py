import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from collections import Counter
import os


lemmatizer = WordNetLemmatizer()
hm_lines = 100000000


def create_dictionnary(directory):
    emails =[os.path.join(directory,f) for f in os.listdir(directory)]
    lexicon = []
    lexicon_f = []

    for mail in emails:
        with open(mail , errors='ignore') as m:
            for i, l in enumerate(m):
                if i>1:
                    all_words = word_tokenize(l.lower())
                    lexicon += list(all_words)
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    lexicon_f = Counter(lexicon)
    list_to_remove = lexicon_f.keys()
    for item in list(list_to_remove):
        if 15000<lexicon_f[item]<300000:
            del lexicon_f[item]
        if item.isalpha() == False:
            del lexicon_f[item]
        elif len(item) == 1:
            del lexicon_f[item]
    lexicon_f = lexicon_f.most_common(3000)
    dictionnary = pd.DataFrame(lexicon_f)
    print(dictionnary[0])

    return lexicon_f


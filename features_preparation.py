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
    dictionary = pd.DataFrame(lexicon_f)
    print(dictionary[0])

    return lexicon_f

def extract_features(mail_dir , dictionary , classification):
    emails = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    featureSet = []
    current_words=[]
    for mail in emails:
        with open(mail, 'r') as f:
            for i, l in enumerate(f):
                all_words = word_tokenize(l.lower())
                current_words +=list(all_words)
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(dictionary))
            for word in current_words:
                    if word.lower() in dictionary:
                        index_value = dictionary.index(word.lower())
                        features[index_value] = current_words.count(word)
        features = list(features)
        featureSet.append([features, classification])

    return featureSet
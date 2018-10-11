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
                if i>=2:
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
    lexicon_f = pd.DataFrame(lexicon_f)
    lexicon_final=list(np.array(lexicon_f[0]))
    print(lexicon_final)
    return lexicon_final

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
        print(featureSet)
    return featureSet

def create_features_and_labels(spam , ham ,test_size=0.1):
    dictionary = create_dictionnary('C:/Users/kirito/Desktop/E-mail_samples')
    #1=spam 0=ham
    features=[]
    features += extract_features(spam, dictionary, 1)
    features += extract_features(ham, dictionary, 0)
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


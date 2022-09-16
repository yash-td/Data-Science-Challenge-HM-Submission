import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def pre_process(data):
    lemmatizer = WordNetLemmatizer()
    new_data = []
    new_labels = []
    for i in range(len(data)):
        tokenized = [word_tokenize(entry.lower()) for entry in data[i]['text'] if entry not in stopwords.words('english')]
        lemmatized = [[lemmatizer.lemmatize(token) for token in sent] for sent in tokenized]
        new_data.append(lemmatized)
        # new_data.append([word_tokenize(entry.lower()) for entry in data[i]['text'] if entry not in stopwords.words('english')])
        new_labels.append(data[i]['labels'])
    return new_data, new_labels

def create_corpus(data):
    corpus = []
    for doc in data:
        for sen in doc:
            for word in sen:
                corpus.append(word)
    return corpus   

def convert_2d(data):
    lst2 = []
    for i in data:
        lst1 = []
        for j in i:
            for k in j:
                lst1.append(k)
        lst2.append(lst1)
    return lst2


def identity_tokenizer(text):
    return text
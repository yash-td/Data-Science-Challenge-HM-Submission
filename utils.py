'''
importing necessary packages
'''
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer

'''
creating a pre-process funciton which will take the data, perform the following operations and return features and labels
separately: 

1) Convert everything to lower case
2) Remove Stopwords
3) Tokenize the sentences (Also Removing punctuatios and non alpha numeric characters)
4) Lemmatize the words
5) Stemming the words

All these operations are performed using the NLTK package in python.
'''
def pre_process(data):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    new_data = []
    new_labels = []
    for i in range(len(data)):
        tokenized = [tokenizer.tokenize(entry.lower()) for entry in data[i]['text'] if entry not in stopwords.words('english')]
        lemmatized = [[lemmatizer.lemmatize(token) for token in sent] for sent in tokenized]
        stemming = [[stemmer.stem(token) for token in sent] for sent in lemmatized]
        new_data.append(lemmatized)
        new_labels.append(data[i]['labels'])
    print('Lower Case + Tokenization + Stop Words Remowal + Only Alphanumeric (Removing Punctuation) + Lemmatization',lemmatized)
    return new_data, new_labels



'''
This function is used to convert our entire training data (after the pre-processing) into a corpus of words
'''
def create_corpus(data):
    corpus = []
    for doc in data:
        for sen in doc:
            for word in sen:
                corpus.append(word)
    return corpus   

'''
This funciton is written to covert the 3-dimensional data into 2 dimensional.
After pre-processing we have a list of lists of lists. But labels we have are the violation of articles and are present for
every document and not every sentence. Hence the desired format we need to train the model is a list of lists. Where one
list is the entrie training data of size 9000 and the second list (list of list) is the documents inside the main list of 9000.
'''
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
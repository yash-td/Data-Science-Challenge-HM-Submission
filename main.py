'''

'''








'''
importing all the necessary libraries and packages
'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from datasets import load_dataset

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection, naive_bayes
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

'''
importing the helper functions from the utils.py file
'''
from utils import *

'''
loading the 'lex_glue ecthr_a' dataset using the datasets package in python
'''
dataset = load_dataset("lex_glue",'ecthr_a')

'''
getting the train and test sets separately. Train set is of size 9000 and the test set is of sieze 1000.
'''
train = dataset['train']
test = dataset['test']

'''
Using the pre-process function created in the utils file, preparing the train and the test sets for classification models.
'''
train_X, train_y  = pre_process(train) 
test_X, test_y = pre_process(test)

'''
Creating a corpus for the train and the test set to crearte a vocabulary in order to use the Tfidf vectorizer.
'''
corpus_train  = create_corpus(train_X)
corpus_test = create_corpus(test_X)

'''
converting the 3 dimentional dataset obtained after tokenization, to a two dimensional set becasue we have 
labels (None, 1 or more than 1) for each document and not each sentence.
'''
train_X_2d = convert_2d(train_X)
test_X_2d = convert_2d(test_X)


'''
Using the tfidf vectorizer from sklearn, converting our train sets and test sets (words) to vectors based on the term
frequency inverse document frequency technique. It assigns a value to a term according to its importance in a document
scaled by its importance across all documents in our corpus, which mathematically eliminates naturally occurring words
in the English language, and selects words that are more descriptive of our text.  
'''
tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)
tfidf.fit(corpus_train+corpus_test)
vectors_train = tfidf.fit_transform(train_X_2d)
vectors_test = tfidf.transform(test_X_2d)

'''
As we have multiple labels for some documents in our problem we need to use a multi label binarizer which will convert
our labels into sequences of ones and zeros (similar to that of one hot encoding)
'''
train_y_enc = MultiLabelBinarizer().fit_transform(train_y)
test_y_enc = MultiLabelBinarizer().fit_transform(test_y)

'''
Creating a pipeline for testing 6 text classification models and reporting their accuracies.
'''
classification_models = [
                         OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)),
                         OneVsRestClassifier(SVC(kernel = "linear" ,C=1)),
                         OneVsRestClassifier(SVC(kernel='rbf', C=1)),
                         OneVsRestClassifier(naive_bayes.MultinomialNB()),
                         OneVsRestClassifier(DecisionTreeClassifier()),
                         OneVsRestClassifier(RandomForestClassifier(n_estimators=500)),
                         ]

model_scores = []
for model in classification_models:
  # Pipeline object is created to perform model training and evaluate the performance of each model.
  model_pipeline = Pipeline([('model_training', model)])
  model_pipeline.fit(vectors_train, train_y_enc)

  model_name = model
  if model_name=='SVC' and model.kernel=='rbf': 
    model_name+='RBF kernel'
  
  model_scores.append((model_name,(f'{100*model_pipeline.score(vectors_test, test_y_enc):.2f}%')))

# Create the dataframe for score of each model
df_model_scores = pd.DataFrame(model_scores,columns=['Classification Model','Accuracy Score'])
df_model_scores.sort_values(by='Accuracy Score',axis=0,ascending=False)
df_model_scores.to_csv('results.csv')
import tqdm
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

from utils import *

dataset = load_dataset("lex_glue",'ecthr_a')

train = dataset['train']
val = dataset['validation']
test = dataset['test']

train_X, train_y  = pre_process(train) 
test_X, test_y = pre_process(test)

corpus_train  = create_corpus(train_X)
corpus_test = create_corpus(test_X)

train_X_2d = convert_2d(train_X)
test_X_2d = convert_2d(test_X)


tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words='english', lowercase=False)
tfidf.fit(corpus_train+corpus_test)
vectors_train = tfidf.fit_transform(train_X_2d)
vectors_test = tfidf.transform(test_X_2d)

train_y_enc = MultiLabelBinarizer().fit_transform(train_y)
test_y_enc = MultiLabelBinarizer().fit_transform(test_y)

# # from sklearn.naive_bayes import MultinomialNB 
# svc = MultiOutputClassifier(SVC())
# svc.fit(vectors_train,train_y_enc)
# pred = svc.predict(vectors_test)
# print("SVC Accuracy Score -> ",accuracy_score(pred, test_y_enc)*100)

# Naive = OneVsRestClassifier(naive_bayes.MultinomialNB())
# Naive.fit(vectors_train,train_y_enc)
# # predict the labels on validation dataset
# predictions_NB = Naive.predict(vectors_test)
# # Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, test_y_enc)*100)

classification_models = [
                         OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)),
                         OneVsRestClassifier(SVC(kernel = "linear" ,C=1)),
                         OneVsRestClassifier(SVC(kernel='rbf', C=1)),
                         OneVsRestClassifier(naive_bayes.MultinomialNB()),
                         OneVsRestClassifier(DecisionTreeClassifier()),
                         OneVsRestClassifier(RandomForestClassifier(n_estimators=500)),
                         ]

model_scores = []
for model in tqdm(classification_models):
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
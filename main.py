import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

from datasets import load_dataset

dataset = load_dataset("lex_glue",'ecthr_a')

train = dataset['train']
val = dataset['validation']
test = dataset['test']
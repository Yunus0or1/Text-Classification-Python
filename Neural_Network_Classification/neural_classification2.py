import collections
import pandas
import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector,\
    Bidirectional, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from tensorflow.python.client import device_lib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from googletrans import Translator
import datetime
import mysql.connector
from django.db import connection
import MySQLdb
from tensorflow.python.keras.layers import SpatialDropout1D

url = "final_english_data_tagged.txt"
names = ['message', 'outcome']
dataset = pandas.read_csv(url, names=names,delimiter='\t')

dataset_x = dataset["message"]
dataset_y = dataset["outcome"].values



sentences_train, sentences_test, y_train, y_test = model_selection.train_test_split(
    dataset_x, dataset_y, test_size=0.25, random_state=1000)

vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)



input_dim = X_train.shape[1]


model = Sequential()
model.add(Dense(256, input_dim=input_dim, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=sparse_categorical_crossentropy,metrics=['accuracy'],
              optimizer=Adam(0.005), )


history = model.fit(X_train, y_train,
                   epochs=10,
               verbose=False,
         validation_data=(X_test, y_test),
            batch_size=10)

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

new_line = 'Fire alert in Mohakhali'
test_line_tfidf = vectorizer.transform([new_line])
prediction = model.predict_classes(test_line_tfidf)

print(prediction)



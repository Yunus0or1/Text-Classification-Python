


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 0 = GPU use; -1 = CPU use

import keras
import tensorflow as tf

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 3} )
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.callbacks import EarlyStopping
np.random.seed(32)
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector,\
    Bidirectional, Dropout, LSTM
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.keras.optimizers import Adam



url = "final_english_data_tagged.txt"
names = ['data', 'class']
df = pd.read_csv(url, names=names,delimiter='\t')


train_text, test_text, train_y, test_y = train_test_split(
    df['data'],df['class'],test_size = 0.2)





# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 1300
# This is fixed.
EMBEDDING_DIM = 300


tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
tokenizer.fit_on_texts(df['data'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


X = tokenizer.texts_to_sequences(df['data'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(df['class']).values
print('Shape of label tensor:', Y.shape)





X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state = 42)



model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(35))
model.add(LSTM(100))
model.add(Dense(9, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.005), metrics=['accuracy'])

epochs = 2
batch_size = 100

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])


accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


new_complaint = ['what is the load condition? .']
seq = tokenizer.texts_to_sequences(new_complaint)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
print(pred)
print("Prediction class: ", Y[np.argmax(pred)])

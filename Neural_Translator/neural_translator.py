import collections
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
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r",encoding="utf-8") as f:
        data = f.read()

    return data.split('\n')

def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer


def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')


def preprocess(x, y):

    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

def logits_to_text(logits, tokenizer):

    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

    learning_rate = 0.005

    model = Sequential()
    model.add(GRU(256, input_shape=input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size+1, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),)
    return model


def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):

    learning_rate = 0.003

    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences=True), input_shape=input_shape[1:]))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size+1, activation='softmax')))

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate))
    return model


banglish_sentences = load_data('bng.txt')
bangla_sentences = load_data('bn.txt')

preproc_banglish_sentences, preproc_bangla_sentences, banglish_tokenizer, bangla_tokenizer = \
    preprocess(banglish_sentences, bangla_sentences)

tmp_x = pad(preproc_banglish_sentences, preproc_bangla_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_bangla_sentences.shape[-2], 1))
bdrnn_model = bd_model(
    tmp_x.shape,
    preproc_bangla_sentences.shape[1],
    len(banglish_tokenizer.word_index) + 1,
    len(bangla_tokenizer.word_index) + 1)

print(bdrnn_model.summary())
bdrnn_model.fit(tmp_x, preproc_bangla_sentences, batch_size=200, epochs=50, validation_split=0.2)


unknown_sentence = ['onek traffic bimanbondor area te']
pre_proc_unknown_sentence= banglish_tokenizer.texts_to_sequences(unknown_sentence)
tmp_x_un = pad(pre_proc_unknown_sentence, preproc_bangla_sentences.shape[1])
tmp_x_un = tmp_x_un.reshape((-1, preproc_bangla_sentences.shape[-2], 1))
print("Prediction:")
print(logits_to_text(bdrnn_model.predict(tmp_x_un[0:1])[0], bangla_tokenizer))

print(bdrnn_model.evaluate(tmp_x, preproc_bangla_sentences,verbose=0))
#
# print("\nCorrect Translation:")
# print(bangla_sentences[100:101])
#
# print("\nOriginal text:")
# print(banglish_sentences[100:101])
#
# print("Prediction:")
# print(logits_to_text(bdrnn_model.predict(tmp_x[100:101])[0], bangla_tokenizer))
#
# print("\nCorrect Translation:")
# print(bangla_sentences[100:101])
#
# print("\nOriginal text:")
# print(banglish_sentences[100:101])


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import SpatialDropout1D, Conv1D
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from nltk.corpus import stopwords

np.random.seed(32)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 150

epochs = 3
batch_size = 64


def get_stop_words():
    inputFile = open("custom_stop_words.txt", "r", encoding="utf-8")

    word_list = []
    for line in inputFile:
        word = line.split()

        for i in range(0, len(word)):
            if (word[i] not in word_list):
                word_list.append(word[i])
            else:
                pass

    inputFile.close()
    return word_list


def load_data():
    remove_word_list = get_stop_words()
    pattern1 = r'\b(?:{})\b'.format('|'.join(remove_word_list))
    stop_words = stopwords.words('english')
    pattern2 = r'\b(?:{})\b'.format('|'.join(stop_words))

    url = "final_english_data_tagged.txt"
    names = ['data', 'class']
    dataset = pd.read_csv(url, names=names, delimiter='\t')
    dataset["data"] = dataset['data'].str.replace(pattern1, '')
    dataset["data"] = dataset['data'].str.replace(pattern2, '')

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    tokenizer.fit_on_texts(dataset["data"].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print(word_index)

    dataset_x = tokenizer.texts_to_sequences(dataset["data"].values)
    dataset_x = pad_sequences(dataset_x, maxlen=MAX_SEQUENCE_LENGTH)
    dataset_y = pd.get_dummies(dataset["class"]).values

    return dataset_x, dataset_y, tokenizer


def get_model(dataset_x):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=dataset_x.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.005), metrics=['accuracy'])

    return model


def simple_model(dataset_x):
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=dataset_x.shape[1]))
    model.add(Conv1D(300, 10, activation='relu'))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.005), metrics=['accuracy'])

    return model


def predict_unknown(tokenizer):
    new_complaint = ['কুড়িল রোডে ভয়াবহ যানজট। সবাই এড়িয়ে চলুন।']
    seq = tokenizer.texts_to_sequences(new_complaint)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = model.predict(padded)
    labels = ['1', '2', '3', '6', '7']
    print(pred)
    print("Prediction class: ", labels[np.argmax(pred)])


def plot_model():
    print(model.metrics_names)
    plt.plot(history.history['acc'], color="#1f77b4")
    plt.plot(history.history['val_acc'], color="#17becf")
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'], color="#1f77b4")
    plt.plot(history.history['val_loss'], color="#17becf")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


dataset_x, dataset_y, tokenizer = load_data()
X_train, X_test, Y_train, Y_test = train_test_split(dataset_x, dataset_y,
                                                    test_size=0.25, random_state=42, shuffle=True)

model = simple_model(dataset_x)
history = model.fit(X_train, Y_train, epochs=epochs,
                    batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

predict_unknown(tokenizer)

# Test Sentences
# আগপাড়ায় ভয়াবহ আগুন লেগেছে। এডিয়ে চলুন
# কুড়িল রোডের অবস্থা কি কেউ বলতে পারেন?
# কুড়িল রোডের অবস্থা কি কেউ বলতে পারেন?
# কুড়িল রোডে ভয়াবহ যানজট। সবাই এড়িয়ে চলুন।

# Data Labels
# 1- Traffic Jam
# 2- No Traffic Jam
# 3- Road Condition
# 6- Accident
# 7- Fire

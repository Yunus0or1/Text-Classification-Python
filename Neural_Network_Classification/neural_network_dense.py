import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer


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



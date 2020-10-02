import pandas
from nltk.corpus import stopwords
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


def get_all_predictions(model_classes, predictions):
    class_merge_prediction = []

    for i in range(0, len(predictions[0])):
        class_merge_prediction.append([model_classes[i], predictions[0][i]])

    return sorted(class_merge_prediction, key=lambda x: x[1], reverse=True)


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


remove_word_list = get_stop_words()
pattern1 = r'\b(?:{})\b'.format('|'.join(remove_word_list))
stop_words = stopwords.words('english')
pattern2 = r'\b(?:{})\b'.format('|'.join(stop_words))

url = "final_english_data_tagged.txt"
names = ['data', 'class']

dataset = pandas.read_csv(url, names=names, delimiter='\t')
dataset["data"] = dataset['data'].str.replace(pattern1, '')
dataset["data"] = dataset['data'].str.replace(pattern2, '')

X = dataset['data']
Y = dataset['class']

X_train, X_validation, \
Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.3, random_state=2, shuffle=True)

cv = TfidfVectorizer(min_df=1)
X_train_cv = cv.fit_transform(X_train)
X_validation_cv = cv.transform(X_validation)

models = []
models.append(('MultinomialNB', MultinomialNB()))
models.append(('LogisticRegression', LogisticRegression()))
models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=3)))
# models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
# models.append(('SVC', SVC()))
# evaluate each model in turn
results = []
names = []

seed = 7
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train_cv, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    accuracy = (cv_results.mean()) * 100
    deviation = cv_results.std() * 100
    print(name, '[ Accuracy : ', accuracy, '% Deviation : ', deviation, '% ]')

    model.fit(X_train_cv, Y_train)
    predictions = model.predict(X_validation_cv)
    print(name, " Accuracy : ", accuracy_score(Y_validation, predictions) * 100, "%")
    print(classification_report(Y_validation, predictions))

classifier = MultinomialNB()
classifier.fit(X_train_cv, Y_train)
predictions = classifier.predict(X_validation_cv)
print('Accuracy :', accuracy_score(Y_validation, predictions))
print('Confusion Matrix: \n', confusion_matrix(Y_validation, predictions))
print('Report: \n', classification_report(Y_validation, predictions))

# New data insert
new_data = ' কুড়িল রোডে ভয়াবহ যানজট। সবাই এড়িয়ে চলুন।'
new_data_cv = cv.transform([new_data])
predictions = classifier.predict(new_data_cv)
print('Given Data : ', new_data)
print('Prediction on given data : ', predictions)
predictions = classifier.predict_proba(new_data_cv)
model_classes = classifier.classes_
class_merge_prediction = get_all_predictions(model_classes, predictions)
print(class_merge_prediction)

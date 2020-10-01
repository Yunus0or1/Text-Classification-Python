import pandas
import pickle
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.svm import SVC


def covert_to_wrong_word_and_character_position(wrong_word):

    temp_wrong_word = ""
    for i in range(0, len(wrong_word)):
        if i < (len(wrong_word) - 1):
            temp_wrong_word = temp_wrong_word + wrong_word[i] + wrong_word[i] + ' ' + wrong_word[i]  + str(i) + ' '
        else:
            temp_wrong_word = temp_wrong_word + wrong_word[i] + wrong_word[i] + ' ' + wrong_word[i]  + str(i)

    return temp_wrong_word


def covert_to_wrong_word_and_no_character_position(wrong_word):

    temp_wrong_word = ""
    for i in range(0, len(wrong_word)):
        if i < (len(wrong_word) - 1):
            temp_wrong_word = temp_wrong_word + wrong_word[i] + wrong_word[i] + ' '
        else:
            temp_wrong_word = temp_wrong_word + wrong_word[i] + wrong_word[i] + ' '

    return temp_wrong_word


def get_all_predictions(model_classes, predictions):
    class_merge_prediction = []

    for i in range(0, len(predictions[0])):
        class_merge_prediction.append([model_classes[i], predictions[0][i]])

    return sorted(class_merge_prediction, key=lambda x: x[1], reverse=True)


def predict_data(choice, wrong_word, main_word):

    url = "mldata.txt"
    column_names = ['wbt','cbt','acbt','correct_word']
    dataset = pandas.read_csv(url, names=column_names)

    if(choice == '1'):
        dataset_x = dataset["wbt"].values.astype('U')
    elif(choice == '2'):
        dataset_x = dataset["cbt"].values.astype('U')
    elif(choice == '3'):
        dataset_x = dataset["acbt"].values.astype('U')

    dataset_y = dataset["correct_word"].values.astype('U')



    cv = TfidfVectorizer(stop_words='english')

    x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset_x, dataset_y,
                                                                        test_size=0.2, random_state=2)

    x_train_cv = cv.fit_transform(x_train)
    x_test_cv = cv.transform(x_test)
    wrong_word_tfidf = cv.transform([wrong_word])

    feature_names = cv.get_feature_names()
    print(feature_names)
    print("Total Features: " + str(len(feature_names)))


    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('MNB',MultinomialNB()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RFC', RandomForestClassifier(n_estimators= 5)))
    models.append(('LRNew', LogisticRegression(solver='liblinear',multi_class='ovr')))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle= True)
        cv_results = model_selection.cross_val_score(model, x_train_cv, y_train,
                                                     cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: Mean : %f  STD : (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

        model.fit(x_train_cv, y_train)

        predictions = model.predict(x_test_cv)
        print(name, " Accuracy : ", accuracy_score(y_test, predictions) * 100, "%")
        print(classification_report(y_test, predictions))

        print("Saved model")
        filename = name + '.sav'
        pickle.dump(model, open(filename, 'wb'))

        model.fit(x_train_cv, y_train)
        prediction = model.predict(wrong_word_tfidf)
        print("prediction on '" + main_word + "' by " + name + " is::")
        print(prediction)

        predictions = model.predict_proba(wrong_word_tfidf)
        model_classes = model.classes_
        class_merge_prediction = get_all_predictions(model_classes, predictions)
        print(class_merge_prediction)


def predict_using_saved_model(model_file_name,wrong_word):
    url = "mldata.txt"
    names = ['wrong_word', 'custom_wrong_word','correct_word']
    dataset = pandas.read_csv(url, names=names)
    dataset_x = dataset["custom_wrong_word"].values.astype('U')
    dataset_y = dataset["correct_word"].values.astype('U')
    cv = TfidfVectorizer(stop_words='english')
    x_train, x_test, y_train, y_test = model_selection.train_test_split(dataset_x, dataset_y,
                                                                        test_size=0.2, random_state=2)
    cv.fit_transform(x_train)
    wrong_word_tfidf = cv.transform([wrong_word])

    fileName = model_file_name+'.sav'
    loaded_model = pickle.load(open(fileName, 'rb'))

    predictions = loaded_model.predict_proba(wrong_word_tfidf)
    model_classes = loaded_model.classes_

    prediction = loaded_model.predict(wrong_word_tfidf)
    print("prediction is::  " + str(prediction[0]))

    print("ALL PREDICTIONS")
    class_merge_prediction = get_all_predictions(model_classes,predictions)
    print(class_merge_prediction)




if __name__ == '__main__':

    print("1. WBT ML processing")
    print("2. CBT ML processing")
    print("3. ACBT ML processing")

    choice1 = input()

    print("Insert wrong word")
    word = input()
    print("Wrong word is: " + word)

    if(choice1 == '1'):
        wrong_word = word
        print("1. NON Saved Model processing")
        print("2. Saved Model processing")

        choice2 = input()

        if(choice2 == '1'):
            predict_data(choice1,wrong_word, word)
        else:
            print("Insert model name")
            model_file_name = input()
            predict_using_saved_model(model_file_name,wrong_word)


    elif(choice1 == '2'):
        wrong_word = covert_to_wrong_word_and_no_character_position(word)
        print("1. NON Saved Model processing")
        print("2. Saved Model processing")

        choice2 = input()

        if(choice2 == '1'):
            predict_data(choice1,wrong_word, word)
        else:
            print("Insert model name")
            model_file_name = input()
            predict_using_saved_model(model_file_name,wrong_word)

    elif (choice1 == '3'):
        wrong_word = covert_to_wrong_word_and_character_position(word)
        print("1. NON Saved Model processing")
        print("2. Saved Model processing")

        choice2 = input()

        if (choice2 == '1'):
            predict_data(choice1,wrong_word, word)
        else:
            print("Insert model name")
            model_file_name = input()
            predict_using_saved_model(model_file_name, wrong_word)







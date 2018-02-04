# -*- coding: utf-8 -*-
'''
@author: Kavitha Rajendran
'''
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv(r"PhishingData.arff",names=["SFH","popUpWidnow","SSLfinal_State","Request_URL","URL_of_Anchor","web_traffic","URL_Length","age_of_domain","having_IP_Address","Result"])
featureVector = df[["SFH","popUpWidnow","SSLfinal_State","Request_URL","URL_of_Anchor","web_traffic","URL_Length","age_of_domain","having_IP_Address"]].values
classVector = df[["Result"]].values

#test data split 
kf = KFold(n_splits=10, shuffle=True)

count = 0


#Decision Tree
print("\nDecision Tree Classifier:")
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    dt_classifier = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
            max_features=None, max_leaf_nodes=60, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1000, splitter='best')
    dt_classifier.fit(X_train,y_train)
    predictions = dt_classifier.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of Perceptron Classifier:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))

#Perceptron
accuracy = 0
print("\nPerceptron Classifier:")
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    perceptron_classifier = Perceptron(penalty='l2', 
                               alpha=0.0001, fit_intercept=True, shuffle=False, verbose=0, eta0=1.0, random_state=1, class_weight=None, warm_start=True)
    perceptron_classifier.fit(X_train,y_train)
    predictions = perceptron_classifier.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of Perceptron Classifier:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))

#Neural network
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(25,25,25,25,25,25,20,20),max_iter=5000,random_state=1, activation='relu',learning_rate='constant', learning_rate_init=0.0001, shuffle=False, momentum=0.9, early_stopping=False)
    #mlp=MLPClassifier(hidden_layer_sizes=(30,30,30,30,30,30), activation='relu', solver='sgd', alpha=0.00001, learning_rate='constant', learning_rate_init=0.001, 
    #                  power_t=0.5, max_iter=5000, shuffle=True, random_state=None, tol=0.0001, 
#warm_start=False, momentum=0.9, early_stopping=False, validation_fraction=0.1, epsilon=1e-08)
    #mlp=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5,5,5),max_iter=1000,random_state=1)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of Neural network:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))

#SVM
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    svc = svm.SVC(C=2.0, kernel='poly', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=1000, decision_function_shape='ovo', random_state=100)
    #mlp=MLPClassifier(hidden_layer_sizes=(30,30,30,30,30,30), activation='relu', solver='sgd', alpha=0.00001, learning_rate='constant', learning_rate_init=0.001, 
    #                  power_t=0.5, max_iter=5000, shuffle=True, random_state=None, tol=0.0001, 
#warm_start=False, momentum=0.9, early_stopping=False, validation_fraction=0.1, epsilon=1e-08)
    #mlp=MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5,5,5),max_iter=1000,random_state=1)
    svc.fit(X_train,y_train)
    predictions = svc.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of SVM:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))


#Gaussian Naive Bayes
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    nb = GaussianNB()
    nb.fit(X_train,y_train)
    predictions = nb.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of Gaussian Naive Bayes:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))

#LogisticRegression
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    lr = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    lr.fit(X_train,y_train)
    predictions = lr.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of Logistic Regression:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))

#k-Nearest Neighbors
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=50, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=15, p=2,weights='distance')
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of k-Nearest Neighbors:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))

#BaggingClassifier
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    bag = BaggingClassifier(base_estimator=None, n_estimators=25, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=100, verbose=0)
    bag.fit(X_train,y_train)
    predictions = bag.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of Bagging Classifier:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))


#RandomForestClassifier
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    randF = RandomForestClassifier(n_estimators=25, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    randF.fit(X_train,y_train)
    predictions = randF.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of RandomForestClassifier:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))


#AdaBoostClassifier
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    AdaBoost = AdaBoostClassifier(base_estimator=None, n_estimators=30, learning_rate=1.0, algorithm='SAMME.R', random_state=100)
    AdaBoost.fit(X_train,y_train)
    predictions = AdaBoost.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of AdaBoostClassifier:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))

#GradientBoostingClassifier
accuracy = 0
for train,test in kf.split(df):
    count += 1
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    gradBoost = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=60, subsample=0.8, criterion='friedman_mse', min_samples_split=100, min_samples_leaf=40, min_weight_fraction_leaf=0.0, max_depth=9,init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    gradBoost.fit(X_train,y_train)
    predictions = gradBoost.predict(X_test)
    accuracy += accuracy_score(predictions, y_test)
print("Accuracy of AdaBoostClassifier:",(accuracy/10)*100)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))
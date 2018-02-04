# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:49:56 2017

@author: kavit
"""
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
accuracyDict = {}
accuracyDict['DT']=0
accuracyDict['perceptron']=0
accuracyDict['mlp']=0
accuracyDict['svm']=0
accuracyDict['nb']=0
accuracyDict['lr']=0
accuracyDict['gradBoost']=0
accuracyDict['knn']=0
accuracyDict['bag']=0
accuracyDict['randF']=0
accuracyDict['AdaBoost']=0
    
for train,test in kf.split(df):
    X_train, X_test = featureVector[train], featureVector[test]
    y_train, y_test = classVector[train], classVector[test]
    #preprocessing
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    count += 1
    
    #Decision Tree
    print("\nDecision Tree Classifier:")
    dt_classifier = tree.DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10,
            max_features=None, max_leaf_nodes=60, min_samples_leaf=5,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=1000, splitter='best')
    dt_classifier.fit(X_train,y_train)
    predictions = dt_classifier.predict(X_test)
    accuracyDict['DT']+=accuracy_score(predictions, y_test)
    count += 1
    #Perceptron
    print("\nPerceptron Classifier:")
    perceptron_classifier = Perceptron(penalty='l2', 
                               alpha=0.0001, fit_intercept=True, shuffle=False, verbose=0, eta0=1.0, random_state=1, class_weight=None, warm_start=True)
    perceptron_classifier.fit(X_train,y_train)
    predictions = perceptron_classifier.predict(X_test)
    accuracyDict['perceptron']+=accuracy_score(predictions, y_test)

    print("\nNeural network:")
    #Neural network
    count += 1
    mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(25,25,25,25,25,25,20,20),max_iter=5000,random_state=1, activation='relu',learning_rate='constant', learning_rate_init=0.0001, shuffle=False, momentum=0.9, early_stopping=False)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    accuracyDict['mlp']+= accuracy_score(predictions, y_test)


    #SVM
    print("\nSVM:")
    svc = svm.SVC(C=2.0, kernel='poly', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=1000, decision_function_shape='ovo', random_state=100)
    svc.fit(X_train,y_train)
    predictions = svc.predict(X_test)
    accuracyDict['svm']+= accuracy_score(predictions, y_test)

    #Gaussian Naive Bayes
    print("\nGaussianNB:")
    nb = GaussianNB()
    nb.fit(X_train,y_train)
    predictions = nb.predict(X_test)
    accuracyDict['nb']+= accuracy_score(predictions, y_test)

    #LogisticRegression
    print("\nLogisticRegression:")
    lr = LogisticRegression(penalty='l1', dual=False, tol=0.001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
    lr.fit(X_train,y_train)
    predictions = lr.predict(X_test)
    accuracyDict['lr']+= accuracy_score(predictions, y_test)

    #k-Nearest Neighbors
    print("k-Nearest Neighbors")
    knn = KNeighborsClassifier(algorithm='ball_tree', leaf_size=50, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=15, p=2,weights='distance')
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    accuracyDict['knn']+= accuracy_score(predictions, y_test)


    #BaggingClassifier
    print("BaggingClassifier")
    bag = BaggingClassifier(base_estimator=None, n_estimators=25, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=True, oob_score=False, warm_start=False, n_jobs=1, random_state=100, verbose=0)
    bag.fit(X_train,y_train)
    predictions = bag.predict(X_test)
    accuracyDict['bag']+= accuracy_score(predictions, y_test)


    #RandomForestClassifier
    print("RandomForestClassifier")
    randF = RandomForestClassifier(n_estimators=25, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    randF.fit(X_train,y_train)
    predictions = randF.predict(X_test)
    accuracyDict['randF']+= accuracy_score(predictions, y_test)

    #AdaBoostClassifier
    print("AdaBoostClassifier")
    AdaBoost = AdaBoostClassifier(base_estimator=None, n_estimators=30, learning_rate=1.0, algorithm='SAMME.R', random_state=100)
    AdaBoost.fit(X_train,y_train)
    predictions = AdaBoost.predict(X_test)
    accuracyDict['AdaBoost'] += accuracy_score(predictions, y_test)

    #GradientBoostingClassifier
    print("GradientBoostingClassifier")
    gradBoost = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=60, subsample=0.8, criterion='friedman_mse', min_samples_split=100, min_samples_leaf=40, min_weight_fraction_leaf=0.0, max_depth=9,init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
    gradBoost.fit(X_train,y_train)
    predictions = gradBoost.predict(X_test)
    accuracyDict['gradBoost'] += accuracy_score(predictions, y_test)

for key in accuracyDict:
    print(key,":")
    print((accuracyDict[key]/10)*100)
    
'''
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))
print("\nOther Metrics:")
print(classification_report(y_test,predictions))
'''
# -*- coding: utf-8 -*-
'''
Created on 12-Oct-2017

@author: Kavitha Rajendran
'''
from sys import argv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


if __name__ == '__main__':
    if(argv.__len__()<2):
        print("Plese enter training data file path")
    else:
        print("rawTrainingData: ",argv[1])
        rawTrainingData = argv[1]
        #df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",names=["buying","maint","doors","persons","lug_boot","safety","class"])
        #featureVector = df[["buying","maint","doors","persons","lug_boot","safety"]]
        #df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",names=["sepal length in cm","sepal width in cm","petal length in cm","petal width in cm","class"])
        #featureVector = df[["sepal length in cm","sepal width in cm","petal length in cm","petal width in cm"]]
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",names=["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","class"])
        #pd.to_numeric(df)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X.apply(LabelEncoder().fit_transform)

        featureVector = df[["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]]
        #print(pd.to_numeric(featureVector))
        classVector = df[["class"]]
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit_transform(featureVector)
        enc = OneHotEncoder()
        enc.fit(featureVector)  
        #scaler = StandardScaler()
        #scaler.fit(featureVector.convert_objects(convert_numeric=True))
        print(featureVector)
        #featureVector = scaler.transform(featureVector.convert_objects(convert_numeric=True))
        
        

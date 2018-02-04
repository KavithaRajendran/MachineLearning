# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 02:07:29 2017

@author: Kavitha Rajendran
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, Imputer
import math
class Preprocessing:
    
    def __init__(self, sf):
        df = pd.read_csv("C:/Users/kavit/OneDrive/Documents/Semester4/CS6375/ANN/Cars.csv")
        #df = pd.read_csv("C:/Users/kavit/OneDrive/Documents/Semester4/CS6375/ANN/Iris.csv")
        #data split
        self.splitingFactor = sf
        #print(df.shape)
        
        rows = math.ceil(df.shape[0]*self.splitingFactor)
        self.cleanupTrainingData(df[:rows])
        self.cleanupTestData(df[rows:])
        
        self.cleanupTrainingData(df)
        #self.preproc(df)
    
    def preproc(self,X):
        labelEncoder = LabelEncoder()
        numberOfInstances,numOfColumns = X.shape
        for i in range(numOfColumns):
            column = (X.iloc[:, i])
            column = labelEncoder.fit_transform(column)
            X.iloc[:, i] = column
        X=self.handlingMissingData(X)
        #print(X)
        X=self.standardize(X)
        print(X)
        a = np.asarray(X)
        np.savetxt("out.csv", a, delimiter=",")
        #X.to_csv('out.csv',sep=',')
        #print(X)
        
    #Encoding categorical data
    def encodeClassData(self, y):
        labelEncoder_y = LabelEncoder()
        y = labelEncoder_y.fit_transform(y)
        self.handlingMissingData(y)
        return y
    
    def encodeFeatureData(self, X):
        numberOfInstances,numOfFeatureColumns = X.shape 
        labelEncoder_X = LabelEncoder()
        for i in range(numOfFeatureColumns):
            colume = (X.iloc[:, i])
            colume = labelEncoder_X.fit_transform(colume)
            X.iloc[:, i] = colume
        X = self.handlingMissingData(X)
        X = self.standardize(X)
        print(X)
        return X
    
    #fill the missing values with mean
    def handlingMissingData(self, data):
        imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)
        imputer.fit(data)
        data = imputer.transform(data)
        return data
        
    #Feature Scaling
    def standardize(self, featureVector):
        sc_X = StandardScaler()
        featureVector = sc_X.fit_transform(featureVector)
        return featureVector
    
    #preprocess training data
    def cleanupTrainingData(self, train):
        self.X_train = train.iloc[:,:-1]
        self.y_train = train.iloc[:, -1]
        '''
        #drop rows with missing data
        self.X_train.dropna(inplace=True)
        self.y_train.dropna(inplace=True)
        '''
        self.y_train = self.encodeClassData(self.y_train.values)
        self.X_train = self.encodeFeatureData(self.X_train)
        self.X_train = self.standardize(self.X_train)
        print("Training Data after preprocessing:")
        print("feature vector:")
        print(self.X_train)
        print("target value:")
        print(self.y_train)
        
    #preprocess testing data
    def cleanupTestData(self, test):
        self.X_test = test.iloc[:,:-1]
        self.y_test = test.iloc[:, -1]
        self.y_test = self.encodeClassData(self.y_test.values)
        self.X_test = self.encodeFeatureData(self.X_test)
        self.X_test = self.standardize(self.X_test)
        print("Testing Data after preprocessing:")
        print("feature vector:")
        print(self.X_test)
        print("target value:")
        print(self.y_test)
        
if __name__ == '__main__':
    Preprocessing(0.7)
    
    

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:05:28 2017

@author: Kavitha Rajendran
"""
import sys
from HiddenLayer import HiddenLayer
import pandas as pd
from Neuron import Neuron
import numpy as np
import math

class NeuronNetwork:
    
    def __init__(self):
        #creating networks
        inputFile = sys.argv[1]
        trainingPercent = sys.argv[2]
        self.maxIteration = sys.argv[3]
        self.noOfHidLayers = int(sys.argv[4]) #number of hidden layers
        self.hiddenLayerList =  list() #neuron count in each hidden layer
        self.hiddenLayer = list() #hiddenlayer objects
        
        self.df = pd.read_csv(inputFile)
        inputCount = (self.df.shape[1])-1
        self.numberOfInstances = self.df.shape[0]
        
        self.classLabel = (self.df.iloc[:,-1]).values
        self.predictedClass = list()
        
        #creating hidden Layers
        for i in range(self.noOfHidLayers):
            self.hiddenLayerList.append(int(sys.argv[4+i+1]))
        
        for i in range(self.noOfHidLayers):
            #print(self.hiddenLayerList[i])
            self.hiddenLayer.append(HiddenLayer(self.hiddenLayerList[i],inputCount))
            inputCount=self.hiddenLayerList[i]
            
        tempWeightVector=list()
        for i in range(int(sys.argv[4+i+1])):
            #tempWeightVector.append(random.random())
            tempWeightVector.append(np.random.uniform(-1,1))
        print("tempWeightVector",tempWeightVector)
        self.outputNeuron=Neuron(np.random.uniform(-1,1),tempWeightVector)
        
    def fwdPass(self):
        #for every instance
        #for i in range(2):
        self.predictedClass=list()
        for i in range(self.numberOfInstances):
            inputVector = self.df.iloc[i,:-1].values
            #print(inputVector)
            #for every hidden layer
            for j in range(self.noOfHidLayers):
                #for every neuron in a hidden layer
                tempInputVector=list()
                for k in range(self.hiddenLayer[j].numOfNeurons):
                    self.hiddenLayer[j].neurons[k].inputVector=inputVector
                    tempInputVector.append(self.hiddenLayer[j].neurons[k].sigmoidFunction())
                inputVector=list()
                inputVector=tempInputVector
            #predicting label
            for a in range(self.hiddenLayer[-1].numOfNeurons):
                #self.outputNeuron.weightVector=list()
                self.outputNeuron.inputVector.append(self.hiddenLayer[-1].neurons[a].output)
            #print("target neuron input:",self.outputNeuron.inputVector)
            #print("target neuron weights:",self.outputNeuron.weightVector)
            #print("target neuron bias:",self.outputNeuron.bias)
            #print("target neuron output before sigmoid:",self.outputNeuron.output)
            self.outputNeuron.sigmoidFunction()
            self.predictedClass.append(self.outputNeuron.output)
            self.outputNeuron.inputVector=list()
            self.outputNeuron.output=0
            
            #print("target neuron output after sigmoid:",self.outputNeuron.output)
            #self.printNetwork()
            
    def printNetwork(self):
        for j in range(self.noOfHidLayers):
            #for every neuron in a hidden layer
            print("Hidden Layer:",j)
            for k in range(self.hiddenLayer[j].numOfNeurons):
                print("Neuron:",k)
                print("inputs:",self.hiddenLayer[j].neurons[k].inputVector)
                print("outputs:",self.hiddenLayer[j].neurons[k].output)
                print("weights:",self.hiddenLayer[j].neurons[k].weightVector)
                print("bias:",self.hiddenLayer[j].neurons[k].bias)
        print("predictedClass:")
        print(self.predictedClass)

    def calculateError(self):
        print("ground truth --",len(self.classLabel))
        print("predicted --",len(self.predictedClass))
        sqrError=0
        for i in range(len(self.predictedClass)):
            diff=(float(self.classLabel[i]))-(float(self.predictedClass[i]))
            sqrError+=math.pow(diff,2)
        finalError = sqrError/(2*self.numberOfInstances)
        print("finalError:",finalError)
        
if __name__ == '__main__':
    nn = NeuronNetwork()
    for i in range(int(nn.maxIteration)):
        nn.fwdPass()
        nn.printNetwork()
        print("Iteration:",i)
        nn.calculateError()
    
    
        
                
                
    
        

             
    
    


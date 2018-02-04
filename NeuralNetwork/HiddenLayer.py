# -*- coding: utf-8 -*-
"""
@author: Kavitha Rajendran
"""

import random
from Neuron import Neuron
import numpy as np
#import NeuronNetwork

class HiddenLayer():
    def __init__(self, numOfNeurons, inputCount):
        #bias per hidden layer generated randomly
        self.bias = random.random()
        self.numOfNeurons = numOfNeurons
        self.neurons = []
        self.inputVector = list()
        self.weightVector=list()
        
        for i in range(self.numOfNeurons):
            for i in range(inputCount):
                #print("w:",i)
                #self.weightVector.append(random.random())
                self.weightVector.append(np.random.uniform(-1,1))
            #print(self.weightVector)
            self.neurons.append(Neuron(self.bias,self.weightVector))
            self.weightVector=list()
            
             
    
        
        
    
# -*- coding: utf-8 -*-
"""
@author: Kavitha Rajendran
"""

import math

class Neuron():
    def __init__(self, bias, weights):
        self.bias = bias
        self.weightVector = weights
        self.inputVector = list()
        #self.deltaWeight=[]
        self.output=0
    
    #Calculating net for forward pass
    def calculateNet(self):
        #calculating net value
        total = 0
        #print(self.weightVector)
        #print(self.inputVector)
        for i in range(len(self.weightVector)):
            #print(i)
            total += self.inputVector[i] * self.weightVector[i]
        return total + self.bias
    
    #sigmoid calculation 
    def sigmoidFunction(self):
        #calculating sigmoid value
        total_net_input = self.calculateNet()
        self.output = 1 / (1 + math.exp(-total_net_input))
        return self.output
    
    
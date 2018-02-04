
# coding: utf-8

# In[77]:

from sklearn.neural_network import MLPClassifier


# In[78]:

X = [[0, 0],[0, 1],[1, 0],[1, 1]]


# In[79]:

y = [0, 1, 1, 0]


# In[95]:

clf = MLPClassifier(hidden_layer_sizes=(4, 2),solver='lbfgs', alpha=1e-5, max_iter=5000, random_state=1)


# In[96]:

clf.fit(X,y)


# In[97]:

clf.score(X,y)


# In[98]:

clf.predict(X)






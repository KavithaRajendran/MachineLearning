
# coding: utf-8

# In[4]:

import pandas as pd


# In[15]:

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",names=["class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"])


# In[16]:

df


# In[17]:

X = df[['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']]


# In[18]:

y = df[['class']]


# In[19]:

from sklearn.model_selection import train_test_split


# In[20]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.27)


# In[21]:

from sklearn.preprocessing import StandardScaler


# In[22]:

scaler = StandardScaler()


# In[23]:

scaler.fit(X_train)


# In[24]:

X_train = scaler.transform(X_train)


# In[25]:

X_test = scaler.transform(X_test)


# In[26]:

X_train[1]


# In[27]:

from sklearn.neural_network import MLPClassifier


# In[40]:

mlp = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(30,30,30),max_iter=5000,random_state=1)


# In[41]:

mlp.fit(X_train,y_train)


# In[42]:

predictions = mlp.predict(X_test)


# In[43]:

predictions


# In[44]:

y_test


# In[45]:

from sklearn.metrics import classification_report,confusion_matrix


# In[46]:

print(confusion_matrix(y_test,predictions))


# In[47]:

print(classification_report(y_test,predictions))


# In[ ]:




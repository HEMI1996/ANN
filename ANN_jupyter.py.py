
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')


# In[3]:


dataset.head()


# In[4]:


X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# In[5]:


X.head()


# In[7]:


X


# In[8]:


# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[9]:


labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


# In[10]:


X


# In[11]:


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()


# In[12]:


X


# In[13]:


# Avoding Dummy variable trap
X = X[:, 1:]


# In[14]:


X


# In[15]:


# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[16]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[17]:


sc = StandardScaler()
X_train = sc.fit_trasform(X_train)
X_test = sc.transform(X_test)


# In[18]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[19]:


X_test


# In[20]:


# Importing the Keras libraries and packages
import tensorflow
import keras


# In[21]:


from keras.models import Sequential
from keras.layers import Dense


# In[22]:


# Initialising the ANN
classifier = Sequential()


# In[23]:


# Adding the Input layer and first hidden layer
classifier.add(Dense(ouput_dim=6, init='uniform', activation='relu', input_dim=11))


# In[24]:


# Adding the Input layer and first hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))


# In[25]:


# Adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))


# In[26]:


# Adding the Output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))


# In[27]:


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[28]:


# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# In[33]:


# Predicting the test set results
y_pred = classifier.predict(X_test)


# In[34]:


y_pred


# In[35]:


y_pred = (y_pred > 0.5)


# In[36]:


y_pred


# In[37]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[38]:


cm


# In[39]:


# Test set accuracy
(1481 + 232) / 2000


# In[40]:


## It's a WOW....


#!/usr/bin/env python
# coding: utf-8

# ## MLP for Binary Classification
# 
# In this lab, you will use the Ionosphere data binary (two-class) classification dataset to demonstrate an MLP for binary classification.
# 
# This dataset involves predicting whether a structure is in the atmosphere or not given radar returns.
# 
# The dataset will be downloaded automatically using Pandas, but you can learn more in the links below.
# 
# [Ionosphere Dataset (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv)
# 
# [Ionosphere Dataset Description (csv)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.names)
# 
# 
# Your task for this is lab is to develop a Keras-based Multi-Layer Perceptron model for this data set. Remember the number of output layers is equal to the number of classes.
# 
# Following we have provided some piece of code to you while you need to complete the rest of the code on your own.
# 
# 

# In[10]:


# Importing Libraries
# Your code to import read_csv class from pandas
from pandas import read_csv
# Your code to import train_test_split class from sklearn. 
#Follow link https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# # Read the dataset from the path below. Store the data in a pandas dataframe named 'df'
# 
# Link to API - https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

# In[11]:


path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'

# Your code to read the csv from the above path.
df = read_csv(path, header=None)


# See the sample dataset. Print few rows of the dataset. Use dataframe.head() method.
# 
# Link to API:  https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html

# In[12]:


# Your code to print first few rows of the dataset.
df.head()


# Print the basic info of the dataset. Use dataframe.info() from pandas library
# 

# In[13]:


# Your code to print information about the dataframe
df.info()


# Print the shape of the dataframe. Select suitable API call from the pandas library

# In[14]:


# Your code to print the shape of the dataset
df.shape


# # Separate the input and output from the dataframe. Input is all columns besides last column. Output is the last column.
# 

# In[15]:


X = df.values[:, :-1]
# Your code to get y - Hint y = df.values[:, some parameters]
y = df.values[:, -1]


# We have converted everthing in X to 'float' and the letters in column y to the numbers in the following cell.

# In[16]:


X = X.astype('float32')
y = LabelEncoder().fit_transform(y)


# Printing the genral information of the X and y in the following cell

# In[17]:


# Your code to print X
print(X)

# Your code to print y
print(y)

# your code to print shape of X. Remember X is a numpy array
print(X.shape)

# your code to print shape of y. Remember y is a numpy array
print(y.shape)


# * Separate X and y into training and test set with a ratio of your choice.
# * Print the shapes of the resulting arrays.
# * Get the number of features from X_train. Remember the number of features are the number of inputs.
# 
# Use sklearn train_test_split class.
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# 

# In[18]:


# Your code to separate the data into trauning and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Your code to print shape of X_train
# Your code to print shape of X_test
# Your code to print shape of y_train
# Your code to print shape of X_test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n_features = X_train.shape[1]
n_features


# # Creating a Multi-layer Perceptron using Keras.
# We have added first and last layers. Create the hidden layers of your choise.
# You can chose any number of hidden layers and activation function of your chose
# https://keras.io/api/layers/core_layers/dense/

# In[29]:


model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))


#
# Add as many layers with activation functions of your choice
#
model.add(Dense(8, activation='relu'))



model.add(Dense(1, activation='sigmoid'))



# In[30]:


print(model.summary())


# In[31]:


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# In the next cell, we trained the above neural network model and tested its accuracy. As this concept has still not benn covered in the class, just run the code to check the accuracy.

# In[32]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[33]:


history = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)


# In[35]:


loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)


# In[36]:


import matplotlib.pyplot as plt

history_dict = history.history
Accuracy = history_dict['accuracy']
plt.figure(num=1, figsize=(15,7))
plt.plot(Accuracy, 'bo', label='Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[37]:


model = Sequential()
model.add(Dense(10, activation='gelu', input_shape=(n_features,)))


#
# Add as many layers with activation functions of your choice
#
model.add(Dense(8, activation='gelu'))
model.add(Dense(5, activation='gelu'))


model.add(Dense(1, activation='sigmoid'))


# In[38]:


print(model.summary())


# In[39]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[40]:


history2 = model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)


# In[41]:


loss, acc = model.evaluate(X_test, y_test, verbose=1)
print('Test Accuracy: %.3f' % acc)


# In[ ]:





# In[ ]:





# ** How much accuracy have you got? Compare the accuracy with your peers. **
# ** Now, change your model and activation function to get the better accuracy as compared to your peers **

# ## **Important:** Document in your lab logbook the accuracy of the improved model. Do not include any code or explanations in your lab logbook. Simply record the accuracy. For example, if the obtained accuracy is 0.98, then enter "0.98" in your lab logbook.
# 
# ## In addition to the accuracy, also document the output of the neural network as provided in Task 2.
# 

# 
# Next, we have provided the code to predict on an unknown value.
# We will cover these concepts later in the class. For now, just run the code to see the prediction.

# In[34]:


row = [1,0,0.99539,-0.05889,0.85243,0.02306,
       0.83398,-0.37708,1,0.03760,0.85243,-0.17755,
       0.59755,-0.44945,0.60536,-0.38223,0.84356,
       -0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,
       -0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,
       -0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = model.predict([row])
print('Predicted: %.3f' % yhat)


# ### Try out the same model with Keras Functional models!
# Refer to [Keras](https://keras.io/) for more details and tutorials for the same.

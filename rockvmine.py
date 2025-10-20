import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and Data processing

# loading data set to pandas data frame
sonar_data = pd.read_csv('Copy of sonar data.csv', header =None)
# print(sonar_data.head())
# number of rows and colmns
#print(sonar_data.shape)
#print(sonar_data.describe()) # statistical measures of data 
#print(sonar_data[60].value_counts())
# M is mine R is rock

#print(sonar_data.groupby(60).mean())

# seperating data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
#print(X)
#print(Y)

# Training and test data

X_train, X_test, Y_train, T_test = train_test_split(X, Y, test_size = .1, stratify=Y, random_state = 1)

print(X.shape, X_train.shape, X_test.shape)

# Model training---> logistic regression

model = LogisticRegression()

# training the logistic regression model with training data
model.fit(X_train, Y_train)

# model evaluation
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data :' , training_data_accuracy)
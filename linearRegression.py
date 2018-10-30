import pandas as pd
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

data_file = pd.read_csv('./SalaryData.csv')
X = data_file.iloc[:, 0:1].values
y = data_file.iloc[:, 1].values


# ! split the data with train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0)
# * X_train and y_tain is used for training the model
# * X_test and y_test is for test the model

# ! import the framework for the linear regression
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)
# * the key part is the X_train must in form of [[], [], [], ......]
# * the linear relationship between y_predict and x_test

# ! drawing the linear graph
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regression.predict(X_train), color="blue")
# plt.show()
plt.savefig('linearRegression.png')



 



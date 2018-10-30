import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pyplot
import pprint as pp 

# * obtain the data set successfully 
# *
dataset = pd.read_csv('./50-Startups.csv')                                                                      # dataframe
# ! x must be a twp-dimention array => [:, :-1]
# ! y must be an one-dimension array
# print(type(dataset.iloc[:, :-1]))                                                                             # dataframe
X = dataset.iloc[:, :-1].values                                                                                 # ndarray
y = dataset.iloc[:, -1].values

# * generate dummy variable for the state
# * 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_state = LabelEncoder()
state_encoding = labelEncoder_state.fit_transform(X[:, 3])
# ! 这样可以直接修改第三列
X[:, 3] = state_encoding
onehoterencoder = OneHotEncoder(categorical_features=[3])
# ! 这就是把encoder融进整个matrix里面，不用手动替换
# ! 注意：这边的 X 都是 ndarray
X = onehoterencoder.fit_transform(X).toarray()

# * 获取 cross validation 的东西
# *
from sklearn.model_selection import train_test_split
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# * 开始训练模型
# *
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)                                                                             # <class 'numpy.ndarray'>








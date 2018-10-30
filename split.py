import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint as pp

'''Target : handle missing data'''
data_set = pd.read_csv('Data.csv')
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, 3].values

# how to solve missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
new_data = imputer.transform(X[:, 1:3])
X[:, 1:3] = new_data
# pp.pprint(X)

'''Target : handle categotical data'''
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
label_for_counteries = labelEncoder_X.fit_transform(X[:, 0])
X[:, 0] = label_for_counteries

'''Target : dummy encoder'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
# 将第一列作为feature，来做onehoterencoder
X = oneHotEncoder.fit_transform(X).toarray()
# X = oneHotEncoder.fit_transform(X).toarray()

'''Target : make y encoder'''
labelEncoder_Y = LabelEncoder()
label_for_purchase = labelEncoder_Y.fit_transform(Y)
Y = label_for_purchase
# pp.pprint(label_for_purchase)

'''Target: split the data into training set and test set'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, random_state = 0)
# print(X_train, X_test, Y_train, Y_test)

'''Target: scalling '''
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# pp.pprint(X_train)


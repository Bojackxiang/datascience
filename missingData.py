import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint as pp

data_set = pd.read_csv('Data.csv')
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, -1].values


# how to solve missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
new_data = imputer.transform(X[:, 1:3])
X[:, 1:3] = new_data



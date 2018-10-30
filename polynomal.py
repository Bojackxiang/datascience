import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

data_set = pd.read_csv('./Position_Salaries.csv')
print(data_set)
X = data_set.iloc[:, 1:2].values
y = data_set.iloc[:, 2:3].values

print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
linear_reg = LinearRegression()
linear_reg.fit(X_poly, y)

print(X)
pred_y = linear_reg.predict(X_poly)
print(pred_y)

plt.scatter(X, y, color='red')
plt.plot(X, pred_y, color='green')
plt.show()

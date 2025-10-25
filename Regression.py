# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 22:42:12 2022

@author: pavel
"""

# regressions



# linear regression

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array([i for i in range(1, 6)]).reshape(-1, 1) #has to have one column and more rows, if not reshaped Error would pop up
y = np.random.randint(0, 10, size = 5)


model = LinearRegression()

model.fit(x, y)

rs = model.score(x ,y) #R-squared
model.intercept_ #the intercept
model.coef_ #the slope

plt.plot(x, y)


#statsmodels

pip install statsmodels
import statsmodels.api as sm
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()
model.params
model.predict(x)
a = model.summary()
a

y_line = model.params[0] + model.params[1] * x[:, 1]

plt.plot(x[:,1], y)
plt.plot(y_line, x[:, 1])


#using numpy only

x = np.array([i for i in range(0, 20)])
y = np.random.randint(0, 10, size = 20)

x = np.siye











# multiple regression

x = np.array([i for i in range(1, 10)]).reshape(-1, 1)
y = np.random.randint(0, 10, size = (9, 2))

model = LinearRegression().fit(x, y)
model.score(x, y) #r squared
model.intercept_
model.coef_



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

X = df.iloc[:,1].values
y = df.iloc[:,-1].values

X,y = X.reshape(-1,1),y.reshape(-1,1)
plt.scatter(X,y)



from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X,y)

y_pred = reg.predict([[21.07931]])

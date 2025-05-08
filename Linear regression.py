import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
x = np.array([1, 2, 3, 4, 5]) 
y = np.array([1, 3, 7, 9, 11])
w0 = 1 
w1 = 3
def predict(x, w1, w0):
    Y = (w1 * x) + w0 
    return Y
def mse(act_val, pre_value):
    return np.mean((act_val - pre_value) ** 2)
def compute_gradient(x, y, w1, w0):
    y_pred = predict(x, w1, w0)
    n = len(x)

    dw1 = -2/n * (np.sum((y - y_pred) * (x))) 
    dw0 = -2/n * (np.sum(y - y_pred)) 

    return dw1, dw0
def Update_parameter(w0, w1, dw0, dw1, alpha):
    w0 = w0 - dw0 * alpha
    w1 = w1 - dw1 * alpha

    return w1, w0
epochs = 10000
alpha = 0.0001

for i in range(epochs):
    dw1, dw0 = compute_gradient(x, y, w1, w0)
    w1, w0 = Update_parameter(w0, w1, dw0, dw1, alpha)

    if i % 100 == 0:
        print(f"epochs {i} ", mse(y, predict(x, w1, w0)))

print(f"w0:{w0} and w1:{w1}")
print("Actual value of y : ", y)
print("Predicted value of y : ", predict(x, w1, w0))
from sklearn.linear_model import LinearRegression

model = LinearRegression()
x = np.array(x).reshape(-1, 1)  

print(x)
print(x.shape)

model.fit(x, y)

y_pred = model.predict(x)

print(y_pred)
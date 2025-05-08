# Linear-Regression-Model
# 📈 Linear Regression from Scratch

This project demonstrates how to implement **Linear Regression** from scratch using **NumPy**, and compares it with the implementation from **scikit-learn**.

---

## 🔍 Project Overview

This project includes:
- Manual implementation of **Gradient Descent** to minimize **Mean Squared Error (MSE)**.
- Parameter updates for weights (`w0` and `w1`).
- Visual and printed comparison of predicted vs actual values.
- Final comparison with `sklearn`'s built-in `LinearRegression` model.

---

## 💡 Concepts Covered

- Linear Regression Hypothesis:  
  \[
  \hat{y} = w_1x + w_0
  \]
- Mean Squared Error (MSE)
- Gradient computation
- Learning rate and epoch-based parameter tuning
- Model comparison with scikit-learn

---

## 🛠 Requirements

- Python 3.x
- NumPy
- pandas
- matplotlib
- scikit-learn

Install dependencies using:

```bash
pip install numpy matplotlib pandas scikit-learn
```
## 🚀 How to Run
Clone the repository or copy the script into a .py file.

Make sure all dependencies are installed.

Run the Python script:

python linear_regression_scratch.py
## 📊 Output Example
During training:

epochs 0  14.960000000000003
epochs 100  0.2558210382258023
...
w0:0.728487388515319 and w1:2.045379719499964
Actual value of y :  [ 1  3  7  9 11]
Predicted value of y :  [2.77386711 4.81924683 6.86462655 8.91000628 10.955386  ]
## 🤖 Model Comparison
After training your custom model, the script compares it with scikit-learn’s LinearRegression for validation:

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x, y)
print(model.predict(x))
This ensures your gradient descent approach aligns well with standard libraries.

## 📈 Future Enhancements
Add data visualization for loss over epochs.

Visualize fitted line vs actual points.

Extend to multivariable linear regression.


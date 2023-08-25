import pandas as pd
from sklearn import linear_model

from Linear_Regression.utils import show_regression, numpy_regression

df = pd.read_csv('../Data/regression/cells.csv')
df_clean = df.dropna(axis=0)

x = df_clean["time"]
y = df_clean["cells"]

method = 'sklearn'

if method == 'numpy':
    # Do Linear Regression using just numpy
    y_fit, p = numpy_regression(x, y)
    show_regression(x, y, y_fit, p)

elif method == 'sklearn':
    # Do Linear Regression using sklearn
    x_df = df_clean.drop('cells', axis='columns')
    y_df = df_clean.cells
    reg = linear_model.LinearRegression()
    reg.fit(x_df, y_df)
    p = [float(reg.coef_), float(reg.intercept_)]
    y_fit = reg.predict(x_df)
    show_regression(x, y, y_fit, p)

else:
    raise ValueError(f'Method: {method} is unknown')

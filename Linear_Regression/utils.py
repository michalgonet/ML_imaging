import matplotlib.pyplot as plt
import numpy as np


def show_data(x, y):
    plt.scatter(x, y, color='blue', marker='x')
    plt.xlabel('time [h]')
    plt.ylabel('# cells')
    plt.show()


def show_regression(x, y, yfit, p):
    plt.scatter(x, y, color='blue', marker='x')
    plt.plot(x, yfit, 'r')
    plt.xlabel('time [h]')
    plt.ylabel('# cells')
    plt.title(f'Model: y = {round(p[0], 3)}*x + {round(p[1], 3)}')
    plt.show()


def numpy_regression(x, y):
    p = np.polyfit(x, y, deg=1)
    y_fit = np.polyval(p, x.to_numpy())
    return y_fit, p

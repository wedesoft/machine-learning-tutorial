import numpy as np
import matplotlib
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = np.arange(-5, 5, 0.01)
    y = 1 / (1 + np.exp(-x))
    plt.plot(x, y, label='y=g(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sigmoid function')
    plt.savefig('sigmoid.pdf')

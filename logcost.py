#!/usr/bin/env python3
import numpy as np
import matplotlib
# https://stackoverflow.com/questions/2801882/generating-a-png-with-matplotlib-when-display-is-undefined
matplotlib.use('Agg') # use TKAgg for X.Org output
import matplotlib.pyplot as plt


if __name__ == '__main__':
    z = np.arange(-3, 3, 0.1)
    plt.plot(z, -np.log(1/(1+np.exp(-z))), label='cost when y=1')
    plt.plot(z, -np.log(1-1/(1+np.exp(-z))), label='cost when y=0')
    plt.title('Cost functions')
    plt.legend()
    plt.xlabel('z')
    plt.savefig('logcost.pdf')

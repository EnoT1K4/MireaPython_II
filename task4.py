import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import tree
from numpy import *
from numpy.random import *


def task1():
    delta = 1.0
    x = linspace(-5, 5, 11)
    print(x)
    y = x ** 3 + delta * (rand(11) - 0.5)
    print(y)
    x += delta * (rand(11) - 0.5)
    x.tofile('x_data.txt', '\n')
    y.tofile('y_data.txt', '\n')
    x = fromfile('x_data.txt', float, sep='\n')
    y = fromfile('y_data.txt', float, sep='\n')

    m = vstack((x ** 3, x, ones(11))).T
    s = np.linalg.lstsq(m, y, rcond=None)[0]
    x_prec = linspace(-5, 5, 101)
    plt.plot(x, y, 'D')
    plt.plot(x_prec, s[0] + x_prec ** 3 + s[1] * x_prec + s[2], '-', lw=2)
    plt.grid()
    plt.savefig('x**3.png')


print('Enter num task 1 or 2 or 3')
inp = int(input())
while inp != 0:
    if inp == 1:
        task1()
        break
    elif inp == 2:
        pass
        #task2()
       # break
    elif inp == 3:
        pass
        #task3()
        #break
    else:
        print("Check input")
        break


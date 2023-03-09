import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
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


def task2():
    beta = (0.15, 0.745)

    def f(x, b0, b1):
        return b0 + b1 * x

    xdata = np.linspace(0, 5, 50)
    y = f(xdata, *beta)
    ydata = y + 0.07 * np.random.randn(len(xdata))
    beta_opt, beta_cow = sp.optimize.curve_fit(f, xdata, ydata)
    print(beta_opt)
    lin_dev = sum(beta_cow[0])
    print(lin_dev)
    residuals = ydata - f(xdata, *beta_opt)
    fres = sum(residuals ** 2)
    print(fres)
    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)
    plt.show()


def task3():
    url = 'https://raw.githubusercontent.com/AnnaShestova/salary-years-simple-linear-regression/master/Salary_Data.csv'
    dataset = pd.read_csv(url)
    dataset.head()
    dataset.describe()
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    print(x)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    df = pd.DataFrame({'Actual': y_test, 'Pred': y_pred})
    df.plot(kind='bar')
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    plt.scatter(x_test, y_test, color='gray')
    plt.plot(x_test, y_pred, color='red', linewidth=2)
    plt.show()
    print(regressor.intercept_)
    print(regressor.coef_)


def task4():
    url = 'https://raw.githubusercontent.com/likarajo/petrol_consumption/master/data/petrol_consumption.csv'
    dataset = pd.read_csv(url)
    dataset.head()
    dataset.describe()
    df1 = dataset.iloc[:, :-1].values
    df2 = dataset.iloc[:, 1:2].values
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    df1 = df1.rename(columns={0: 'y'}, inplace=False)
    df2 = df2.rename(columns={0: 'x1'}, inplace=False)
    frames = [df1, df2]
    dataset = pd.concat([df1, df2], axis=1, join="inner")
    dataset.head()
    dataset.describe()
    #print(dataset)
    x = dataset[['x1']]
    y = dataset[['y']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns=['Coeff'])
    print(coeff_df)
    y_pred = regressor.predict(x_test)
    y_test_1 = []
    for i in range(len(y_test.values)):
        y_test_1.append(y_test.values[i][0])
    y_pred_1 = []
    for i in range(len(y_pred)):
        y_pred_1.append(y_pred[i][0])
    df = pd.DataFrame({'Actual': y_test_1, 'Pred': y_pred_1})
    print(df)
    print('Errors', metrics.mean_squared_error(y_test, y_pred))


print('Enter num task 1 or 2 or 3 or 4')
inp = int(input())
while inp != 0:
    if inp == 1:
        task1()
        inp = int(input())
        break
    elif inp == 2:
        task2()
        inp = int(input())
        break
    # break
    elif inp == 3:
        task3()
        inp = int(input())
        break
    elif inp == 4:
        task4()
        break
    else:
        print("Check input")
        break

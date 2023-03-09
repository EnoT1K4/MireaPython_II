import numpy as np
import pandas as pd


def task1():
    x = np.zeros((8, 8), dtype=int)
    x[1::2, 0::2] = 1
    x[0::2, 1::2] = 1
    print(x)


def task2():
    s = np.zeros((5, 5))
    s += np.arange(5)
    print(s)


def task3():
    a = np.random.random((3, 3, 3))
    print(a)


def task4():
    x = np.ones((4, 4))
    x[1:-1, 1:-1] = 0
    print(x)


def task5():
    x = [10, 20, 7, 11, 5, 2, 6]
    x.sort(reverse=True)
    print(x)


def task6():
    a = np.arange(49).reshape(7, 7)
    print(a)
    print(np.shape(a))
    print(np.size(a))
    print(np.ndim(a))


def task21():
    a = pd.Series([2, 4, 6, 8])
    b = pd.Series([1, 3, 5, 7])
    print(sum((a - b) ** 2) ** .5)


def task22():
    import pandas
    url = 'https://github.com/akmand/datasets/raw/main/kobe.csv'
    dataframe = pandas.read_csv(url)
    print(dataframe[dataframe['quarter'] == '1'].head(20))


def task23():
    url = 'https://github.com/akmand/datasets/raw/main/kobe.csv'
    dataframe = pd.read_csv(url)
    a = dataframe[dataframe['quarter'] == '1'].head(20)
    print('Последние три строки:\n\n', a.tail(3), '\n\nПервые три строки:\n\n ',
          a.head(3), '\n\nКоличество строк и столбцов: ', a.shape,
          '\n\n Описательная статистика:\n\n', a.describe()
          , '\n\nВыбор нескольких строк:\n', a.iloc[1:3])


def task32():
    url = 'https://raw.githubusercontent.com/akmand/datasets/master/iris.csv'
    dataframe = pd.read_csv(url)
    minimax = lambda x: (x - x.min()) / (x.max() - x.min())
    dataframe['sepal_length_cm'] = minimax(dataframe['sepal_length_cm'])
    z_scal = lambda x: (x - x.mean()) / x.std()
    dataframe['sepal_width_cm'] = z_scal(dataframe['sepal_width_cm'])
    print(dataframe.head())


print('Enter num task')
inp = int(input())
while inp != 0:
    if inp == 1:
        task1()
        break
    if inp == 2:
        task2()
        break
    if inp == 3:
        task3()
        break
    if inp == 4:
        task4()
        break
    if inp == 5:
        task5()
        break
    if inp == 6:
        task6()
        break
    elif inp == 21:
        task21()
        break
    elif inp == 22:
        task22()
        break
    elif inp == 23:
        task23()
        break
    elif inp == 32:
        task32()
        break
    else:
        print("Check input")

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import tree, metrics
from sklearn.metrics import classification_report, confusion_matrix


def task1():
    class Trigon:
        def __init__(self, chislo):
            self.chislo = int(chislo)

        def print_cos(self):
            print('Cos = {}'.format(math.cos(self.chislo)))

        def print_sin(self):
            print('Sin = {}'.format(math.sin(self.chislo)))

        def print_tan(self):
            print('Tan = {}'.format(math.tan(self.chislo)))

        def print_arcsin(self):
            print('Arcsin = {}'.format(math.asin(self.chislo)))

        def print_arccos(self):
            print('ArcCos = {}'.format(math.acos(self.chislo)))

        def print_arctg(self):
            print('Arctg = {}'.format(math.atan(self.chislo)))

        def rad(self):
            print('Rad = {}'.format(math.radians(self.chislo)))

    print('What u need:')
    inp = input()
    print('Input value')
    a = int(input())
    c = Trigon(a)
    if inp == 'print_cos':
        c.print_cos()
    elif inp == 'print_sin':
        c.print_sin()
    elif inp == 'print_tan':
        c.print_tan()
    elif inp == 'print_arcsin':
        c.print_arcsin()
    elif inp == 'print_arccos':
        c.print_arccos()
    elif inp == 'print_arctg':
        c.print_arctg()
    elif inp == 'print_rad':
        c.rad()


def task2():
    myTree = ['a', ['b', ['d', [], []], ['e', [], []]], ['c', ['f', [], []], []]]
    print(myTree)
    print('left subtree = ', myTree[1])
    print('root = ', myTree[0])
    print('right subtree = ', myTree[2])


def task3():
    class TreeNode:
        def __init__(self, data):
            self.left = None
            self.right = None
            self.data = data

        def print_tree(self):
            if self.left:
                self.left.print_tree()
            print(self.data)
            if self.right:
                self.right.print_tree()

        def insert(self, val):
            if val < self.data:
                if self.left:
                    self.left.insert(val)
                else:
                    self.left = TreeNode(val)
            elif val > self.data:
                if self.right:
                    self.right.insert(val)
                else:
                    self.right = TreeNode(val)

    tree = TreeNode(1)
    tree.left = TreeNode(2)
    tree.right = TreeNode(3)
    tree.left.left = TreeNode(4)
    tree.left.right = TreeNode(5)
    tree.insert(5)
    tree.insert(10)
    tree.print_tree()


def task4():
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    df1 = pd.DataFrame(X)
    target = [0, 0, 0, 1, 1, 1]
    df2 = pd.DataFrame(target)
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    df1 = df1.rename(columns={0: '0', 1: '1'}, inplace=False)
    df2 = df2.rename(columns={0: '0'}, inplace=False)
    dataset = pd.concat([df1, df2], axis=1, join="inner")
    #dataset = sns.load_dataset('iris')
    dataset.head()
    x_train, x_test, y_train, y_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=0.2)
    x_train.head()
    y_train.head()
    classifier = DecisionTreeClassifier(random_state=0)
    classifier = classifier.fit(x_train, y_train)
    tree.plot_tree(classifier)
    y_pred = classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def task5():
    url = 'https://raw.githubusercontent.com/likarajo/petrol_consumption/master/data/petrol_consumption.csv'
    dataset = pd.read_csv(url)
    dataset.head()
    dataset.describe()
    print(dataset)
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1:2].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = DecisionTreeRegressor()
    regressor.fit(x_train,y_train)
    print(tree.plot_tree(regressor))
    y_pred = regressor.predict(x_test)
    y_test_1 = []
    for i in range(len(y_test)):
        y_test_1.append(y_test[i][0])
    df = pd.DataFrame({'Actual': y_test_1, 'Pred': y_pred})
    print('Squared Error', metrics.mean_squared_error(y_test, y_pred))
    print('Absolute Error', metrics.mean_absolute_error(y_test, y_pred))


print('Enter num task 1 or 2 or 3 or 4 or 5')
inp = int(input())
while inp != 0:
    if inp == 1:
        task1()

        break
    elif inp == 2:
        task2()

        break
    elif inp == 3:
        task3()

        break
    elif inp == 4:
        task4()
        break
    elif inp == 5:
        task5()
        break
    else:
        print("Check input")
        break

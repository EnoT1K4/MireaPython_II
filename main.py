import math
import random
from math import sqrt
from statistics import median
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from numpy import trapz


def task13():
    x = 5 >= 2
    A = {1, 3, 7, 8}
    B = {2, 4, 5, 20, 'apple'}
    C = A & B
    df = 'Antonova', 34, 'women'
    z = 'type'
    D = [1, 'title', 2, 'content']
    print(x , '->' , type(x) , '\n' ,
          A , '->' , type(A) , '\n' ,
          B , '->' , type(B) , '\n' ,
          C , '->' , type(C) , '\n'
          , df , '->' , type(df) ,'\n' ,
          z , '->' , type(z) , '\n' ,
          D , '->' , type(D) , '\n')


def task23():
    print('Enter x')
    x = int(input())
    if x < -5:
        print('(-inf;-5)')
    elif -5 <= x <= 5:
        print('[-5;5]')
    else:
        print('(-5;inf)')

def task33():
    print('input task name: (1,2,3,4,5)')
    num = int(input())
    if num == 1:
        x = 10
        while x >=1:
            print(x)
            x-=3
    if num ==2:
        print(['smth', 'smth', 'smth'])
    if num ==3:
        a = []
        for i in range(2,16,2):
            a.append(i)
        print(a)
    if num ==4:
        a = []
        for i in range(105,4,-25):
            print(i)
    if num ==5:
        x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        x[0::2] = reversed(x[0::2])
        print(x)


def task43():
    print('Enter num')
    num = int(input())
    if num == 1:
        items = [random.random() for i in range(10)]
        sred = sum(items)/len(items)
        medi = median(items)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(items,items)
        plt.show()
        print(items)
    if num == 2:
        fun = []
        for x in range(1,11):
            fun.append((sqrt(1 + math.e ** sqrt(x) + math.cos(x**2)))/abs(1- (math.sin(x))**3))
        print(fun)
        newfun = fun[:5]
        plt.plot(fun)
        plt.show()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(newfun, newfun)
        plt.show()
    if num ==3:
        x = np.arange(0,10)
        y = np.abs(np.cos(x * np.exp(np.cos(x) + np.log(x+1))))
        plt.grid()
        plt.plot(x,y,c = 'purple')
        plt.fill_between(x,y)
        plt.show()
        area = trapz(y)
        print(area)
    if num ==4:
        Apple = [random.randint(140,150) for i in range(12)]
        Google = [random.randint(140,160) for i in range(12)]
        Microsoft = [random.randint(120,150) for i in range(12)]
        plt.plot(Apple)
        plt.plot(Google)
        plt.plot(Microsoft)
        plt.show()
    if num == 5:
        print('Input x')
        x = int(input())
        print('Input y')
        y = int(input())

        print(x+y)
        print(x-y)
        print(x*y)
        print(x/y)
        print(math.e**(x+y))
        print(math.sin(x+y))
        print(math.cos(x+y))
        print(x**y)



print('Enter num task 13 or 23 or 33 or 43')
inp = int(input())
while inp != 0:
    if inp == 13:
        task13()
        break
    elif inp == 23:
        task23()
        break
    elif inp == 33:
        task33()
        break
    elif inp == 43:
        task43()
        break


    else:
        print("Check input")
        break

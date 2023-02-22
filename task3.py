import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer


def euclid(v1, v2):
    return sum((x - y) ** 2 for x, y in zip(v1, v2))


def task1():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    s = tuple([1, 2, 3])
    z = tuple([2, 3, 1])
    ax.scatter(0, 0, 0)
    ax.scatter(3, 3, 3)
    ax.scatter(*s)
    ax.scatter(*z)
    print(euclid(np.array(s), np.array(z)))
    plt.show()


def task2():
    iris = sns.load_dataset('iris')
    iris
    plt.figure(figsize=(16, 7))
    plt.subplot(121)
    sns.scatterplot(data=iris, x='petal_width', y='petal_length', hue='species', s=70)
    plt.xlabel('Length')
    plt.ylabel('width')
    plt.legend()
    plt.grid()
    plt.subplot(122)
    sns.scatterplot(data=iris, x='sepal_width', y='sepal_length', hue='species', s=70)
    plt.xlabel('Length')
    plt.ylabel('width')
    plt.legend()
    plt.grid()
    X_train, X_test, y_train, y_test = train_test_split(iris.iloc[:, :-1],
                                                        iris.iloc[:, -1],
                                                        test_size=0.15)  # процент тут
    X_train.shape, X_test.shape, y_train.shape, y_test.shape
    X_train.head()
    y_train.head()
    model = KNeighborsClassifier(n_neighbors=10)  # соседи тут
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=iris, x='petal_width', y='petal_length', hue='species', s=70)
    plt.xlabel('Length')
    plt.ylabel('width')
    plt.legend(loc=2)
    plt.grid()
    for i in range(len(y_test)):
        if np.array(y_test)[i] != y_pred[i]:
            plt.scatter(X_test.iloc[i, 3], X_test.iloc[i, 2], color='red', s=150)

    print(f'accurasy:{accuracy_score(y_test, y_pred):.3}')


def task3():
    data_dict = [{"blue": 1, "red": 2},
                 {"green": 3, "brown": 4},
                 {"blue": 1, "red": 4},
                 {"blue": 2, "red": 2},
                 ]
    dictvect = DictVectorizer(sparse=False)
    feat = dictvect.fit_transform(data_dict)
    print(feat)


print('Enter num task 1 or 2 or 3')
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
    else:
        print("Check input")
        break

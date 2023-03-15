import networkx as nx
import math


def task1():
    def qZ(x, y):
        return (3 - 3 * y + 1) / (3 * x * x + y * y + 1)

    def qSumZ(Z):
        return sum(Z)

    def exchange(oldX, oldY, sortedId):
        X = [0 for i in range(4)]
        Y = [0 for i in range(4)]

        X[2] = oldX[sortedId[2]]
        X[3] = oldX[sortedId[2]]
        X[0] = oldX[sortedId[0]]
        X[1] = oldX[sortedId[1]]
        Y[0] = oldX[sortedId[2]]
        Y[1] = oldX[sortedId[2]]
        Y[2] = oldX[sortedId[0]]
        Y[3] = oldX[sortedId[1]]
        return X, Y

    def sorting(Z):
        sortedId = sorted(range(len(Z)), key=lambda k: Z[k])
        return sortedId

    def evoStep(X, Y, Z):
        _, minId = min((value, id) for (id, value) in enumerate(Z))
        X = X[:]
        Y = Y[:]
        Z = Z[:]
        X.pop(minId)
        Y.pop(minId)
        Z.pop(minId)

        return X, Y, Z

    def evo(X, Y, steps=4):
        result = []
        for i in range(4):
            arrZ = [qZ(x, Y[i]) for i, x in enumerate(X)]
            X, Y, Z = evoStep(X, Y, arrZ)
            X, Y = exchange(X, Y, sorting(Z))
            result.append([X, Y, qSumZ(arrZ), arrZ])
        return X, Y, result

    X = [-2, -1, 0, 2]
    Y = [-2, 1, 0, -1]
    results = evo(X, Y)
    for i in range(len(results[2])):
        print(f'max_{i + 1}_step:  {results[2][i][2]}')
    qualityArrZ = []
    for i in range(len(results[2])):
        qualityArrZ += results[2][i][3]
    print(max(qualityArrZ))


def task2():
    distances = [(1, 2, 18), (1, 3, 41), (1, 4, 36), (1, 5, 29), (1, 6, 19),
                 (2, 3, 27), (2, 4, 31), (2, 5, 37), (2, 6, 15),
                 (3, 4, 19), (3, 5, 42), (3, 6, 23),
                 (4, 5, 24), (4, 6, 17),
                 (5, 6, 24)]
    V = [1, 3, 4, 5, 6, 2, 1]
    Z = [(4, 5), (5, 6), (4, 6), (5, 6)]
    P = [63, 49, 45, 53]
    T = 100

    def probability(delta, T):
        return T * math.e ** (-delta / T)

    def reductTemp(prevT):
        nexT = 0.5 * prevT
        return nexT

    graph = nx.Graph()
    graph.add_weighted_edges_from(distances)

    def edgeLength(i, j, distances, roundTip=True):
        if roundTip:
            return max([(item[2] if (item[0] == i and item[1] == j) or (item[1] == i and item[0] == j) else -1)
                        for item in distances])
        else:
            return max([(item[2] if (item[0] == i and item[1] == j) else -1)
                        for item in distances])

    def routeLen(V, distances):
        edges = []
        for i in range(len(V) - 1):
            edges.append(edgeLength(V[i], V[i + 1], distances))
        return sum(edges)

    def routeOneRep(arrV, Z, replacement=True):
        decrement = 1 if replacement else 0
        arrV[Z[0] - decrement], arrV[Z[1] - decrement] = arrV[Z[1] - decrement], arrV[Z[0] - decrement]
        return arrV

    def chooseRoute(distances, V, Z, T, P):
        sumLength = routeLen(V, distances)
        arrSum = [sumLength]

        for i in range(len(Z)):
            newV = routeOneRep(V[:], Z[i])
            newS = routeLen(newV, distances)
            arrSum.append(newS)
            deltaS = newS - sumLength

            if deltaS > 0:
                p = probability(deltaS, T)
                if p > P[i]:
                    V = newV
                    sumLength = newS
            else:
                V = newV
                sumLength = newS
            T = reductTemp(T)
        return V, arrSum

    def drawRouteGraph(distances, bestRoute):
        newDist = []
        for i in range(len(bestRoute) - 1):
            for distance in distances:
                if distance[0] == bestRoute[i] and distance[1] == bestRoute[i + 1] \
                        or distance[1] == bestRoute[i] and distance[0] == bestRoute[i + 1]:
                    newDist.append(distance)
        graph = nx.Graph()
        graph.add_weighted_edges_from(newDist)
        nx.draw_kamada_kawai(graph, node_color='#fb7258', node_size=2000, with_labels=True)

    bestRoute, arrLength = chooseRoute(distances, V, Z, T, P)
    print(f'Best = {bestRoute}')
    print(f'Best len = {routeLen(bestRoute, distances)}')
    drawRouteGraph(distances, bestRoute)


print('Enter num task 1 or 2')
inp = int(input())
while inp != 0:
    if inp == 1:
        task1()
        break
    elif inp == 2:
        task2()
        break
    else:
        print("Check input")
        break

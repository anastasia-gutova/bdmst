import numpy as np
import codecs
import matplotlib.pyplot as plt
import argparse
import random
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric
from pyclustering.cluster.center_initializer import random_center_initializer

def BFS(G, start):
    
    visited = [False] * len(G)
    q = [start]
    distances = [0] * len(G)
    visited[start] = True
    while q:
        vis = q[0]
        q.pop(0)
        for i in range(len(G)):
            if (G[vis][i] != 0 and (not visited[i])):
                distances[i] = distances[vis] + 1
                q.append(i)
                visited[i] = True
    return distances

def diameter(G):
    v = u = w = 0
    d = BFS(G, v)
    for i in range(len(G)):
        if (d[i] > d[u]):
            u = i
    d = BFS(G, u)
    for i in range(len(G)):
        if (d[i] > d[w]):
            w = i
    return d[w]

def Prim(G):
    INF = 9999999
    N = len(G)
    selected_node = np.zeros(N)
    no_edge = 0
    selected_node[0] = True
    res = np.zeros((N, N))
    while (no_edge < N - 1):
        
        minimum = INF
        a = 0
        b = 0
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):  
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n
        res[a][b] = G[a][b]
        res[b][a] = G[a][b]
        selected_node[b] = True
        no_edge += 1
    return res

def Prim2(G):
    N = len(G)
    selected_node = np.zeros(N)
    selected_node[0] = True
    res = np.zeros((N, N))
    random_columns = random.sample(range(0, N), 2)

    for col in random_columns:
        for i in range(len(res)):
            res[col][i] = G[col][i]
            res[i][col] = G[i][col]
    
    return res

def onlyMinimumValue(a):
    copy = np.copy(a)
    minIndex = 9999999999
    for i in range(len(copy)):
        if (int(copy[i]) < minIndex and copy[i] != 0 ):
            minIndex = i
    for i in range(len(copy)):
        if (i != minIndex):
            copy[i] = 0
    return copy

def getMatrixByVerticesIndex(G, vertices):
    res = np.ndarray((len(vertices), len(vertices)))
    for ind, v1 in enumerate(vertices):
        for jnd, v2 in enumerate(vertices):
            res[ind][jnd] = G[v1][v2]
    return res

def updateMatrixByVerticesIndex(G, newG, vertices):
    res = np.copy(G)
    for ind, v1 in enumerate(vertices):
        for jnd, v2 in enumerate(vertices):
            res[v1][v2] = newG[ind][jnd]
    return res

def getWeight(G):
    sum = 0
    for i in range(len(G)):
        for j in range(len(G)):
            if (i < j):
                sum += G[i][j]
    return sum

def getEdgesCount(G):
    sum = 0
    for i in range(len(G)):
        for j in range(len(G)):
            if (i < j and G[i][j] != 0):
                sum += 1
    return sum

def distance(v1, v2):
    return abs(int(v1[0]) - int(v2[0])) + abs(int(v1[1]) - int(v2[1]))

def onlyShortestEdge(G, target, vertexIndex):
    newG = np.copy(G)
    minDistance = newG[target][vertexIndex]
    newG[target, :] = 0
    newG[:, target] =  0
    newG[target][vertexIndex] = minDistance
    newG[vertexIndex][target] = minDistance    
    return newG

def connectNearestVertex2(G, target, vertices):
    vertexIndex = -1
    minDist = 9999999
    for v1 in vertices:
        if(G[target][v1] != 0 and G[target][v1] < minDist):
            minDist = G[target][v1]
            vertexIndex = v1
    G = onlyShortestEdge(G, target, vertexIndex)
    return G

def showGraph(G, data, w, d):
    plt.grid()
    for i in range(0, len(data)):
        plt.plot(data[i][0], data[i][1], 'go')
        plt.text(data[i][0], data[i][1], "%d (%d, %d)" % (i+1, data[i][0], data[i][1]))
    for indA, a in enumerate(G):
        for indB, b in enumerate(G[indA]):
            if(G[indA][indB] != 0):
                plt.plot([data[indA][0], data[indB][0]], [data[indA][1], data[indB][1]], '-')
    plt.title("Weight: %d, diameter: %d" % (w, d))
    plt.show()

def plotLineManhettan(x, y):
    plt.plot([x[0], x[0]], [y[0], y[1]], '-')
    plt.plot([x[0], x[1]], [y[1], y[1]], '-')

def isArrayNotHaveCombValue(history, comb):
    for history_comb in history:
        if ((np.array(comb) == np.array(history_comb)).all()):
            return False
    return True

def minNonZeroIndex(a, comb, index, history):
    min = 999999999
    minIndex = -1
    for ind, t in enumerate(a):
        if(t < min and t != 0):
            comb_copy = np.copy(comb)
            comb_copy[index] = ind
            if (isArrayNotHaveCombValue(history, comb_copy)):
                min = t
                minIndex = ind
    return minIndex

def remakeGraph(G, vertices, history):
    resMin = 9999999
    resMinV = None
    for index, v in enumerate(vertices):
        minIndex = minNonZeroIndex(G[v], vertices, index, history)
        if (minIndex not in vertices and minIndex != -1):
            resMin = minIndex
            resMinV = v
    vertices[vertices.index(resMinV)] = resMin
    return vertices
    
def distanceFloat(v1, v2):
    return abs(v1[0] - v2[0]) + abs(v1[1] - v2[1])

def getCombByClusterCenter(center, data):
    data_copy = np.copy(data)
    res = []
    for cp in list(center):
        d = []
        for ind, p in enumerate(data_copy):
            if (ind not in res):
                d.append(distanceFloat(cp, p))
            else: d.append(1000000000000)
        res.append(np.argmin(np.array(d)))
    return res


def outputResult(n, weight, diameter, matrix):
    writefile = codecs.open("2/Chernyshova%d.txt" % (n), 'w', 'utf-8')
    writefile.write("c Вес дерева = %d, диаметр = %d\n" % (weight, diameter))
    writefile.write("p edge %d %d\n" % (len(matrix), getEdgesCount(matrix)))
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (i < j and matrix[i][j] != 0):
                writefile.write("e %d %d\n" % (i + 1, j + 1))
    writefile.close()


parser = argparse.ArgumentParser(description='Diameter')
parser.add_argument("--d")
args = parser.parse_args()
files = ["Taxicab_64.txt"] # ["Taxicab_64.txt", "Taxicab_128.txt", "Taxicab_512.txt", "Taxicab_2048.txt", "Taxicab_4096.txt"]
for file in files:
    file1 = open(file, "r")
    raw_data = []
    t = 0
    X = []
    Y = []
    while True:
        line = file1.readline()
        if not line:
            break
        if t != 0:
            a = line.strip().split('\t')

            X.append(int(a[0]))
            Y.append(int(a[1]))
            raw_data.append([int(a[0]), int(a[1])])
        else: n = int(line.split('=')[1])
        t = t + 1
    X = np.array(X)
    Y = np.array(Y)
    data = np.array(raw_data)
    d = len(data)/32 + 2
    matrix = []
    for r1 in data:
        row = []
        for r2 in data:
            row.append(abs(int(r1[0]) - int(r2[0])) + abs(int(r1[1]) - int(r2[1])))
        matrix.append(row)

    minWeight = 99999999999999999999
    min_matrix = None
    newD = int(args.d)
    for newnewD in range(newD, newD + 20000):
        for z in range(0, newD-1):
            for k in range(0, newD-1):
                history = [np.zeros(newD - 1)]
                prevNice = False
                comb = None
                repeat = 1001
                for it in range(80):
                    repeat += 1
                    initial_centers = random_center_initializer(data, newD - 1 ).initialize()
                    instanceKm = kmeans(data, initial_centers=initial_centers, metric=distance_metric(2))
                    instanceKm.process()
                    pyCenters = instanceKm.get_centers()

                    if(len(pyCenters) != (newD - 1)):
                        repeat -=1
                        continue

                    if (repeat > 20):
                        comb = getCombByClusterCenter(pyCenters, data)
                        comb.sort()
                        if (not isArrayNotHaveCombValue(history, comb) ):
                            repeat -=1
                            continue
                    
                    history.append(np.array(comb))
                    matrix_copy = np.copy(matrix)
                    matrix_part = getMatrixByVerticesIndex(matrix_copy, comb)
                    prim_res = Prim2(matrix_part)
                    prim_res = Prim(prim_res)
                    
                    diameter_part = diameter(prim_res)
                    if (diameter_part <= d-2):
                        matrix_copy = updateMatrixByVerticesIndex(matrix_copy, prim_res, comb)
                        a = []
                        for t in range(len(matrix_copy)):
                            if(t not in comb):
                                a.append(t)
                        for i in a:
                            matrix_copy = connectNearestVertex2(matrix_copy, i, comb)
                        weight = getWeight(matrix_copy)
                        if(weight <= minWeight + 10):
                            diam = diameter(matrix_copy)
                            if (diam <= d):
                                if(weight <= minWeight):
                                    minWeight = weight
                                min_matrix = matrix_copy
                                outputResult(n, weight, diam, matrix_copy)
                                repeat = 0
                            else:
                                print('Overthrow')
                        comb = remakeGraph(matrix_copy, comb, history)
                        history.append(np.array(comb))

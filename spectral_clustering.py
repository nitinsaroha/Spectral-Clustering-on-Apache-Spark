from __future__ import print_function
from pyspark import SparkContext
import sys
import numpy as np
import math
from pyspark.mllib.clustering import PowerIterationClustering, PowerIterationClusteringModel
import matplotlib.pyplot as plt
# number of clusters
num_clusters = int(sys.argv[1])
# upper bound for number of iterations
upper_bound = int(sys.argv[2])
# floating point value for gamma
gamma = float(sys.argv[3])
# to get the input file name
input_file = str(sys.argv[4])
# small method to print the contents of RDD
def print_rdd(RDD):
    for line in RDD.collect():
        print (line)

def split_function(d):
    temp = d.encode('utf8').split('\t')
    return [float(temp[0]), float(temp[1])]

def affinities(d):
    point1 = d[0]
    point2 = d[1]
    src_id = point1[0]
    dst_id = point2[0]
    sum_squares = ( (point1[1][0] - point2[1][0]) * (point1[1][0] - point2[1][0]) + 
                (point1[1][1] - point2[1][1]) * (point1[1][1] - point2[1][1]) )
    Aij = math.exp(-gamma * sum_squares)
    return src_id, dst_id, Aij

def two_clusters(joined_RDD):
    global x1, x2, y1, y2
    x1 = []
    x2 = []
    y1 = []
    y2 = []

    for x in joined_RDD.collect():
        if(x[1][1] == 0):
            x1.append(float(x[1][0][0]))
            y1.append(float(x[1][0][1]))
        else:
            x2.append(float(x[1][0][0]))
            y2.append(float(x[1][0][1]))

def three_clusters(joined_RDD):
    global x1, x2, x3, y1, y2, y3
    x1 = []
    x2 = []
    x3 = []
    y1 = []
    y2 = []
    y3 = []

    for x in joined_RDD.collect():
        if(x[1][1] == 0):
            x1.append(float(x[1][0][0]))
            y1.append(float(x[1][0][1]))
        elif(x[1][1] == 1):
            x2.append(float(x[1][0][0]))
            y2.append(float(x[1][0][1]))
        else:
            x3.append(float(x[1][0][0]))
            y3.append(float(x[1][0][1]))

if __name__ == "__main__":

    # made the spark contest
    sc = SparkContext(appName="Spectral Clustering in Spark")
    # input file
    input_file_RDD = sc.textFile(input_file)

    withIndex = input_file_RDD.map(split_function).zipWithIndex()
    indexKey = withIndex.map(lambda (k,v): (v,k))

    C = indexKey.cartesian(indexKey)
    
    input_affinities = C.map(affinities)
    
    model = PowerIterationClustering.train(input_affinities, num_clusters,  upper_bound)

    joined = sc.parallelize(sorted(indexKey.join(model.assignments()).collect()))

    if (num_clusters == 2):
        two_clusters(joined)
        plt.scatter(x1, y1, c='r')
        plt.scatter(x2, y2, c='g')
        plt.show()
    elif (num_clusters == 3):
        three_clusters(joined)
        plt.scatter(x1, y1, c='r')
        plt.scatter(x2, y2, c='g')
        plt.scatter(x3, y3, c='b')
        plt.show()

    # Save and load model
    # model.save(sc, "myModelPath")
    # sameModel = PowerIterationClusteringModel.load(sc, "myModelPath")

    # input_affinities.coalesce(1, True).saveAsTextFile("output")
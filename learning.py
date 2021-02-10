'''
hw1.py
Author: TODO

Tufts COMP 135 Intro ML

'''
from array import *
import numpy as np
from numpy import random
import math

def letsgo(x_all_LF, frac_test=0.5, random_state=None):
    xcopy = x_all_LF.copy()
    if random_state is None:
        random_state = np.random
    else:
        random_state = np.random.seed(random_state)
    np.random.shuffle(xcopy)
    train_size = int(frac_test * len(xcopy))
    train_arr = xcopy[:train_size]
    test_arr = xcopy[train_size:]
    return test_arr, train_arr


def testing1():
    print("----------------------------------------")
    print("Problem 1 testing/learning")
    T = np.random.rand(4,2)
    print("BEFORE OPERATIONS")
    lick_mine(T)
    test_arr, train_arr = letsgo(T, frac_test = 0.5, random_state = None)
    print("TESTING ARRAY")
    lick_mine(test_arr)
    print("TRAINING ARRAY")
    lick_mine(train_arr)
    print("AFTER OPERATIONS")
    lick_mine(T)
    print("----------------------------------------")
    return None

def lick_mine(T):
    print("----------------------------------------")
    print(T.shape)
    for r in T:
        for c in r:
            print(c,end = " ")
        print()
    print("----------------------------------------")
    return None



def testing2():
    print("----------------------------------------")
    print("Problem 2 testing/learning")
    T = np.random.rand(2,4)
    F = np.random.rand(2,4)
    lick_mine(T)
    lick_mine(F)
    result = calc_k_nearest_neighbors(T, F)
    print("SHAPE OF RESULT = ")
    print(result.shape)
     
    
    print("----------------------------------------")
    return None



def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Any ties in distance may be broken arbitrarily.

    Args
    ----
    data_NF : 2D array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, positive (must be >= 1)
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of the k-th neighbor of the q-th query
    '''
    print("----------------------------------------")
    print("Entered trampoline function")
    neighbors = []
    for x2 in query_QF:
        d = {}
        hold = []
        for x1 in data_NF:
            dist = euclidean_distance(x1,x2)
            if dist in d:
                d[dist].append(x1)
            else:
                d[dist] = [x1]
        for i in range(K):
            cur = min(d.keys())
            hold.append(d[cur].pop())
            if len(d[cur]) == 0:
                d.pop(cur)
        neighbors.append(hold)
    print("Finished?")
    print("----------------------------------------")
    return np.array(neighbors)
    

def euclidean_distance(elem_1, elem_2):
    print("----------------------------------------")
    print("Entered calc euclidean distance")
    distance = 0
    for i in range(len(elem_1) - 1):
        distance += (elem_1[i] - elem_2[i]) * (elem_1[i] - elem_2[i])
    return math.sqrt(distance)

'''
def learning():
    T = np.random.rand(4,4)
    lick_mine(T)
    x = random.rand()
    test_arr, train_arr = splitnprint(T, x)
    lick_mine(test_arr)
    lick_mine(train_arr)
    return None



def splitnprint(T, frac):
    
    print("Poop in my face")
    print(frac)
    xcopy = T.copy()
    lick_mine(xcopy)
    np.random.shuffle(xcopy)
    print("testing shuffle")
    lick_mine(xcopy)
    print("shuffle testing done")
    train_size = int(frac * len(xcopy))
    train_arr = xcopy[:train_size]
    test_arr = xcopy[train_size:]
    print(train_arr.shape)
    print(test_arr.shape)
    lick_mine(train_arr)
    print("FUCK MATIAS")
    lick_mine(T)
    
    return test_arr, train_arr

learning()
'''

testing1()
testing2()
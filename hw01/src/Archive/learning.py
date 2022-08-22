'''
hw1.py
Author: TODO

Tufts COMP 135 Intro ML

'''
import numpy as np
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
    printer(T)
    test_arr, train_arr = split_into_train_and_test(T, frac_test = 0.5, random_state = None)
    print("TESTING ARRAY")
    printer(test_arr)
    print("TRAINING ARRAY")
    printer(train_arr)
    print("AFTER OPERATIONS")
    printer(T)
    print("----------------------------------------")
    return None

def printer(T):
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
    T = np.random.rand(4,100)
    F = np.random.rand(4,100)
    printer(T)
    printer(F)
    result = calc_k_nearest_neighbors(T, F)
    print("SHAPE OF RESULT = ")
    print(result.shape)
     
    
    print("----------------------------------------")
    return None

def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test set along first dimension

    User can provide a random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
    random_state : np.random.RandomState instance or integer or None
        If int, code will create RandomState instance with provided value as seed
        If None, defaults to the current numpy random number generator np.random

    Returns
    -------
    x_train_MF : 2D array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. The provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------
    >>> x_LF = np.eye(10)
    >>> xcopy_LF = x_LF.copy() # preserve what input was before the call
    >>> train_MF, test_NF = split_into_train_and_test(
    ...     x_LF, frac_test=0.3, random_state=np.random.RandomState(0))
    >>> train_MF.shape
    (7, 10)
    >>> test_NF.shape
    (3, 10)
    >>> print(train_MF)
    [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    >>> print(test_NF)
    [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]

    ## Verify that input array did not change due to function call
    >>> np.allclose(x_LF, xcopy_LF)
    True

    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''
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
    neighb_QKF  = np.array(neighbors)
    return neighb_QKF


def euclidean_distance(arr_1, arr_2):
    print("----------------------------------------")
    print("Entered calc euclidean distance")
    distance = 0
    for i in range(len(arr_1) - 1):
        distance += (arr_1[i] - arr_2[i]) * (arr_1[i] - arr_2[i])
    return math.sqrt(distance)


'''
def learning():
    T = np.random.rand(4,4)
    printer(T)
    x = random.rand()
    test_arr, train_arr = splitnprint(T, x)
    printer(test_arr)
    printer(train_arr)
    return None



def splitnprint(T, frac):
    
    print(frac)
    xcopy = T.copy()
    printer(xcopy)
    np.random.shuffle(xcopy)
    print("testing shuffle")
    printer(xcopy)
    print("shuffle testing done")
    train_size = int(frac * len(xcopy))
    train_arr = xcopy[:train_size]
    test_arr = xcopy[train_size:]
    print(train_arr.shape)
    print(test_arr.shape)
    printer(train_arr)
    printer(T)
    
    return test_arr, train_arr

learning()
'''

testing1()
testing2()
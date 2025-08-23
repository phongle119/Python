import numpy as np
from code_for_hw02 import *

def perceptron(data, labels, params = {}, hook = None):
    print("")
    print("Perceptron algorithm started")
    d, n = data.shape
    theta = np.zeros((d, 1))
    theta0 = np.zeros((1, 1))
    T = params.get('T', 1000)
    for j in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if (np.sign(np.dot(theta.T, x) +theta0)* y)> 0:
                continue
            else:
                theta = theta + y * x
                theta0 += y
    print("d = {}, n = {}, theta shape = {}, theta_0 shape = {}".format(d,n,theta.shape,theta0.shape))
    
    return theta, theta0

def averaged_perceptron(data, labels, params = {}, hook = None):
    print("")
    print("Average Perceptron algorithm started")
    d, n = data.shape
    theta = np.zeros((d, 1))
    theta0 = np.zeros((1, 1))
    theta_average = np.zeros((d, 1))
    theta0_average = np.zeros((1, 1))
    T = params.get('T', 1000)
    for j in range(T):
        for i in range(n):
            x = data[:,i:i+1]
            y = labels[:,i:i+1]
            if (np.sign(np.dot(theta.T, x) +theta0)* y)<= 0:
                theta = theta + y * x
                theta0 += y
            theta_average += theta
            theta0_average += theta0
    print("d = {}, n = {}, theta shape = {}, theta_0 shape = {}".format(d,n,theta.shape,theta0.shape))
    return theta_average / (T * n), theta0_average / (T * n)

def scorePhong(data, labels, theta, theta0):
    dot= np.sign(np.dot(theta.T,data)+theta0)==labels
    return np.sum(dot)

def eval_classifier(learner, data, labels, data_test, labels_test):
    theta, theta0 = learner(data, labels)
    if data_test is None or labels_test is None:
        score1 = score(data, labels, theta, theta0)
        total = data.shape[1]
        print(f"Score of {learner.__name__} on training set: {score1} out of {total}")
        return 
    else:
        score1 = score(data_test, labels_test, theta, theta0)
        total = data_test.shape[1]
        print(f"Score of {learner.__name__} on training set: {score1/total} out of {total}")
    return score1/total

def generate_data(shape):
    d, n = shape
    data = np.random.randn(d, n)
    theta = np.random.randn(d, 1)
    theta0 = np.random.randn(1, 1)
    labels = np.sign(np.dot(theta.T, data) + theta0)
    return data, labels

def eval_learning_alg(learner, data_gen, n_train, n_test, it):
    data, labels = data_gen(n_train);
    return eval_classifier(learner, data, labels, None, None)

data, labels = xor_more()
score2 = eval_learning_alg(averaged_perceptron, generate_data, (2,10), None, None)
print("Averaged perceptron on random data: ", score2)


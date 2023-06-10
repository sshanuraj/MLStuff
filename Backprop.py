import numpy as np
from matplotlib import pyplot as plt
import math
np.random.seed(0)

#create the real data points
def create_points(num_points):
    radius = 5
    data = []
    labels = []
    # Generate positive examples (labeled 1).
    for i in range(num_points // 2):
        # the radius of positive examples
        r = np.random.uniform(0, radius*0.5)
        angle = np.random.uniform(0, 2*math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        data.append([x, y])
        labels.append(1)
        
    # Generate negative examples (labeled 0).
    for i in range(num_points // 2):
        # the radius of negative examples
        r = np.random.uniform(radius*0.7, radius)
        angle = np.random.uniform(0, 2*math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        data.append([x, y])
        labels.append(0)
    print(data[0]) 
    data = np.asarray(data).T
    labels = np.asarray([labels])
    return data, labels

def sigmoid(x):
    return 1/(1 + np.e**(-x))

def get_params(layers):
    n = len(layers)
    params = {}
    vel = {}
    for i in range(n-1):
        params["W"+str(i+1)] = np.random.randn(layers[i+1], layers[i])
        params["b"+str(i+1)] = np.zeros((layers[i+1], 1))
        #vel["W"+str(i+1)] = np.zeros((layers[i+1], layers[i]))
        #vel["b"+str(i+1)] = np.zeros((layers[i+1], 1))
    params["W1"] = np.array([[ 1.76405235,  0.40015721],
       [ 0.97873798,  2.2408932 ],
       [ 1.86755799, -0.97727788]])
    params["W2"] = np.array([[ 0.95008842, -0.15135721, -0.10321885]])
    return params, vel

def normalize(X):
    return X/np.max(X)

def diff_sigmoid(x):
    return x*(1-x)

def get_diff(op, Y, params):
    diff = {}
    L = len(params)//2
    # C = 0.5*(y_hat - y)^2
    Y_hat = op["A"+str(L)]

    #output layer backprop
    dZ = np.multiply(Y_hat-Y, diff_sigmoid(Y_hat))
    diff["W"+str(L)] = np.dot(dZ, op["A"+str(L-1)].T)
    diff["b"+str(L)] = dZ

    #print(diff["W2"])
    #print(diff["b2"])

    #backprop for hidden layers
    for i in range(L-1, 0, -1):
        Ai = op["A"+str(i)]
        Ai_1 = op["A"+str(i-1)]
        dZ = np.multiply(np.dot(params["W"+str(i+1)].T, dZ), diff_sigmoid(Ai))
        diff["W"+str(i)] = np.dot(dZ, Ai_1.T)
        diff["b"+str(i)] = dZ
        #print(diff["W"+str(i)])
        #print(diff["b"+str(i)])

    return diff

def output(X, params):
    L = len(params)//2
    A = X
    op = {}
    op["A0"] = X
    for i in range(1, L+1):
        Z = np.dot(params["W"+str(i)], A) + params["b"+str(i)]
        A = sigmoid(Z)
        op["A"+str(i)] = A
    #print(op)
    return A, op

def get_diff_init(layers):
    diff={}
    L = len(layers)
    for i in range(L-1):
        diff["W"+str(i+1)] = np.zeros((layers[i+1], layers[i]))
        diff["b"+str(i+1)] = np.zeros((layers[i+1], 1))
   
    return diff

def update_params(params, diff, learning_rate):
    L = len(params)//2
    for i in range(1, L+1):
        params["W"+str(i)] -= learning_rate*(diff["W"+str(i)])
        params["b"+str(i)] -= learning_rate*(diff["b"+str(i)])
    return params

def update_params_momentum(params, diff, learning_rate, alpha, vel):
    L = len(params)//2
    for i in range(1, L+1):
        vel["W"+str(i)] = alpha*vel["W"+str(i)] - learning_rate*diff["W"+str(i)]
        vel["b"+str(i)] = alpha*vel["b"+str(i)] - learning_rate*diff["b"+str(i)] 
        params["W"+str(i)] += vel["W"+str(i)]
        params["b"+str(i)] += vel["b"+str(i)]
    return params, vel

def train(X, Y, layers, params, learning_rate, n_steps):
    for i in range(n_steps):
        num_points = X.shape[1]
        cost = 0
        diff = get_diff_init(layers)

        for j in range(num_points):
            Xi = X[:,j:j+1]
            Yi = Y[:,j:j+1]
            Y_hat, op = output(Xi, params)
            mcost = loss_fn(Y_hat, Yi)
            cost += mcost
            mdiff = get_diff(op, Yi, params)
            for key in diff:
                diff[key] += mdiff[key]
            #print(op)
            #print(cost)
            #print(params)
            #break
        for key in diff:
            diff[key] = diff[key]/num_points
        cost = cost/num_points
        #print(cost)
        params = update_params(params, diff, learning_rate)
        if i % 25 == 0:
            print("Cost at step {:3d}: {:3.2f}".format(i, cost))
    return params

def loss_fn(y_hat, y):
    #print(y_hat, y)
    loss_array = ((y_hat-y)**2)/2
    return sum(loss_array[0])

X, Y = create_points(200)
layers = [2, 3, 1]
params, velocity = get_params(layers)
params = train(X, Y, layers, params, 5, 500)
#print(params)

import numpy as np 
import math 

def get_points():
    X=[]
    Y=[]

    x = np.array([-0.5, 0.25, -0.8, -1])
    X.append(x)
    Y.append(1)

    x = np.array([-1, -0.1, -0.1, -1])
    X.append(x)
    Y.append(1)

    x = np.array([0.5, 0, 0.25, 0.1])
    X.append(x)
    Y.append(-1)

    x = np.array([-0.2, -0.3, 0.2, 0])
    X.append(x)
    Y.append(-1)

    x = np.array([-0.8, 0, -0.8, -1])
    X.append(x)
    Y.append(1)

    x = np.array([-0.15, -0.5, 0.05, -0.25])
    X.append(x)
    Y.append(-1)

    x = np.array([-1, 0, -1, -1])
    X.append(x)
    Y.append(1)

    x = np.array([0, -0.25, 0.25, 0.1])
    X.append(x)
    Y.append(-1)

    Y = np.array([Y])
    X = np.array(X).T

    return X, Y

def get_cost(w, x, y):
    indicator = np.multiply(y, np.dot(w.T, x)) #y.wTx
    indicator = indicator[0][0]

    if indicator >= 1:
        return 0
    return 0.5*(1 - indicator)**2

def get_norm2(x):
    s = 0
    for i in x:
        s = s + i**2
    return s**0.5

def normalize(x):
    X = []
    for i in x.T:
        norm = get_norm2(i)
        i = i/norm
        X.append(i)
    X = np.array(X)
    X = X.T
    
    return X

def get_params():
    return np.array([[0, 0, 0, 0]]).T

def get_output(x, w):
    op = np.dot(w.T, x)
    op = op[0][0]

    if(op>0):
        return 1
    elif op == 0:
        return 0
    return -1

def update_params(w, diffw, learning_rate):
    w = w - learning_rate*diffw
    return w

def update_params_momentum(v, w, diffw, learning_rate, alpha):
    v = alpha*v - learning_rate*diffw
    w = w + v
    return w, v

def get_diff(w, x, y):
    op = np.dot(w.T, x)
    coeff = op - y
    
    coeff = coeff[0][0]
    diffw = coeff*x
    
    return diffw

def train(w, X, Y, learning_rate, n_steps):
    n = X.shape[1]
    cost = 0
    for i in range(n_steps):
        mcost = 0
        diffw = np.array([[0,0,0,0]]).T
        for j in range(n):
            xi = X[:,j:j+1]
            yi = Y[:,j:j+1]

            mcost = get_cost(w, xi, yi)
            if mcost == 0:
                continue
            mdiffw = get_diff(w, xi, yi)
            diffw = diffw + mdiffw
            cost = cost + mcost
        cost = cost/n
        diffw = diffw/n
        w = update_params(w, diffw, learning_rate)
        if i%5 == 0:
            print("Cost at step %d: %f"%(i, cost))
    return w

def train_sgd(w, X, Y, learning_rate, n_steps):
    n = X.shape[1]
    cost = 0
    for i in range(n_steps):
        cost = 0
        diffw = np.array([[0,0,0,0]]).T
        it = (i%n)
        for j in range(1):
            xi = X[:,it:it+1]
            yi = Y[:,it:it+1]

            mcost = get_cost(w, xi, yi)
            if mcost == 0:
                continue
            mdiffw = get_diff(w, xi, yi)
            diffw = mdiffw
            cost = mcost
        w = update_params(w, diffw, learning_rate)
        if i%10 == 0:
            print("Cost at step %d: %f"%(i, cost))
    return w

def train_momentum(w, X, Y, learning_rate, n_steps, alpha):
    n = X.shape[1]
    cost = 0
    v = np.array([[0,0,0,0]]).T
    for i in range(n_steps):
        mcost = 0
        diffw = np.array([[0,0,0,0]]).T
        for j in range(n):
            xi = X[:,j:j+1]
            yi = Y[:,j:j+1]

            mcost = get_cost(w, xi, yi)
            if mcost == 0:
                continue
            
            mdiffw = get_diff(w, xi, yi)
            diffw = diffw + mdiffw
            cost = cost + mcost
        cost = cost/n
        diffw = diffw/n
        w, v = update_params_momentum(v, w, diffw, learning_rate, alpha)
        if i%5 == 0:
            print("Cost at step %d: %f"%(i, cost))
    return w

def perceptron(w, X, Y):
    n = X.shape[1]
    flag = True
    for i in range(n):
        xi = X[:,i:i+1]
        yi = Y[:,i:i+1]
        op = np.dot(w.T, xi)
        op = np.multiply(yi, op)
        op = op[0][0]
        if op <= 0:
            w = w + yi[0][0]*xi
            flag = True
    return w

def update_params_adagrad(r, w, diffw, learning_rate, delta):
    r = r + diffw**2
    adagrad = learning_rate/(delta + r**0.5)
    w = w - np.multiply(adagrad, diffw)
    return w, r

def train_adagrad(w, X, Y, learning_rate, n_steps, delta):
    n = X.shape[1]
    cost = 0
    r = np.array([[0,0,0,0]]).T
    for i in range(n_steps):
        cost = 0
        diffw = np.array([[0,0,0,0]]).T
        it = (i%n)
        for j in range(1):
            xi = X[:,it:it+1]
            yi = Y[:,it:it+1]

            mcost = get_cost(w, xi, yi)
            if mcost == 0:
                continue
            mdiffw = get_diff(w, xi, yi)
            diffw = mdiffw
            cost = mcost
        w, r = update_params_adagrad(r, w, diffw, learning_rate, delta)
        if i%10 == 0:
            print("Cost at step %d: %f"%(i, cost))
    return w

X, Y = get_points()
print("Input X:")
print(X)

print("Output Y:")
print(Y)

w = get_params()
w = train(w, X, Y, 0.5, 25)
print("GD Output:\n")
print(w)

w = get_params()
w = train_sgd(w, X, Y, 0.1, 80)
print("SGD Output:\n")
print(w)

w = get_params()
w = train_momentum(w, X, Y, 0.5, 25, 0.5) 
print("Momentum output:\n")
print(w)

w = get_params()
w = perceptron(w, X, Y)
print("Perceptron output:\n")
print(w)

w = get_params()
w = train_adagrad(w, X, Y, 0.1, 80, 10**(-6))
print("Adagrad output:\n")
print(w)




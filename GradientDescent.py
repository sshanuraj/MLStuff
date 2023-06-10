import numpy as np
from matplotlib import pyplot as plt

def get_op(x):
    return 3 + 2*x[1] + 4*x[2] + np.random.normal(0, 0.01)

def ip_op_map(N, inpX): #N=number of data points
    x = []  #keep x dimensions as NxinpX
    y = []  #output dimension is nx1

    #for linear problem -> keep model as y = 3 + 2x1 + 4x2 + normal_dist_noise

    for i in range(N):
        xi = []
        for j in range(inpX):
            if j == 0:
                xi.append(1)
            else:
                xi.append(i+j)
        x.append(xi)
        yi = get_op(xi)
        y.append(yi)

    y = np.array([y])
    y = y.T #making the dimension from 1xn to nx1

    x = np.array(x) #dimensions are still nxinpX

    return x, y

def get_params(M):  #M=inpX
    w = np.random.randn(M, 1)
    return w

def get_individual_pred_op(x, w):
    op_mat = np.dot(x, w) #1x1 result
    return op_mat[0][0]

def get_pred_op(x, w):
    return np.dot(x, w)

def get_grads(x, y, w):
    N = x.shape[0]
    grad_matrix = np.dot(x.T, x)
    grad_matrix = np.dot(grad_matrix, w)
    grad_matrix = grad_matrix - np.dot(x.T, y)
    grad_matrix = (1/N)*grad_matrix

    return grad_matrix

def compute_loss(y_pred, y):
    N = y.shape[0]
    loss = (1/N)*(y_pred-y)**2
    return sum(loss)[0]

def update_params(grads, w, lr):
    w = w - lr*grads
    return w

def train(w, x, y, N_steps):
    y_pred = get_pred_op(x, w)
    print("Step %s, Loss: %s"%("0", str(compute_loss(y_pred, y))))
    for i in range(1, N_steps+1):
        y_pred = get_pred_op(x, w)
        loss = compute_loss(y_pred, y)
        if i%10 == 0:
            print("Step %s, Loss: %s"%(str(i), str(loss)))
            print("Weights: %s"%(str(w.T)))
        grads = get_grads(x, y, w)
        w = update_params(grads, w, 0.001)
    return w

x, y = ip_op_map(15, 3)
w = get_params(3)
w = train(w, x, y, 100)
y_pred = get_pred_op(x, w)
plt.plot(x[:,1], y, color = "blue")
plt.plot(x[:,1], y_pred, color = "green")
plt.show()

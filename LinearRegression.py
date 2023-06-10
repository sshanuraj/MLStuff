import numpy as np
from matplotlib import pyplot as plt

#create the real data points
def create_points():
    y = []
    x = []
    i = 0

    while i <= 2*np.pi:
        p = np.sin(i) + np.random.normal(0, 0.05)
        y.append(p)
        x.append(i)
        i = i + 0.01
    
    return x, y

def create_function(m_dim, x, y):
    x=np.array([x])  #x - > (1, n)
    x = x.T #x -> (n, 1)
    y = np.array([y])
    y = y.T

    phi = []
    n = x.shape[0]
    y_hat = [0] * n

    if m_dim <= 0:
        return y_hat

    for i in range(n):
        phi_i = []
        for j in range(m_dim + 1):
            phi_i.append(x[i][0]**j)
        phi.append(phi_i)
    phi = np.array(phi)
    pseudo_inv_phi = np.linalg.inv(np.dot(phi.T, phi))
    app_pip = np.dot(pseudo_inv_phi, phi.T)
    wstar = np.dot(app_pip, y)
    y_hat = np.dot(phi, wstar)

    print(y_hat.shape)
    return y_hat


#loss function
def loss_fn(y_hat, y):
    y_hat = np.array(y_hat)
    y = np.array(y)

    loss_array = ((y_hat-y)**2)/len(y)
    return sum(loss_array)

x, y = create_points()

"""
y_loss = []
x_loss = []
for i in range(1, 10):
    x_loss.append(i)
    y_hat = create_function(i, x, y)
    l = loss_fn(y_hat.T[0], y)
    print(l)
    y_loss.append(l)

plt.plot(x_loss, y_loss)
plt.show()
"""

y_hat = create_function(6, x, y)
plt.plot(x, y, color = "blue")  #real data points
plt.plot(x, y_hat, color = "red")   #predicted data curve
plt.show()


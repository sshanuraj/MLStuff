import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)

def generate_data(ndim, num_points):
    X = np.random.randn(ndim, num_points)
    for i in range(ndim):
        avg = 0
        for j in range(num_points):
            X[i][j] = np.random.randint(0,5)
            avg = avg + X[i][j]
        avg = avg/num_points
        X[i] = X[i]-avg  #centred around the mean
    return X

def get_eigenvalues(data_matrix):
    N = data_matrix.shape[1] - 1
    cov = (1/N)*np.dot(data_matrix, data_matrix.T)

    evalues, evectors = np.linalg.eig(cov)
    return evalues, evectors, cov

def get_proj_basis(evectors, proj_dim):
    proj_basis = evectors[:, 0:proj_dim]
    return proj_basis

def get_projected_points(data, proj_dims, proj_basis):
    proj_points = np.dot(proj_basis.T, data)
    return proj_points

ndim = 3
num_points = 10
data = generate_data(ndim, num_points)
np.set_printoptions(precision = 2)

print("\nData design matrix:")
print(data)

evalues, evectors, cov = get_eigenvalues(data)

print("\nCovariance matrix:")
print(cov)

print("\nEigenvalues:")
print(evalues)

print("\nEigenvectors:")
print(evectors)

proj_points = get_projected_points(data, 2, get_proj_basis(evectors, 2))
print("\nNew points on eigenvector basis:")
print(proj_points)

fig = plt.figure()
ax = plt.axes(projection = "3d")

ax.scatter3D(data[0:1][0], data[1:2][0], data[2:3][0])
ax.scatter3D(proj_points[0:1][0], proj_points[1:2][0])
plt.show()

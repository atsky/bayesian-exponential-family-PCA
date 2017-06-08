import numpy as np
import matplotlib.pyplot as plt

c=100
x = np.arange(1, c+1)
y = 2 * x + np.random.randn(c) * 2
X = np.vstack((x, y))
print(X)

Xcentered = (X[0] - x.mean(), X[1] - y.mean())
m = (x.mean(), y.mean())
print(Xcentered)
print("Mean vector: ", m)

covmat = np.cov(Xcentered)
print(covmat, "\n")
print("Variance of X: ", np.cov(Xcentered)[0, 0])
print("Variance of Y: ", np.cov(Xcentered)[1, 1])
print("Covariance X and Y: ", np.cov(Xcentered)[0, 1])

# снижение размерности
_, vecs = np.linalg.eig(covmat)
v = -vecs[:, 1]
Xnew = np.dot(v, Xcentered)
print(Xnew)

plt.figure(figsize=(8, 8))
plt.scatter(Xnew[0:c/2], Xnew[c/2:c+1])
plt.show()
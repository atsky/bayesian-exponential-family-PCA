import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

mean = [[], [], []]
Xc0 = [[], [], []]

m = (X[:, 0].mean(), X[:, 1].mean(), X[:, 2].mean(), X[:, 3].mean())
Xcentered0 = X[:, 0] - m[0]
Xcentered1 = X[:, 1] - m[1]
Xcentered2 = X[:, 2] - m[2]
Xcentered3 = X[:, 3] - m[3]

Xcentered = (X - X.mean(axis=0)).T
print(Xcentered)
print("Mean vector: ", m)


covmat = np.cov(Xcentered)
print(covmat)


# снижение размерности
_, vecs = np.linalg.eig(covmat)
print(vecs)
v = vecs.T
Xnew = np.dot(v, Xcentered)
print("Xnew   ", Xnew)
colors = ['navy', 'turquoise', 'darkorange']
plt.figure(figsize=(8, 8))

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(Xnew[0, y == i], Xnew[1, y == i],
                color=color, lw=2, label=target_name)

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.axis([-4, 4, -1.5, 1.5])
plt.show()


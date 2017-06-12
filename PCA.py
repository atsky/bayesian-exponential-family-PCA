import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

Xcentered = (X - X.mean(axis=0)).T
covmat = np.cov(Xcentered)

# снижение размерности
_, vecs = np.linalg.eig(covmat)
v = vecs.T
Xnew = np.dot(v, Xcentered)

colors = ['navy', 'turquoise', 'darkorange']
plt.figure(figsize=(8, 8))

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(Xnew[0, y == i], Xnew[1, y == i],
                color=color, lw=2, label=target_name)

plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.axis([-4, 4, -1.5, 1.5])
plt.show()


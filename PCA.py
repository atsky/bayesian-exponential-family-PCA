import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

mean = [[], [], []]
Xc0 = [[], [], []]

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    m = (X[y == label, 0].mean(), X[y == label, 1].mean(), X[y == label, 2].mean(), X[y == label, 3].mean())
    Xcentered0 = X[y == label, 0] - m[0] #для одного вида ириса
    Xcentered1 = X[y == label, 1] - m[1]
    Xcentered2 = X[y == label, 2] - m[2]
    Xcentered3 = X[y == label, 3] - m[3]
    Xc0[label].append(Xcentered0)
    Xc0[label].append(Xcentered1)
    Xc0[label].append(Xcentered2)
    Xc0[label].append(Xcentered3)
    mean[label].append(m)


Xcentered = Xc0
print("Mean vector: ", mean)

covmat = [[], [], []]
covmat[0].append(np.cov(Xcentered[0]))
covmat[1].append(np.cov(Xcentered[1]))
covmat[2].append(np.cov(Xcentered[2]))


# снижение размерности
_, vecs0 = np.linalg.eig(covmat[0])
_, vecs1 = np.linalg.eig(covmat[1])
_, vecs2 = np.linalg.eig(covmat[2])

v0 = -vecs0[:, :]
v1 = -vecs1[:, :]
v2 = -vecs2[:, :]

Xnew0=np.dot(v0, Xcentered[0])
Xnew1=np.dot(v1, Xcentered[1])
Xnew2=np.dot(v2, Xcentered[2])

colors = ['navy', 'turquoise', 'darkorange']
plt.figure(figsize=(8, 8))

plt.scatter(Xnew0[0, 0], Xnew0[0, 1], color=colors[0], lw=2, label="setosa")
plt.scatter(Xnew1[0, 0], Xnew1[0, 1], color=colors[1], lw=2, label="versicolor")
plt.scatter(Xnew2[0, 0], Xnew2[0, 1], color=colors[2], lw=2, label="virginica")
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.axis([-4, 4, -1.5, 1.5])
plt.show()


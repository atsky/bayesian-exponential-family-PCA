import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    m = (X[y == label, 0].mean(), X[y == label, 1].mean(), X[y == label, 2].mean())
    Xcentered0 = X[y == label, 0] - m[0]
    Xcentered1 = X[y == label, 1] - m[1]
    Xcentered2 = X[y == label, 2] - m[2]

Xcentered = (Xcentered0, Xcentered1, Xcentered2)
print(Xcentered)
print("Mean vector: ", m)

covmat = np.cov(Xcentered)
print(covmat, "\n")

# снижение размерности
_, vecs = np.linalg.eig(covmat)
v = -vecs[:, :]
Xnew = np.dot(v, Xcentered)
print("Xnew: ", Xnew)

#colors = ['navy', 'turquoise', 'darkorange']
#plt.figure(figsize=(8, 8))
#for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
#        plt.scatter(Xnew[y == label, 0], Xnew[y == label, 1])
#plt.show()


import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

#Data set
x_pos = np.array([[4, 3], [4, 8], [7, 2]])
y_pos = np.array([1, 1, 1])
x_neg = np.array([[-1, -2], [-1, 3], [2, -1], [2, 1]])
y_neg = np.array([-1, -1, -1, -1])

X = np.concatenate((x_pos, x_neg), axis = 0)
y = np.concatenate((y_pos, y_neg), axis = 0)

clf = SVC(kernel = 'linear', random_state = 8787)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]

print('w = ', w)
print('b = ', b)
print('Support vectors = ', clf.support_vectors_)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 30, cmap = plt.cm.Paired)

# plot the decision function
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method = "contour",
    colors = "k",
    levels = [-1, 0, 1],
    alpha = 0.5,
    linestyles = ["--", "-", "--"],
    ax = ax,
)

# plot support vectors
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s = 100,
    linewidth = 1,
    facecolors = "none",
    edgecolors = "k"
)

plt.show()


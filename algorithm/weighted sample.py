import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
"""
weight outlier makes the deformation of the decision boundary very visible
"""

def plot_decision_function(classifier, sample_weight, axis, title):
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black')
    axis.axis('off')
    axis.set_title(title)


np.random.seed(0)
X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
# print(X)
y = [1] * 10 + [-1] * 10  # Y.shape 为 20*1
# print(Y)
sample_weight_last_ten = abs(np.random.randn(len(X)))
sample_weight_constant = np.ones(len(X))
# bigger weights to the outliers
sample_weight_last_ten[15:] * 5
sample_weight_last_ten[9] *= 15
clf_weights=svm.SVC()
clf_weights.fit(X,y,sample_weight_last_ten)
clf_no_weights = svm.SVC()
clf_no_weights.fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_function(clf_no_weights, sample_weight_constant, axes[0], "constant weight")
plot_decision_function(clf_weights, sample_weight_last_ten, axes[1], 'modified weight')
plt.show()

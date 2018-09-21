import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
make_blobs 聚类数据生成器 n_samples 待生成的样本数，centers 要生成的样本中心数或者确定的中心点
return X 生成的样本数据集 y 样本数据集的标签

"""

X, y = make_blobs(n_samples=40, centers=2, random_state=6)  # X.shape(40,2) y.shape (40,)
# print(X.shape)
# print(y.shape)
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
# print(xlim)
# print(ylim)
xx = np.linspace(xlim[0], xlim[1], 30)
# print(xx)-
yy = np.linspace(ylim[0], ylim[1], 30)
# print(yy.shape)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
# numpy.ravel() 将多维度数组降为一维数组 numpy.vstack() 沿竖直方向堆叠起来
Z = clf.decision_function(xy).reshape(XX.shape)
# 画出 decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# 画出支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none')
plt.show()

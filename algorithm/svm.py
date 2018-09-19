from sklearn import svm
# x=[[0,0],[1,1]]
# y=[0,1]
# clf=svm.SVC()
# clf.fit(x,y) # fit the svm model according to the training data
# print(clf.predict([[2.,2.]]))
# print(clf.support_vectors_)
# print(clf.support_) # the indices of support vectors
# print(clf.n_support_)

# X=questions, Y= list of tags for each question from X
# X=[[0],[1],[2],[3]]
# Y=[0,1,2,3]
# clf=svm.SVC(decision_function_shape='ovo')
# print(clf.fit(X,Y))
# dec=clf.decision_function([[1]])
# print(dec.shape[1])
# clf.decision_function_shape='ovr'
# dec=clf.decision_function([[1]])
# print(dec.shape[1])

# from sklearn.datasets import load_iris
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
#
# data=load_iris()
# X,y=data.data,data.target
# estim1=OneVsRestClassifier(SVC(kernel='linear',decision_function_shape='ovo'))
# a=estim1.fit(X,y)
# print(a)

# decision function 分类决策函数来分割超平面（hyperplane）
# import numpy as np
# from sklearn.svm import SVC
# X=np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
# y=np.array([1,1,2,2])
# clf=SVC()
# clf.fit(X,y)
# print(clf.predict([[-0.8,-1]]))

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """

    :param x: x-axis
    :param y: y-axis
    :param h: stepsize for meshgrid
    :return:
    xx,yy:ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """
    plot the decision boundaries for a classifier
    :param ax: matplotlib axes object
    :param clf: a classifier
    :param xx: meshgrid ndarray
    :param yy: meshgrid ndarray
    :param params: dictionary of params
    :return:
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

C = 1.0
models = (svm.SVC(kernel='linear', C=C), svm.LinearSVC(C=C), svm.SVC(kernel='rbf',
                                                                     gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

titles = ('svc with linear kernel', 'linearSVC(linear kernel)', 'svc with RBF kernel', 'svc with polynomial')
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.6, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('length')
    ax.set_ylabel('width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# xx,yy=np.meshgrid(np.linspace(-3,3,100),np.linspace(-3,3,100))
#
# np.random.seed(0) # seed the generator
# X=np.random.randn(300,2)
# Y=np.logical_xor(X[:,0]>0,X[:,1]>0)
# clf=svm.NuSVC()
# clf.fit(X,Y)
#
# Z=clf.decision_function(np.c_[xx.ravel(),yy.ravel()]) # distance of the samples X to the separating hyperplane
# print(Z.shape)
# Z=Z.reshape(xx.shape)
# print(Z.shape)
# plt.imshow(Z,interpolation='nearest',extent=(xx.min(),xx.max(),yy.min(),yy.max()),
#            aspect='auto',origin='lower')
# contours=plt.contour(xx,yy,Z,levels=[0],linewidths=2,linetypes='--')
# plt.scatter(X[:,0],X[:,1],s=30,c=Y,cmap=plt.cm.Paired,edgecolors='k')
# plt.xticks(())
# plt.yticks(())
# plt.axis([-3,3,-3,3])
# plt.show()

# X=np.array([[0,0],[1,1]])
# # print(X)
# y=[0,1]
# clf=svm.SVC(kernel='precomputed')
# gram=np.dot(X,X.T)
# print(X.T)
# print(gram)
# print(clf.fit(gram,y))
# print(clf.predict(gram))
# linear_svc=svm.SVC(kernel='rbf')
# print(linear_svc.kernel)
#
# def my_kernel(X,Y):
#     return np.dot(X,Y.T)
# clf2=svm.SVC(kernel=my_kernel)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target


def my_kernel(X, Y):
    M = np.array([[2, 0], [0, 1]])
    return np.dot(np.dot(X, M), Y.T)


h = .02
clf = svm.SVC(kernel=my_kernel)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # numpy.c_ 按{行连接}两个矩阵，把矩阵左右相加，要求行数相等

Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-class classification''kernel')
plt.axis('tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# make_gaussian_quantiles 生成分组多维正态分布的数据
X1, y1 = make_gaussian_quantiles(cov=2., n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
# 将两组数据合并为一组数据
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm='SAMME', n_estimators=200)
"""
:param DecisionTreeClassifier (base_estimator)弱分类学习器
:param max_depth 决策树的最大深度 当模型样本量多或者特征多的情况下，推荐限制这个最大深度
:param algorithm 包括SAMME和SAMME.R 两个主要区别在于弱学习器权重的度量，SAMME.R使用了概率度量的连续值
:param n_estimator 弱学习器的最大迭代次数或者说最大的弱学习器的个数
"""
bdt.fit(X, y)
plot_colors = "br"
plot_step = 0.02
class_names = "AB"

# plot the decision boundaries
plt.subplot(132)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# 用网格图来观察拟合的区域
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis("tight")

for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1],
                c=c, cmap=plt.cm.Paired,
                s=20, edgecolors='k',
                label="class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('decision bounary')

twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(133)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[y == i],
             bins=10,
             range=plot_range,
             facecolor=c,
             label='class %s' % n,
             alpha=.5,
             edgecolor='k')
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('samples')
plt.xlabel('score')
plt.title('decision score')
plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()

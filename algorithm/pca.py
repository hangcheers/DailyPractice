import numpy as np
from numpy.linalg import svd
from sklearn.decomposition import PCA

X = np.random.random((5, 8))
X.round(2)
print(X)
#n_components 表示的降维后的维数
pca = PCA(n_components=3,copy=False)
# pca is the unsupervised method ,thus y is none
pca.fit(X)
print(pca)
# remove the column means from data
X_0mean = X - X.mean(0)
X_0mean.round(2)
print(X_0mean)
# do svd
U, s, Vh = svd(X_0mean, full_matrices=False)
print(U.shape, s.shape, Vh.shape)
# display singular value
print(s.round(2))
print(pca.singular_values_.round(2))
print(pca.components_.round(2))
trans_data=pca.fit_transform(X).round(2)
print(X.round(2))
print(trans_data)

X1 = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca1 = PCA(n_components=1)
pca1.fit(X1)
print(pca1.explained_variance_ratio_)
print(pca1.singular_values_)
new_data = pca1.fit_transform(X1)
print(X1)
print(new_data)

import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()  # GNB params:prior
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
print(clf.predict_log_proba([[-0.8, -1]]))
print(clf.predict_proba([[-0.8, -1]]))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))  # incremental fit on a batch of samples
print(clf_pf.predict([[-0.8, -1]]))
print(clf.predict_log_proba([[-0.8, -1]]))
print(clf.predict_proba([[-0.8, -1]]))

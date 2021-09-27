from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from SupervisedLearningUtils import plot_learning_curve
from SupervisedLearningUtils import titanicInput, titanicTarget, seedInput, seedTarget


'''
fig, axes = plt.subplots(3, 2, figsize = (2, 3))

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
title = "Learning Curves (SVM, Linear kernel)"
Knn = KNeighborsClassifier(n_neighbors = 5)
plot_learning_curve(Knn, title, titanicInput, titanicTarget, cv=cv, n_jobs=4)


Knn = KNeighborsClassifier(n_neighbors = 4)
plot_learning_curve(Knn, title, titanicInput, titanicTarget, cv=cv, n_jobs=4)
'''

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
for k in range(1, 6):
    title = "Learning Curve, K-NN w/ K = " + str(k)
    Knn = KNeighborsClassifier(n_neighbors = k)
    plot_learning_curve(Knn, title, titanicInput, titanicTarget, cv=cv, n_jobs=4)
    
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
for k in range(1, 6):
    title = "Learning Curve, K-NN w/ K = " + str(k)
    Knn = KNeighborsClassifier(n_neighbors = k)
    plot_learning_curve(Knn, title, seedInput, seedTarget, cv=cv, n_jobs=4)

plt.show()


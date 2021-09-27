import matplotlib.pyplot as plt
from sklearn.svm import SVC, NuSVC
from sklearn.model_selection import ShuffleSplit
from SupervisedLearningUtils import titanicInput, titanicTarget, seedInput, seedTarget
from SupervisedLearningUtils import plot_learning_curve

fig, axes = plt.subplots(3, 4, figsize = (2, 3))

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

SVC1 = SVC(kernel = 'linear')
title = "Learning Curves (SVM, Linear kernel)"
plot_learning_curve(SVC1, title, titanicInput, titanicTarget, axes=axes[:, 0], cv=cv, n_jobs=4)

SVC2 = SVC(kernel = 'poly')
title = "Learning Curves (SVM, Poly kernel)"
plot_learning_curve(SVC2, title, titanicInput, titanicTarget, axes=axes[:, 1], cv=cv, n_jobs=4)

SVC3 = SVC(kernel = 'rbf')
title = "Learning Curves (SVM, RBF kernel)"
plot_learning_curve(SVC3, title, titanicInput, titanicTarget, axes=axes[:, 2], cv=cv, n_jobs=4)


SVC4 = SVC(kernel = 'sigmoid')
title = "Learning Curves (SVM, Sigmoid kernel)"
plot_learning_curve(SVC4, title, titanicInput, titanicTarget, axes=axes[:, 3], cv=cv, n_jobs=4)



fig, axes = plt.subplots(3, 4, figsize = (2, 3))

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
SVC1 = SVC(kernel = 'linear')
title = "Learning Curves (SVM, Linear kernel)"
plot_learning_curve(SVC1, title, seedInput, seedTarget, axes=axes[:, 0], cv=cv, n_jobs=4)


SVC2 = SVC(kernel = 'poly')
title = "Learning Curves (SVM, Poly kernel)"
plot_learning_curve(SVC2, title, seedInput, seedTarget, axes=axes[:, 1], cv=cv, n_jobs=4)

SVC3 = SVC(kernel = 'rbf')
title = "Learning Curves (SVM, RBF kernel)"
plot_learning_curve(SVC3, title, seedInput, seedTarget, axes=axes[:, 2], cv=cv, n_jobs=4)


SVC4 = SVC(kernel = 'sigmoid')
title = "Learning Curves (SVM, Sigmoid kernel)"
plot_learning_curve(SVC4, title, seedInput, seedTarget, axes=axes[:, 3], cv=cv, n_jobs=4)


#plt.show()

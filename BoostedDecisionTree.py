import matplotlib.pyplot as plt
import numpy as np
import timeit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from SupervisedLearningUtils import titanicInput, titanicTarget, seedInput, seedTarget


def fit_data_prune_tree(constructor, inputs, targets, train_size_ratio):

    trainingInput, testInput, trainingTarget, testTarget = train_test_split(inputs, targets, train_size = train_size_ratio)

    decisionTree = constructor()
    decisionTree.fit(trainingInput, trainingTarget)
        
    pruningPath = decisionTree.cost_complexity_pruning_path(trainingInput, trainingTarget)
    ccp_alphas, impurities = pruningPath.ccp_alphas, pruningPath.impurities

    '''
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    '''


    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = constructor(random_state=0, ccp_alpha=ccp_alpha)
        clf = AdaBoostClassifier(clf)
        clf.fit(trainingInput, trainingTarget)
        clfs.append(clf)


    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    '''
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    '''

    train_scores = [clf.score(trainingInput, trainingTarget) for clf in clfs]
    test_scores = [clf.score(testInput, testTarget) for clf in clfs]
    print(max(test_scores))

    '''
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training size ratio: " + str(train_size_ratio))
    ax.plot(ccp_alphas, train_scores, marker='o', label="Training", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="Testing", drawstyle="steps-post")
    ax.legend()
    '''


'''
for i in np.linspace(.1, .5, 5):
        fit_data_prune_tree(DecisionTreeClassifier, titanicInput, titanicTarget, train_size_ratio = i)

for i in np.linspace(.1, .5, 5):
        fit_data_prune_tree(DecisionTreeClassifier, seedInput, seedTarget, train_size_ratio = i)
'''


start = timeit.default_timer()
fit_data_prune_tree(DecisionTreeClassifier, titanicInput, titanicTarget, train_size_ratio = .5)
stop = timeit.default_timer()
print('Time: ', stop - start)  


start = timeit.default_timer()
fit_data_prune_tree(DecisionTreeClassifier, seedInput, seedTarget, train_size_ratio = .4)
stop = timeit.default_timer()
print('Time: ', stop - start)  


#plt.show()
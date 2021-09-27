from sklearn.neural_network import MLPClassifier
import pandas as pd
import timeit
import warnings
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import ConvergenceWarning
from SupervisedLearningUtils  import titanicInput, titanicTarget, seedInput, seedTarget
from SupervisedLearningUtils import params, labels, plot_args


def plot_learning_rate(X, y):
    # for each dataset, plot learning for each learning strategy
    ##print("\nlearning on dataset %s" % name)
    
    fig, ax = plt.subplots()
    ax.set_title('Learning rate vs Iterations ')
    
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    mlp_scores = {}
    

    max_iter = 400

    for label, param in zip(labels, params):
        ##print("training: %s" % label)
        #start = timeit.default_timer()
  
        mlp = MLPClassifier(random_state=0, max_iter = max_iter, **param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                    module="sklearn")
            mlp.fit(X, y)
        
        ##stop = timeit.default_timer()
        ##print('Time: ', stop - start)

        
        mlps.append(mlp)
        mlp_scores.update({label :{'Score' : mlp.score(X,y), 'Loss' : mlp.loss_}})
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
        print("Iterations: %f" %mlp.n_iter_)
        
    for mlp, label, args in zip(mlps, labels, plot_args):
        ##ax.plot(mlp.loss_curve_, label=label, **args)
        asdf = 1
       
    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Learning rate')

    return pd.DataFrame(mlp_scores)


Df = plot_learning_rate(titanicInput, titanicTarget)

data = plot_learning_rate(seedInput, seedTarget)

##plt.show()


Df.to_csv("titanic_out.csv")
data.to_csv("seeds_out.csv")
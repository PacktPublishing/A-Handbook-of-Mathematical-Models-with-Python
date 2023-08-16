
import  numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

#generate classification dataset
X, y = make_blobs(n_samples = 500, centers = 3, n_features = 2) ##3 class labels

#Model
model = KNeighborsClassifier()

#define search space 
search_space = [Integer(1, 5, name = 'n_neighbors'), Integer(1, 2, name = 'p')]

@use_named_args(search_space)
def evaluate_model(**params):
    model.set_params(**params)
    result = cross_val_score(model, X, y, cv = 5, n_jobs = -1, scoring = 'accuracy')
    estimate = np.mean(result)
    return 1.0 - estimate

#Optimize
result = gp_minimize(evaluate_model, search_space)

print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))

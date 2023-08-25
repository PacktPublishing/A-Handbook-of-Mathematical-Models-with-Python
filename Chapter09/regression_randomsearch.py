
import pandas as pd
from scipy.stats import loguniform
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV

#dataset
df = pd.read_csv('auto-insurance.csv')
data = df.values
X, y = data[:, :-1], data[:, -1]

#Model
model = Ridge()
#validation
cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

#Define search space
space = dict()
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']
space['alpha'] = loguniform(1e-5, 100)
space['fit_intercept'] = [True, False]
space['normalize'] = [True, False]

search = RandomizedSearchCV(model, space, n_iter = 500, scoring = 'neg_mean_absolute_error', n_jobs = -1, cv = cv, random_state = 1)
result = search.fit(X, y)

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)

      

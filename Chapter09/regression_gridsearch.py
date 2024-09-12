#author: ranja.sarkar@gmail.com

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


#dataset for regression
url = 'auto-insurance.csv'
df = pd.read_csv(url, header = None)

data = df.values
X, y = data[:, :-1], data[:, -1]

#Model
model = Ridge()
#validation
cv = RepeatedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

#Define search space
space = dict()
space['alpha'] = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
space['fit_intercept'] = [True, False]
#space['normalize'] = [True, False]
space['max_iter'] = [100, 200, 300]
space['random_state'] = [1, 42]
space['solver'] = ['svd', 'cholesky', 'lsqr', 'sag']

search = GridSearchCV(model, space, scoring = 'neg_mean_absolute_error', n_jobs = -1, cv = cv)
result = search.fit(X, y)

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)




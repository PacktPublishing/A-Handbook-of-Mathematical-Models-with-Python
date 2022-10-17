
#Author: ranja.sarkar@gmail.com

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


X, y = make_classification(n_samples = 1000, n_features = 8, n_informative = 8, n_redundant = 0, random_state = 1)

model = LinearDiscriminantAnalysis()
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)

grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']

search = GridSearchCV(model, grid, scoring = 'accuracy', cv = cv, n_jobs = -1)
results = search.fit(X, y)
print('Mean Accuracy: %.3f' % results.best_score_)

row = [0.1277, -3.6440, -2.2326, 1.8211, 1.7546, 0.1243, 1.0339, 2.3582] #new example
yhat = search.predict([row]) #predict on test data
print('Predicted Class: %d' % yhat) #class probability of new example



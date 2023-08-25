
from numpy import mean
from numpy import std
#from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def get_models():
    models = dict()
    for i in range(1, 11):
        steps = [('pca', PCA(n_components = i)), ('m', LogisticRegression())]
        models[i] = Pipeline(steps = steps)
    return models

def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
    scores = cross_val_score(model, X,  y, scoring = 'accuracy', cv = cv, n_jobs = -1,  error_score = 'raise')
    return scores

#dataset
X, y = make_classification(n_samples = 1000, n_features = 10, n_informative = 8, n_redundant = 2, random_state = 7)

models = get_models()
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(scores)
    names.append(name)

#print('Mean Accuracy: %.4f (%.4f)' % (mean(results), std(results)))

#Visualization
red_square = dict(markerfacecolor = 'r', marker = 's')
plt.boxplot(results, labels = names, showmeans = True, showfliers = True, flierprops = red_square)
plt.grid()
plt.xlabel('Principal Components')
plt.ylabel('Accuracy')
plt.xticks(rotation = 45)
plt.show()

row = [0.1277, -3.6440, -2.2326, 1.8211, 1.7546, 0.1243, 1.0339, 2.3582, -2.8264,0.4491] #new example
steps = [('pca', PCA(n_components = 8)), ('m', LogisticRegression())]
model = Pipeline(steps = steps)
model.fit(X, y)
yhat = model.predict([row]) #predict
print('Predicted Class: %d' % yhat) 

#Import der Libraries

import sys
print('Python version:', sys.version)

import IPython
print('IPython:', IPython.__version__)

from time import time

import numpy as np
print('numpy:', np.__version__)

import scipy
print('scipy:', scipy.__version__)

import pandas as pd
print('pandas:', pd.__version__)

import matplotlib.pyplot as plt
#print('matplotlib:', plt.__version__)

import sklearn
print('scikit-learn:', sklearn.__version__)
from sklearn import metrics
from sklearn import ensemble

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

import sklearn.model_selection as ms
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

import string


# Daten einlesen
data_train = pd.read_csv('D:/Projekte/Toxicity NLP/Data/train.csv')
data_test = pd.read_csv('D:/Projekte/Toxicity NLP/Data/test.csv')

# Definiere Features (X) und Labels (y)
X = data_train['comment_text']
y = data_train['toxic']



pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', ensemble.RandomForestClassifier()),
])

# Random Forest mit Grid Search
params = {
   'clf__n_estimators': (5, 6),
   'clf__max_depth': (1, 2),
   'clf__criterion' : ('entropy', 'gini'),
}



if __name__ == "__main__":
    grid_search = GridSearchCV(pipeline, param_grid=params, n_jobs=-1, verbose=1, cv=2)
    t0 = time()
    grid_search.fit(X, y)
    t1 = time() - t0
    print(f"done in {t1}")
    print()
    
    print(grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")

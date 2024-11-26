import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'peatal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names = names)

tag = dataset['class'].unique()
key = dict((v,k) for k,v in enumerate(tag))
dataset['class'] = dataset['class'].map(key)

array = dataset.values
X = array[:, 0:3]
Y = array[:, 4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


scoring = 'accuracy'

## Nadzorowane
models_1 = []
models_1.append(('LinearRegression', LinearRegression()))
models_1.append(('DecisionTreeRegressor', DecisionTreeRegressor()))
results_1 = []
names_1 = []

for name, model in models_1:
    cv_results = model_selection.cross_val_score(model, X_train, Y_train)
    results_1.append(cv_results)
    names_1.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_1)
ax.set_xticklabels(names_1)
plt.show()
print()


## Nienadzorowane
models_2 = []
models_2.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
models_2.append(('Partial Least Squares Regression', PLSRegression(n_components=3)))

results_2 = []
names_2 = []

for name, model in models_2:
    cv_results = model_selection.cross_val_score(model, X_train, Y_train)
    results_2.append(cv_results)
    names_2.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_2)
ax.set_xticklabels(names_2)
plt.show()
print()

## Częściowo nadzorowane
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X_train, Y_train, test_size=0.8, random_state=42)

results_3 = []
names_3 = []

model = SelfTrainingClassifier(LogisticRegression())
model.fit(X_combined, Y_combined)
cv_results = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring)

results_3.append(cv_results)
names_3.append('LogisticRegression')
msg = "%s: %f (%f)" % ('LogisticRegression', cv_results.mean(), cv_results.std())
print(msg)


#.......

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results_3)
ax.set_xticklabels(names_3)
plt.show()
print()
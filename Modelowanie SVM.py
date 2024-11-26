import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import model_selection
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
groups = []
for i in range(5):
    for j in range(30):
        groups.append(j-2*i)

print()

model = svm.SVC(kernel='linear', C=1)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
custom_precision_scorer = make_scorer(precision_score, average='macro', zero_division=0)
custom_recall_scorer = make_scorer(recall_score, average='macro', zero_division=0)
custom_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

cv_model = []
#cv_model.append(('K-Fold', model_selection.KFold(n_splits=2)))
#cv_model.append(('Repeated K-Fold', model_selection.RepeatedKFold(n_splits=2, n_repeats=2)))
#cv_model.append(('Leave One Out', model_selection.LeaveOneOut()))
#cv_model.append(('Leave P Out', model_selection.LeavePOut(p=2)))
#cv_model.append(('Shuffle & Split', model_selection.ShuffleSplit(n_splits=2, test_size=0.25)))

#cv_model.append(('Stratified k-fold', model_selection.StratifiedKFold(n_splits=2)))
#cv_model.append(('Stratified Shuffle Split', model_selection.StratifiedShuffleSplit()))

cv_model.append(('Grupowanie k-fold', model_selection.GroupKFold(n_splits=5)))
cv_model.append(('Opuść jedną grupę', model_selection.LeaveOneGroupOut()))
cv_model.append(('Leave P Groups Out', model_selection.LeavePGroupsOut(n_groups=3)))
cv_model.append(('Group Shuffle Split', model_selection.GroupShuffleSplit(n_splits=5)))

value = []
scores = []

for name, cv in cv_model:
    #cv_results1 = model_selection.cross_validate(model, X, y, cv=cv, scoring={'accuracy': 'accuracy',                                             
    cv_results1 = model_selection.cross_validate(model, X, y, cv=cv, groups=groups, scoring={'accuracy': 'accuracy',
                                                  'precision_macro': custom_precision_scorer,
                                                  'recall_macro': custom_recall_scorer,
                                                  'f1_macro': custom_f1_scorer})
    #cv_results2 = model_selection.cross_val_score(model, X, y, cv=cv)
    cv_results2 = model_selection.cross_val_score(model, X, y, cv=cv, groups=groups)
    value.append(name)
    value.append(f"cross_val_score: {np.mean(cv_results2)}'")
    for metric in scoring:
        str = f"{metric}: {np.mean(cv_results1['test_' + metric])}"
        value.append(str)
    scores.append(value)
    value = []

for score in scores:
    print(score, end='\n')
print()

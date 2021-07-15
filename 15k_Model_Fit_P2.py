import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

os.chdir('C:/Users/Manish/Desktop/Final_ML/Final_ML/HFC_Data')

X = pd.read_csv('Prospect_Master.csv', index_col ='Prospectno', dtype= {'Prospectno': 'str'})
Y = pd.read_csv('Target.csv', index_col = 'prospectno',usecols =['TGT','prospectno'])

Z = Y.join(X).fillna(0)
Z.isna().sum()
drop_col = [waste for waste in Z.columns if waste.startswith('month_to_consider')]
Z.drop(labels =drop_col, axis =1, inplace = True)

x = Z.iloc[:,1:]
Y = Z['TGT']

Data = pd.concat([x,Y], axis=1)

Data["Value1"] = pd.cut(Data["num_enq_Secured_LFT"],
                               bins=[-1.0, 1.0, 3.0, 5.0, 8.0, np.inf],
                               labels=[1, 2, 3, 4, 5])
Data["Value2"] = pd.cut(Data["num_enq_Unsecured_LFT"],
                               bins=[-1.0, 1.0, 3.0, 5.0, 8.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

train_set, test_set = train_test_split(Data, test_size=0.2, random_state=42,stratify=Data[['TGT', 'Value1','Value2']])

Cols = ['Value1','Value2']

train_set.drop(labels = Cols,axis=1,inplace=True)
test_set.drop(labels = Cols,axis=1,inplace=True)

X_train, y_train = train_set.iloc[:,0:-1],train_set['TGT']
X_test, y_test = test_set.iloc[:,0:-1],test_set['TGT']


#Logistic Classifier

Logreg = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
Logreg.fit(X_train, y_train)

Logreg_Scores_KFold = cross_val_score(Logreg, X_train, y_train, cv=5, scoring="accuracy")


Pred_y_train_Logreg = cross_val_predict(Logreg, X_train, y_train, cv=5)

CM_Logreg = confusion_matrix(y_train, Pred_y_train_Logreg)


precision_score(y_train, Pred_y_train_Logreg)

recall_score(y_train, Pred_y_train_Logreg)


f1_score(y_train, Pred_y_train_Logreg)

y_scores_Logreg = cross_val_predict(Logreg, X_train, y_train, cv=5,method="decision_function")

Logreg2 = LogisticRegression(max_iter=1000, tol=1e-3, random_state=42)
Logreg2.fit(X_train, y_train)
y_scores_Logreg2 = Logreg2.predict_proba(X_train)[:,1]

Pred_y_train_Logreg2 = np.multiply(y_scores_Logreg2 > 0.5,1)

CM_Logreg2 = confusion_matrix(y_train, Pred_y_train_Logreg2)
precision_score(y_train, Pred_y_train_Logreg2)
recall_score(y_train, Pred_y_train_Logreg2)
f1_score(y_train, Pred_y_train_Logreg2)

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_Logreg2)
Logreg2_DF = pd.DataFrame(['precisions','recalls','thresholds'], axis=1)
k = pd.concat([pd.DataFrame(precisions),pd.DataFrame(recalls),pd.DataFrame(thresholds)],axis=1)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([0, 1, 0, 1])             # Not shown

plt.figure(figsize=(8, 4))                      # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
threshold_70_precision = thresholds[np.argmax(precisions >= 0.6)]




#threshold = 0
#y_train_pred2 = np.multiply(y_scores > threshold,1)
#
#threshold_70_recall = thresholds[np.argmin(recalls >= 0.7)]
#threshold = threshold_70_recall
#y_train_pred2 = np.multiply(y_scores > threshold,1)
#confusion_matrix(y_train, y_train_pred2)
#precision_score(y_train, y_train_pred2)
#recall_score(y_train, y_train_pred2)
#f1_score(y_train, y_train_pred2)


#threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

%matplotlib qt

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown

plt.figure(figsize=(8, 4))                      # Not shown
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)



def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                         # Not shown
plot_roc_curve(fpr, tpr)


from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_scores)


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3,
                                    method="predict_proba")


y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train,y_scores_forest)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([4.837e-3, 4.837e-3], [0., 0.4368], "r:")
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")
plt.plot([4.837e-3], [0.4368], "ro")
plt.plot([4.837e-3, 4.837e-3], [0., 0.9487], "r:")
plt.plot([4.837e-3], [0.9487], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)
plt.show()

roc_auc_score(y_train, y_scores_forest)

y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train, cv=3)
precision_score(y_train, y_train_pred_forest)

recall_score(y_train, y_train_pred_forest)
f1_score(y_train, y_train_pred_forest)



from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)
knn_clf.fit(X_train, y_train)

y_knn_pred = knn_clf.predict(X_train)

knn_scores = cross_val_score(knn_clf, X_train, y_train, cv=10)
knn_scores


roc_auc_score(y_train, y_knn_pred)
precision_score(y_train, y_knn_pred)
recall_score(y_train, y_knn_pred)


from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)



svm_clf2.fit(X_train, y_train)

svm_clf2 = SVC(kernel="linear", C=10**9)
svm_clf2.fit(X_train, y_train)


LinearSVC(C=1, loss="hinge", random_state=42)

svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

SVC(kernel="rbf", gamma=5, C=0.001)




y_pred_svm = svm_clf.predict(X_train)
roc_auc_score(y_train, y_pred_svm)
precision_score(y_train, y_pred_svm)
recall_score(y_train, y_pred_svm)

#
#from sklearn.model_selection import GridSearchCV
#
#param_grid = [
#    # try 12 (3×4) combinations of hyperparameters
#    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#    # then try 6 (2×3) combinations with bootstrap set as False
#    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#  ]
#
#forest_reg = RandomForestRegressor(random_state=42)
## train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
#grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                           scoring='neg_mean_squared_error',
#                           return_train_score=True)
#grid_search.fit(housing_prepared, housing_labels)





from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
log_reg.fit(X_train, y_train)
y_proba = log_reg.predict_proba(X_train)

from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X_train, y_train)
tree_reg2.fit(X_train, y_train)

tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(X_train, y_train)
tree_reg2.fit(X_train, y_train)


from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')


voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    


#soft voting
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)







# D feature scaling now
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.decomposition import PCA

os.chdir('C:/Users/Manish/Desktop/Final_ML/Final_ML/HFC_Data')



X = pd.read_csv('Final_Applicant_Data.csv', index_col ='Prospectno', dtype= {'Prospectno': 'str'})
#
Y = pd.read_csv('Target.csv', index_col = 'prospectno',usecols =['TGT','prospectno'])


Z = Y.join(X).fillna(0)
Z.isna().sum()
drop_col = [waste for waste in Z.columns if waste.startswith('month_to_consider')]
Z.drop(labels =drop_col, axis =1, inplace = True)

x = Z.iloc[:,1:]
Y = Z['TGT']

X_train, X_test, y_train, y_test = train_test_split(
    x,
    Y,
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape

#y_train = pd.DataFrame(y_train)
#y_test = pd.DataFrame(y_test)


Cols_Null = [i for i in Z.columns if Z[i].isnull().sum()>0]

constant_features = [
    i for i in X_train.columns if X_train[i].std() == 0
]


X_train.drop(labels =constant_features, axis =1, inplace = True)
X_test.drop(labels =constant_features, axis =1, inplace = True)


## and now find those columns that contain only 1 label:
#constant_features = [
#    feat for feat in X_train.columns if len(X_train[feat].unique()) == 1
#]
#
#len(constant_features)

quasi_constant_features = []
for i in X_train.columns:
    if (100*X_train[i].value_counts()/len(X_train[i])).sort_values(ascending = False)[0]>99.8:
        quasi_constant_features.append(i)
        
X_train.drop(labels =quasi_constant_features, axis =1, inplace = True)
X_test.drop(labels =quasi_constant_features, axis =1, inplace = True)

#X_train['D1'] = X_train['num_trade_Credit_Card_12M']

# check for duplicated features in the training set
#duplicated_features1 =[]
#duplicated_features2 =[]
#for i in range(len(X_train.columns)):
#    for n in range(i+1,len(X_train.columns)):
#        if X_train.iloc[:,i].equals(X_train.iloc[:,n]):
#            duplicated_features1.append(X_train.columns[i])
#            duplicated_features2.append(X_train.columns[n])
#duplicated_features = pd.DataFrame(list(zip(duplicated_features1,duplicated_features2)))
            


scaler = StandardScaler()
scaler.fit(X_train)

#sel_ = SelectFromModel(LogisticRegression(C=10, penalty='l1'))
#sel_.fit(scaler.transform(X_train), y_train)
#
#Lasso_imp = list(X_train.columns[(sel_.get_support())])

#SGD Classifier

#np.sum(sel_.estimator_.coef_ == 0)

#X_train = X_train.loc[:,Lasso_imp]
#X_test = X_test.loc[:,Lasso_imp]


roc_values = []
for feature in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train[feature].to_frame(), y_train)
    y_scored = clf.predict_proba(X_test[feature].to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
    
ROC_Coeff = pd.DataFrame(list(zip(list(X_train.columns), roc_values)))
ROC_Coeff.columns = ['Var','roc']
ROC_Coeff.sort_values(by='roc', ascending = False, inplace= True)

ROC_Coeff = ROC_Coeff[ROC_Coeff['roc']>.55]


#corr = X_train.corr().abs()
#
#corr = pd.DataFrame(corr.unstack())
#corr = corr.reset_index()
#corr.columns = ['Var1','Var2','corr']
#corr = corr[(corr['corr']<1) & (corr['corr']>=.8)].sort_values(by = ['corr'],ascending = False)
#corr['counter'] = 1
#corr['cumsum'] = corr.groupby(['Var1'])['counter'].cumsum()
#corr_mat = corr.pivot_table(index = 'Var1', columns = 'cumsum', values = 'Var2', aggfunc = 'sum')
#corr_mat = corr_mat.reset_index().sort_values(by = 'Var1')


#ROC_Copy = ROC_Coeff.copy()
#ROC_Copy.set_index('Var',inplace = True)
#k = ROC_Copy.groupby(['Var','roc'])
#mapp = ROC_Copy.to_dict()
#
#corr_mat['C1'] = corr_mat['Var1'].map(mapp)



#dictt = dict(zip(ROC_Coeff['Var'],ROC_Coeff['roc']))
#
#for col in range(len(corr_mat.columns)):
#    corr_mat['C' +str(col+1)] = corr_mat.iloc[:,col].map(dictt)
#    
#
#corr_mat['Max'] = corr_mat.iloc[:,10:20].max(axis=1)
#corr_mat['ID'] = corr_mat.iloc[:,10:20].idxmax(axis=1)
#
#corr_mat['Max_Feature'] = corr_mat['ID'].apply(lambda x: int(x[1])-1)
#
#for r in range(corr_mat.shape[0]):
#    corr_mat.loc[r,'Feature'] = corr_mat.iloc[r,corr_mat['Max_Feature'][r]]
#    
#Final_Features = list(corr_mat['Feature'].unique())

Final_Features = list(ROC_Coeff['Var'])
X_train = X_train.loc[:,Final_Features]
X_test = X_test.loc[:,Final_Features]


sum(y_test==1)




#corr = X_train.corr().abs()
#corr = pd.DataFrame(corr.unstack())
#corr = corr.reset_index()
#corr.columns = ['Var1','Var2','corr']
#corr = corr[(corr['corr']<1) & (corr['corr']>=.8)].sort_values(by = ['corr'],ascending = False)
#corr['counter'] = 1
#corr['cumsum'] = corr.groupby(['Var1'])['counter'].cumsum()
#corr_mat = corr.pivot_table(index = 'Var1', columns = 'cumsum', values = 'Var2', aggfunc = 'sum')
#corr_mat = corr_mat.reset_index().sort_values(by = 'Var1')
#
#dictt = dict(zip(ROC_Coeff['Var'],ROC_Coeff['roc']))
#
#for col in range(len(corr_mat.columns)):
#    corr_mat['C' +str(col+1)] = corr_mat.iloc[:,col].map(dictt)
#    
#
#corr_mat['Max'] = corr_mat.iloc[:,4:8].max(axis=1)
#corr_mat['ID'] = corr_mat.iloc[:,4:8].idxmax(axis=1)
#
#corr_mat['Max_Feature'] = corr_mat['ID'].apply(lambda x: int(x[1])-1)
#
#for r in range(corr_mat.shape[0]):
#    corr_mat.loc[r,'Feature'] = corr_mat.iloc[r,corr_mat['Max_Feature'][r]]
#    
#Final_Features = list(corr_mat['Feature'].unique())
#
#X_train = X_train.loc[:,Final_Features]
#X_test = X_test.loc[:,Final_Features]



#pca =PCA(.98)
#pca.fit(X_train)
#
#pca.n_components_ 
#
#
#principal_components = pca.fit_transform(X_train)
#principal_df = pd.DataFrame(data = principal_components)
#print(principal_df.shape)
#
#
#pca.n_features_
#
#
#
#
#sel_ = SelectFromModel(RandomForestClassifier(n_estimators=100))
#sel_.fit(X_train, y_train)
#
#selected_features =list( X_train.columns[(sel_.get_support())])


log_reg = LogisticRegression(C = 0.0001)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict_proba(X_test)[:,1]
y_pred = (log_reg_pred > .45271)*1
sum(y_pred==1)




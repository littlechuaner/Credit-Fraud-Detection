#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd #To hand with data 
import numpy as np #To math 
import seaborn as sns #to visualization
import matplotlib.pyplot as plt # to plot the graphs
import matplotlib.gridspec as gridspec # to do the grid of plots
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import time


# In[40]:


df_credit = pd.read_csv("creditcard.csv")
#looking the how data looks
df_credit.head()


# In[53]:


#Looking the V's features
columns = df_credit.iloc[:,1:29].columns

frauds = df_credit.Class == 1
normals = df_credit.Class == 0

grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(15,20*4))

for n, col in enumerate(df_credit[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(df_credit[col][frauds], bins = 50, color='g') #Will receive the "semi-salmon" violin
    sns.distplot(df_credit[col][normals], bins = 50, color='r') #Will receive the "ocean" color
    ax.set_ylabel('Density')
    ax.set_title(str(col))
    ax.set_xlabel('')
plt.savefig("variablehist.png")
plt.show()


# In[54]:


colormap = plt.cm.Greens

plt.figure(figsize=(20,16))

sns.heatmap(round(df_credit.corr(),2),linewidths=0.1,vmax=1.0, 
            square=True, cmap = colormap, linecolor='white', annot=True)
plt.savefig("correlationplot.png")
plt.show()


# In[35]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


# In[41]:


df_credit = df_credit.sample(frac=0.2,replace=True, random_state=1212)
X = df_credit.drop(["Class"], axis=1).values #Setting the X to do the split
y = df_credit["Class"].values # transforming the values in array
# the function that we will use to better evaluate the model
def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f: {}".format(fbeta_score(true_value, pred, beta=0.5)))
#Showing the diference before and after the transformation used
print("normal data distribution: {}".format(Counter(y)))
X_smote, y_smote = SMOTE().fit_sample(X, y)
print("SMOTE data distribution: {}".format(Counter(y_smote)))
# splitting data into training and test set
X_train, X_test_o, y_train, y_test_o = train_test_split(X_smote, y_smote, random_state=2, test_size=0.20)
X_train_o, X_test, y_train_o, y_test = train_test_split(X_smote, y_smote, random_state=2, test_size=0.20)


# In[42]:


#RF
#params of the model
start = time.clock()
param_grid = {"max_depth": [3,5,7],
              "n_estimators":[3,5,10],
              "max_features": [5,6,7,8]}
# Creating the classifier
classifier = RandomForestClassifier(max_features=3, max_depth=2 ,n_estimators=10, 
                                    random_state=3, criterion='entropy', n_jobs=-1, verbose=1 )
grid_search = GridSearchCV(classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
end = time.clock()
print('运行时间:',end - start)


# In[43]:


rf = RandomForestClassifier(max_features=grid_search.best_params_['max_features'], 
                               max_depth=grid_search.best_params_['max_depth'] ,
                               n_estimators=grid_search.best_params_['n_estimators'],
                               random_state=43, 
                               criterion='entropy', n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)
rf_pre = rf.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, rf_pre))
print_results("\nSMOTE + RandomForest classification", y_test, rf_pre)


# In[44]:


#Bagging KNN
start = time.clock()
knn = KNeighborsClassifier()
k_range = list(range(1,10))
weight_options = ['uniform','distance']
algorithm_options = ['auto']
param_gridknn = dict(n_neighbors = k_range,weights = weight_options,algorithm=algorithm_options)
#grid search
grid_search = GridSearchCV(knn,param_gridknn,cv=5,scoring='accuracy',verbose=1,n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
end = time.clock()
print('运行时间:',end - start)


# In[45]:


knn = BaggingClassifier(KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'],
                                              weights=grid_search.best_params_['weights'],
                                              algorithm=grid_search.best_params_['algorithm']),
                          max_samples=0.5, max_features=0.5)
knn.fit(X_train, y_train)
knn_pre = knn.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, knn_pre))
print_results("\nSMOTE + Bagging KNNclassification", y_test, knn_pre)


# In[46]:


#logit
start = time.clock()
param_grid = {'C': [0.01, 0.1, 1, 10],
             'penalty':['l1', 'l2']}

logreg = LogisticRegression(random_state=2)

grid_search_lr = GridSearchCV(logreg, param_grid=param_grid, scoring='recall', cv=5,n_jobs=-1)

grid_search_lr.fit(X_train, y_train)
# The best recall obtained
print(grid_search_lr.best_score_)
#Best parameter on trainning set
print(grid_search_lr.best_params_)
end = time.clock()
print('运行时间:',end - start)


# In[49]:


# Creating the model
lr = LogisticRegression(C=grid_search_lr.best_params_['C'], penalty=grid_search_lr.best_params_['penalty'])
lr.fit(X_train, y_train)
logit_pre = lr.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, logit_pre))
print_results("\nSMOTE + LogisticRegression", y_test, logit_pre)


# In[50]:


#GBDT
start = time.clock()
parameters = {'loss': ['deviance', 'exponential'], 
              'learning_rate': [0.05,0.1,0.2],
             'n_estimators':[100],
             'max_depth':[3,5,7,10]}
classifier=GradientBoostingClassifier()
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
print("The best accuracy using gridSearch is", best_accuracy)

best_parameters = grid_search.best_params_
print("The best parameters for using this model is", best_parameters)
end = time.clock()
print('运行时间:',end - start)


# In[51]:


# Creating the model 
gbdt = GradientBoostingClassifier(loss=grid_search.best_params_['loss'],
                                  learning_rate=grid_search.best_params_['learning_rate'],
                                  n_estimators=grid_search.best_params_['n_estimators'],
                                  max_depth=grid_search.best_params_['max_depth'])
gbdt.fit(X_train, y_train)
gbdt_pre = gbdt.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, gbdt_pre))
print_results("\nSMOTE + GBDT", y_test, gbdt_pre)


# In[87]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#ROC
rf_fpr, rf_tpr, rf_thresold = roc_curve(y_test, rf_pre)
knn_fpr, knn_tpr, knn_threshold = roc_curve(y_test, knn_pre)
logit_fpr, logit_tpr, logit_threshold = roc_curve(y_test, logit_pre)
gbdt_fpr, gbdt_tpr, gbdt_threshold = roc_curve(y_test, gbdt_pre)


def graph_roc_curve_multiple(rf_fpr, rf_tpr, knn_fpr, knn_tpr, logit_fpr, logit_tpr, gbdt_fpr, gbdt_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC Curve \n 4 Classifiers', fontsize=18)
    plt.plot(rf_fpr, rf_tpr, label='Random Forest AUC: {:.4f}'.format(roc_auc_score(y_test, rf_pre)))
    plt.plot(knn_fpr, knn_tpr, label='Bagging kNN AUC: {:.4f}'.format(roc_auc_score(y_test, knn_pre)))
    plt.plot(logit_fpr, logit_tpr, label='Logistic Regression AUC: {:.4f}'.format(roc_auc_score(y_test, logit_pre)))
    plt.plot(gbdt_fpr, gbdt_tpr, label='GBDT AUC: {:.4f}'.format(roc_auc_score(y_test, gbdt_pre)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(rf_fpr, rf_tpr, knn_fpr, knn_tpr, logit_fpr, logit_tpr, gbdt_fpr, gbdt_tpr)
plt.savefig("ROC.png",dpi=400)
plt.show()


# In[85]:


X_train = pd.DataFrame(X_train)
features = X_train.columns
importances = rf.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize = (10,6))
plt.title('Feature Importances(Random Forest)')
plt.barh(range(len(indices)), importances[indices], color='coral', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig("RFimp.png",dpi=400)
plt.show()


# In[84]:


X_train = pd.DataFrame(X_train)
features = X_train.columns
importances = gbdt.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize = (10,6))
plt.title('Feature Importances(GBDT)')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig("GBDTimp.png",dpi=400)
plt.show()


# In[83]:


log_reg_cf = confusion_matrix(y_test, logit_pre)
kneighbors_cf = confusion_matrix(y_test, knn_pre)
rf_cf = confusion_matrix(y_test, rf_pre)
gbdt_cf = confusion_matrix(y_test, gbdt_pre)

fig, ax = plt.subplots(2, 2,figsize=(22,12))


sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, cmap=plt.cm.copper)
ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, cmap=plt.cm.copper)
ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(rf_cf, ax=ax[1][0], annot=True, cmap=plt.cm.copper)
ax[1][0].set_title("Random Forest \n Confusion Matrix", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(gbdt_cf, ax=ax[1][1], annot=True, cmap=plt.cm.copper)
ax[1][1].set_title("GBDT \n Confusion Matrix", fontsize=14)
ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

plt.savefig("cm.png",dpi=400)
plt.show()


# In[ ]:





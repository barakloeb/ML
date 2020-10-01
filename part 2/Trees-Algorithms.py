#        Machine Learning Project Part B     #
#    This Class is For DesicisionTrees only  #
#                                            #
#                                            #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import validation_curve
from xgboost.sklearn import XGBClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from sklearn.ensemble import RandomForestClassifier
#import xgboost
#from xgboost.sklearn import XGBClassifier







###############################Features##############################################




#Extraction the Data and seperate it to X and Y
EditedData = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\EditedData.csv")


##Scaling The DATA for All Orlinaly and Continueus Features
def ScalingMinMaxFeat(data,ListOfFeatures):
    for Featu in ListOfFeatures:
        data[Featu] = (data[Featu] - np.mean(data[Featu])) / (
                    np.max(data[Featu]) - np.min(data[Featu]))

ListOfFeaturesToScale = ['Age','Log(Age)','Fee','Log(Fee)','Quantity','PhotoAmt','Pop','Density','GDPPP']
ScalingMinMaxFeat(EditedData,ListOfFeaturesToScale)



X = EditedData[EditedData.columns[:-1]]
Y = EditedData['y']


##Handeling Binary Features Putting 0,1
def EncodeBinaryFeatures(data,ListOfFeatures):
    lb_make = LabelEncoder()
    for Featu in ListOfFeatures:
        EditedData[Featu] = lb_make.fit_transform(EditedData[Featu])

ListOfFeaturesToBinary = ['Type']
EncodeBinaryFeatures(X,ListOfFeaturesToBinary)

X = pd.get_dummies(X, columns=['Gender','Color1','Color2','MaturitySize','Vaccinated','Dewormed','Sterilized','Health','State'])

print(X.head())



######################################################################################################


##Run A Default tree
#DefaultTree = DecisionTreeClassifier()
#DefaultTree_cv_results = cross_validate(DefaultTree, X, Y, cv=8 ,return_train_score=True)
#print('DefaultTree train_score ' , np.mean(DefaultTree_cv_results['train_score']))
#print('DefaultTree test_score ' , np.mean(DefaultTree_cv_results['test_score']))


##Run the best tree
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
clf = DecisionTreeClassifier(random_state=123, max_depth=6)
model = clf.fit(X_train, y_train)


##DecisionTreeClassifier , Running to range Depth Tree, Returning a table with results
def DepthSearch(range):
    DesicionTreeRes = pd.DataFrame(columns=['Max-depth', 'Mean_Score_Test', 'Mean_Score_Train'])
    for i in range:
        clf = DecisionTreeClassifier(random_state=1111,max_depth=i)
        cv_results = cross_validate(clf, X, Y, cv=8,return_train_score=True)
        Mean_Score_Train = np.mean(cv_results['train_score'])
        Mean_Score_Test = np.mean(cv_results['test_score'])
        DesicionTreeRes = DesicionTreeRes.append({'Max-depth':i,'Mean_Score_Test':Mean_Score_Test,'Mean_Score_Train':Mean_Score_Train},ignore_index=True)
    print(DesicionTreeRes)


#DepthSearch(list(range(1,10)))



##Example
##print(X.iloc[7000])
##print(Y.iloc[7000])


##Run A Special Tree
clf = DecisionTreeClassifier(random_state=123, max_depth=6)
model = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import  confusion_matrix,accuracy_score
from sklearn.metrics import f1_score
print(confusion_matrix(y_test,predictions))
print('F1 Score   ' ,f1_score(y_test,predictions,average='micro'))
print('------------------Test---------------')
print(accuracy_score(y_test,predictions))
print('------------------Train---------------')
print(accuracy_score(y_train,clf.predict(X_train)))

def plot_validation_curve(X, y, estimator, param_range, title='Valdation_curve', alpha=0.1,
                          scoring='accuracy', param_name="max_depth", cv=10, save=False, rotate=False):
    train_scores, test_scores = validation_curve(estimator,
                                                 X, y, param_name=param_name, cv=cv,
                                                 param_range=param_range, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    best_param_value = max(test_scores_mean)
    best_param = param_range[list(test_scores_mean).index(best_param_value)]

    plt.figure(figsize=(15, 15))
    sort_idx = np.argsort(param_range)
    param_range = np.array(param_range)[sort_idx]
    train_mean = np.mean(train_scores, axis=1)[sort_idx]
    train_std = np.std(train_scores, axis=1)[sort_idx]
    test_mean = np.mean(test_scores, axis=1)[sort_idx]
    test_std = np.std(test_scores, axis=1)[sort_idx]
    plt.plot(param_range, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(param_range, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)

    plt.title(title + "- Best {} is {} with {} CV mean score of {}".format(param_name, best_param, scoring,
                                                                           round(best_param_value, 3)))

    plt.grid(ls='--')
    plt.xlabel(param_name)
    plt.xticks(param_range)
    if rotate:
        plt.xticks(rotation='vertical')
    plt.ylabel('Average values and standard deviation')
    plt.legend(loc='best')

    print(
        "Best {} is {} with {} CV mean score of {}".format(param_name, best_param, scoring, round(best_param_value, 3)))
    if save:
        plt.savefig('Plots/' + title + '_' + param_name)
    plt.show()


#plot_validation_curve(X, Y,DecisionTreeClassifier(),list(range(2,12)),cv=8)


#RNFRes2=pd.DataFrame(columns=['n_est','Mean_Score_Test','Mean_Score_Train'])
#for i in range(1,2):
#    RNF = RandomForestClassifier(n_estimators=i, max_depth=20,random_state=1438)
#    cv_results_RNF = cross_validate(RNF, X, Y, cv=8,return_train_score=True)
#    Mean_Score_Train_RNF = np.mean(cv_results_RNF['train_score'])
#    Mean_Score_Test_RNF = np.mean(cv_results_RNF['test_score'])
#    RNFRes2 = RNFRes2.append({'n_est':i,'Mean_Score_Test':Mean_Score_Test_RNF,'Mean_Score_Train':Mean_Score_Train_RNF},ignore_index=True)

#print(RNFRes2)

##-------RandomForest--------##
'''
RNFRes2=pd.DataFrame(columns=['n_est','depth','Mean_Score_Test','Mean_Score_Train'])
for i in range(5,20):
    RNF = RandomForestClassifier(n_estimators=50, max_depth=i,random_state=1111)
    cv_results_RNF = cross_validate(RNF, X, Y, cv=8,return_train_score=True)
    Mean_Score_Train_RNF = np.mean(cv_results_RNF['train_score'])
    Mean_Score_Test_RNF = np.mean(cv_results_RNF['test_score'])
    RNFRes2 = RNFRes2.append({'n_est':'40','depth':i,'Mean_Score_Test':Mean_Score_Test_RNF,'Mean_Score_Train':Mean_Score_Train_RNF},ignore_index=True)

d = RNFRes2.to_csv(r'C:\\Users\yonat\PycharmProjects\PaetB\export_RN.csv' , index=False)
print(RNFRes2)

'''

##-------GradientBoosting--------##
from sklearn.ensemble import GradientBoostingClassifier

#GBRes=pd.DataFrame(columns=['n_est','depth','Mean_Score_Test','Mean_Score_Train'])
#for i in range(2,10):
#    GB = GradientBoostingClassifier(n_estimators=30, max_depth = i, random_state = 0)
#    cv_results_GB = cross_validate(GB, X, Y, cv=8,return_train_score=True)
#    Mean_Score_Train_GB = np.mean(cv_results_GB['train_score'])
#    Mean_Score_Test_GB = np.mean(cv_results_GB['test_score'])
#    GBRes = GBRes.append({'n_est':'40','depth':i,'Mean_Score_Test':Mean_Score_Test_GB,'Mean_Score_Train':Mean_Score_Train_GB},ignore_index=True)


#d = GBRes.to_csv(r'C:\Users\yonat\PycharmProjects\PaetB\export_GB.csv' , index=False)
#print(GBRes)



#d = KNNRes.to_csv(r'C:\Users\yonat\PycharmProjects\PaetB\export_KNN.csv' , index=False)
#print(KNNRes)


##------------------------------------XGBooat----------------------------------------------##
from sklearn.ensemble import GradientBoostingClassifier

XGBRes=pd.DataFrame(columns=['n_est','depth','Mean_Score_Test','Mean_Score_Train'])
for i in range(3,12):
    XGB = XGBClassifier(n_estimators=30, max_depth = i, random_state = 0)
    cv_results_XGB = cross_validate(XGB, X, Y, cv=8,return_train_score=True)
    Mean_Score_Train_XGB = np.mean(cv_results_XGB['train_score'])
    Mean_Score_Test_XGB = np.mean(cv_results_XGB['test_score'])
    XGBRes = XGBRes.append({'n_est':'40','depth':i,'Mean_Score_Test':Mean_Score_Test_XGB,'Mean_Score_Train':Mean_Score_Train_XGB},ignore_index=True)

print(XGBRes)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans
from sklearn import metrics
import seaborn as sns
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import classification_report, confusion_matrix


################---------------------------Features----------------------########################


EditedData = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\EditedDataa.csv")


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

X = pd.get_dummies(X, columns=['Breed1','Gender','Color1','Color2','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','State'])


##########################-------------------------------------------------#################################





from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,shuffle =False)





##Buliding A regular model
model=KMeans(n_clusters=4, random_state=0)
model.fit(X_train)
y_pred = model.predict(X_test)


print('yyyyyyyyyyy' , type(y_test))

print(y_test.value_counts())


#result = pd.concat([X, y_pred] , axis=1)
#result = pd.concat([y_pred, y_test] , axis=1,ignore_index =True)
Match_Martix = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

#print(result)

#for i in range(0,len(result)):
#    Match_Martix[int(result.iloc[i,0])][int(result.iloc[i,1])] = Match_Martix[int(result.iloc[i,0])][int(result.iloc[i,1])]+1








##Centers Analisys
def CenterAnalisys():
    CentersTot=[]
    for i in range(1,210):
        CentersTot.append(0)


    for i in range (0,209):
        CentersTot[i] = np.abs(model.cluster_centers_[0][i] - model.cluster_centers_[1][i]) + np.abs(model.cluster_centers_[1][i] - model.cluster_centers_[2][i]) + np.abs(model.cluster_centers_[2][i] - model.cluster_centers_[0][i])

    print(CentersTot)
    df = pd.DataFrame(CentersTot,columns=['Sum dist to Feature - Group 10'])
    df.set_index(X.columns.values)
    df = df.sort_values(by = ['Sum dist to Feature - Group 10'])
    lines = df.plot(kind = "bar")
    plt.show()
#CenterAnalisys()

#print('Adj_Rand_idx ', round(metrics.adjusted_rand_score(Y, model.labels_), 4))
#print('Homogenity ', round(metrics.homogeneity_score(Y, model.labels_), 4))
#print('inertia ', model.inertia_)
#print('silhouette_score ', metrics.silhouette_score(X, model.labels_))


##Elbow !!
from scipy.spatial.distance import cdist

# create new plot and data
#plt.plot()

#colors = ['b', 'g', 'r']
#markers = ['o', 'v', 's']



##k means determine k
distortions = []
K = range(2,3)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

##Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

####-----------------------------------------------------#####






####---------------Dimentionalty Reduction---------------#####


def PlotPCA():
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2).fit(X)
    pca_2d = pca.transform(X)
    pca_2d_c = pca.transform(model.cluster_centers_)

    for i in range(0, pca_2d.shape[0]):
        if Y.iloc[i] == 0:
            c1 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='y',
            marker='o')
        elif Y.iloc[i] == 1:
            c2 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='g',
            marker='o')
        elif Y.iloc[i] == 2:
            c3 = plt.scatter(pca_2d[i,0],pca_2d[i,1],c='b',
        marker='o')

    for i in range(0, pca_2d_c.shape[0]):
        if i == 0:
            c11 = plt.scatter(pca_2d_c[i,0],pca_2d_c[i,1],c='y',
            marker='x')
        elif i == 1:
            c22 = plt.scatter(pca_2d_c[i,0],pca_2d_c[i,1],c='g',
            marker='x')
        elif i == 2:
            c33 = plt.scatter(pca_2d_c[i,0],pca_2d_c[i,1],c='b',
        marker='x')

    plt.legend([c1, c2, c3], ['0', '1', '2'])

    plt.title('PCA - Dimentionality Reduction - True Classes')

    plt.show()

#PlotPCA()
###########-------------------------------------------####################


def kMeansReslookup():
    kMeansRes = pd.DataFrame(columns=['n_clusters', 'silhouette_avg', 'inertia', 'inertia/silhouette'])
    for n_clusters in range(2, 10):
      km = KMeans(n_clusters=n_clusters)
      cluster_labels = km.fit_predict(X)
      iner = km.inertia_
      silhouette_avg = metrics.silhouette_score(X, cluster_labels)
      kMeansRes = kMeansRes.append({'n_clusters':n_clusters,'silhouette_avg':silhouette_avg,
                                    'inertia':iner,'inertia/silhouette': iner/silhouette_avg},ignore_index=True)
    plt.plot(list(range(2,10)), kMeansRes['inertia/silhouette'], label='score', color='blue', marker='o')
    plt.title('inertia/silhouette VS. n_clusters')
    plt.xlabel('K')
    plt.show()
    print(kMeansRes)
    d = kMeansRes.to_csv(r'C:\Users\yonat\PycharmProjects\PaetB\kMeansRes.csv', index=False)


kMeansReslookup()


def conditions():
    if (result['y_pred'] == 0):
        return 'g'
    elif(result['y_pred'] == 1):
        return 'y'
    else:
        return 'b'

def ScatterFeat():
    d = {0: 'g', 1: 'b', 2: 'y'}
    result['color'] = result['y_pred'].map(d)

    print(result)

    print(result[result.columns[7]])
    #result['color']=result['y_pred'].apply(conditions() , axis=1)
    #print(result)
    sns.regplot(result[result.columns[10]],result[result.columns[1]],data=result, fit_reg=False, marker='o',scatter_kws={'facecolors':result['color']})
    plt.show()





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif , chi2

Data = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\EditedData.csv")
Data.fillna('Breed1')
Data.fillna('Breed2')

#Full
predictors = ["Age",'Log(Age)' ,"Gender" , "Fee" , "Type",'Breed1','Breed2','Color2','Color3','IsPure','Color1','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','Quantity','State','Pop','GDPPP','Density','VideoAmt','HasVid','PhotoAmt','Log(Fee)']

#Semi
#predictors = ["Age", "Gender" , "Fee" , "Type",'Breed1','Breed2','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','Quantity','State','VideoAmt','PhotoAmt']

# Perform feature selection
selector = SelectKBest(f_classif, k=7)
selector.fit(Data[predictors], Data["y"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation=30)
plt.title('Best K_Features - fClassic')
plt.show()
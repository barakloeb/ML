import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


Data = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\Data.csv")


Data0 = Data.where(Data["y"] == 0,inplace=False)
Data0 = Data0.dropna(thresh=2)
Data0 = pd.DataFrame(Data0,columns=['y','Age'])
Data0.columns = ['y', 'y=0']



Data1 = Data.where(Data["y"] == 1,inplace=False)
Data1 = Data1.dropna(thresh=2)
Data1 = pd.DataFrame(Data1,columns=['y','Age'])
Data1.columns = ['y', 'y=1']

Data2 = Data.where(Data["y"] == 2,inplace=False)
Data2 = Data2.dropna(thresh=2)
Data2 = pd.DataFrame(Data2,columns=['y','Age'])
Data2.columns = ['y', 'y=2']


print(Data0)
print(Data1)
print(Data2)

p1=sns.kdeplot(Data0['y=0'], shade=True, color="r")
p1=sns.kdeplot(Data1['y=1'], shade=True, color="b")
p1=sns.kdeplot(Data2['y=2'], shade=True, color="y")
plt.legend()
plt.title('Age Distribution By y')
plt.show(p1)




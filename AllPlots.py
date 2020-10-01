import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from scipy.stats import gaussian_kde
import folium



Data = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\Data.csv")
EditedData = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\EditedData.csv")
DataTest = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\Test.csv")
DataBubbles = pd.read_csv("C:\\Users\yonat\Desktop\תעשייה וניהול\שנה ג'\סמס ב\לימוד מכונה\פרויקט\Bubbles.csv")

##Print For Check
print(DataBubbles.head())


#ax = sns.countplot(x="y", data=Data , color="Blue")
#plt.title('y Counting')

#ax = sns.countplot(x="IsPure", data=Data,hue='y' , color="Blue")
#plt.title('IsPure and y')

#ax = sns.countplot(x="Vaccinated", data=Data,hue='y' , color="Blue")
#plt.title('Vaccinated and y')

#ax = sns.countplot(x="Type", data=DataTest , color="Purple")
#plt.title('Type Distribution - Test DataSet')

#ax = sns.countplot(x="Gender", data=Data, color="Blue")
#plt.title('Gender Distribution - Train')

#ax = sns.countplot(x="Gender", data=Data,hue='y' ,color="Blue")
#plt.title('Gender Distribution By y')

#ax = sns.countplot(x="NumberOfColors" ,data =Data ,hue='y',color=(0.2, 0.1, 0.1))
#plt.title('Color1 Distribution - TEST')

#ax = sns.countplot(x="NumberOfColors" ,data =Data ,hue='y',color=(0.2, 0.1, 0.1))
#plt.title('Color1 Distribution - TEST')

##-------Pie Chart
#labels = Data['y'].astype('category').cat.categories.tolist()
#counts = Data['y'].value_counts()
#sizes = [counts[var_cat] for var_cat in labels]
#fig1, ax1 = plt.subplots()
#ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
#ax1.axis('equal')


#ax = sns.distplot( Data["PhotoAmt"] )
#plt.title('PhotoAmt Distribution - Train')

#ax = sns.countplot(x="Color3" ,data =Data,color=(0.2, 0.1, 0.1))
#plt.title('Color1 Distribution - TEST')
#Coorelation

#corrr=np.corrcoef(EditedData[:,0],EditedData[:,1],EditedData[:,2],EditedData[:,3,],
#EditedData[:,4])
# marriage ~balance

#xy = np.vstack([EditedData["Color2"],EditedData["Gender"]])
#z = gaussian_kde(xy)(xy)
#fig, ax = plt.subplots()
#plt.hist2d(EditedData["Dewormed"], EditedData["Vaccinated"], (50, 50), cmap=plt.cm.jet)
#plt.colorbar()
#plt.xlabel("Color3")
#plt.ylabel("Gender")
#plt.title("Point observations and Reggasion")
#plt.show()


#plt.scatter(DataBubbles['Dewormed'], DataBubbles['Vaccinated'], s=DataBubbles['Count'], c="blue", alpha=0.4, linewidth=2)
#plt.xlabel("Dewormed")
#plt.ylabel("Vaccinated")
#plt.title("BubblePlot By Count")
#plt.show()


##Drawing a Map and save it as an HTML Document
#m = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=2)
#for i in range(0, len(DataBubbles)):
#    folium.Circle(
#        location=[int(DataBubbles.iloc[i]['Lat']), int(DataBubbles.iloc[i]['Long'])],
#        popup=DataBubbles.iloc[i]['Name'],
#       radius=int(DataBubbles.iloc[i]['Count'])*10 ,
#        color='crimson',
#        fill=True,
#        fill_color='crimson'
#    ).add_to(m)
#m.save('mymap2.html')
ax = sns.countplot(x="Color3", data=Data,hue='y' ,color="Blue")
plt.title('Color3 Distribution By y')
plt.show()
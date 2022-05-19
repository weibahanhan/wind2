

# import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



path= r'C:\Users\donal\runit\data.xlsx'
df=pd.read_excel(path,sheet_name='input')




df=df.fillna(0)
df=df.set_index('Material')
x=df.fillna(0)
transform = preprocessing.StandardScaler()
transform.fit(x)
x = transform.transform(x)



wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss, linewidth=4, markersize=12,color = 'blue')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()




k=5
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=800, n_init=10, random_state=0)
y_pred = kmeans.fit_predict(x)

df['label']=y_pred
df.to_excel(r'C:\Users\donal\runit\output.xlsx',sheet_name='output1')
df.head()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data pre-processing
df = pd.read_csv("Live.csv")
pd.set_option('display.max_columns', None)
df.head()
# I'm not interested in status_id and status_published columns - bye bye
df = df.drop(["status_id", "status_published"], axis=1)
# Columns "Column x" should be removed.
df = df.drop(["Column1", "Column2", "Column3", "Column4"], axis=1)
df = pd.get_dummies(df, columns=["status_type"], prefix="type")
df.shape
df.describe()
df.info()

lst = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(df)
    lst.append(kmeans.inertia_)
plt.plot(range(1, 8), lst)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()

# It seems that optimal number of clusters is 3
KMeans_model = KMeans(n_clusters=3).fit(df)
KMeans_model.cluster_centers_
df["cluster"] = KMeans_model.predict(df)
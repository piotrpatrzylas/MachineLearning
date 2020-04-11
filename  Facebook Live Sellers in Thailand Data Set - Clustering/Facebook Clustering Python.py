import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Data pre-processing
df = pd.read_csv("Live.csv")
pd.set_option('display.max_columns', None)
df.head()
# I'm not interested in status_id and status_published columns - bye bye
df = df.drop(["status_id", "status_published"], axis=1)
# Columns "Column x" should be removed.
df = df.drop(["Column1", "Column2", "Column3", "Column4"], axis=1)
# Convert status_type to integers, 0 = status, 1 = photo, 2 = status 3 = video
df["status_type_n"] = pd.Categorical(df["status_type"]).codes
df = df.drop("status_type", axis=1)
df.shape
df.describe()
df.info()

lst = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 0)
    kmeans.fit(df)
    lst.append(kmeans.inertia_)
plt.plot(range(1, 8), lst)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
plt.show()
# It seems that optimal number of clusters is 3


X = df
y = df["status_type_n"]
cols = X.columns
ms = MinMaxScaler()
df = ms.fit_transform(df)
df = pd.DataFrame(df, columns=cols)

# Let's check accuracy for number of clusters in range 2 - 5
accuracy = {}
for i in range(2, 6):
    KMeans_model = KMeans(n_clusters=i, random_state=0).fit(df)
    acc_val = sum(y == KMeans_model.labels_)
    accuracy[i] = acc_val
for i in range(4):
    print(f"Model with {list(accuracy.keys())[i]} clusters classified correctly {list(accuracy.values())[i]}"
          f" observations")

KMeans_model_final = KMeans(n_clusters=4, random_state=0).fit(df)
predict = KMeans_model_final.predict(df)
df['cluster'] = predict
# Multidimensional data plot, not great plot, would require some reduction of dimensions
pd.plotting.parallel_coordinates(df, "cluster")

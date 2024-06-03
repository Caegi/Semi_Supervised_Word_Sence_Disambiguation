#%%
from data_preparation import get_data
import pandas as pd
import seaborn as sns

df = get_data()

print(df)

# counts = pd.DataFrame(df["lemma"].value_counts())
# print(counts)
# ax = sns.countplot(counts, x="lemma")
# ax.set(xlabel='Number of examples', ylabel='Count')



#%%

from sklearn.metrics.cluster import contingency_matrix
from seaborn import heatmap
from classes.KMeans import kmeans
from data_preparation import get_data
import matplotlib.pyplot as plt
import numpy as np

df = get_data()

verb = "traduire"

verb_df = df[df['lemma'] == verb].reset_index()
k = len(verb_df['word_sense'].unique())

my_kmeans = kmeans(verb_df, k)
my_kmeans.fit()
y = verb_df["sense_id"]
y_1 = my_kmeans.get_dataframe()["cluster"]

M1 = contingency_matrix(y, y_1)
ax = heatmap(M1, annot=True, fmt="d")
ax.set_xlabel("Clusters")
ax.set_ylabel("Original Classes")
ax.set_title("Contingency Matrix y vs y1")
plt.show()

# P = M1.max(axis=0)/M1.sum(axis=0)

# for i in range(len(P)):
#   print(f"Cluster {i} purity - {P[i].round(2)}")

# np.sum(np.amax(M1, axis=0)) / np.sum(M1)



#%%

from sklearn.metrics.cluster import contingency_matrix
from seaborn import heatmap
from classes.kmeans import Kmeans
from classes.data_preparation import get_data
import matplotlib.pyplot as plt
import numpy as np

df = get_data()

verb = "traduire"

verb_df = df[df['lemma'] == verb].reset_index()
k = len(verb_df['word_sense'].unique())

my_kmeans = Kmeans(verb_df, k)
my_kmeans.fit()
y = verb_df["sense_id"]
y_1 = my_kmeans.get_dataframe()["cluster"]

m1 = contingency_matrix(y, y_1)
# ax = heatmap(M1, annot=True, fmt="d")
# ax.set_xlabel("Clusters")
# ax.set_ylabel("Original Classes")
# ax.set_title("Contingency Matrix y vs y1")
# plt.show()

# P = M1.max(axis=0)/M1.sum(axis=0)

# for i in range(len(P)):
#   print(f"Cluster {i} purity - {P[i].round(2)}")

# print(np.sum(np.amax(m1, axis=0)) / np.sum(m1)) # type: ignore
# print(m1.max(axis=0)/m1.sum(axis=0))
max_indices = np.argmax(m1, axis=0) # type: ignore
print(max_indices)

new_cluster = []
for c in y_1.to_list():
    new_cluster.append(max_indices[c])

#print(verb_df) 

verb_df["cluster"] = new_cluster

print(verb_df[["sense_id", "cluster"]])

#%%

#from classes.data_preparation import get_data
import pandas as pd
import numpy as np

data = pd.read_json("fse_data_w_embeddings.json")#, dtype={'ft_embeddings': np.float32})
#data['ft_embeddings'] = data["ft_embeddings"].astype(np.float32)
print(type(data["ft_embeddings"][0]))


# %%

from classes.data_preparation import get_data

df = get_data()

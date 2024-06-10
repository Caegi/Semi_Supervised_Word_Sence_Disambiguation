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

data = pd.read_csv("fse_data_w_embeddings.csv")
data.astype({'ft_embeddings': 'numpy.dtype'})

#%%

verb_df = data[data['lemma'] == "aboutir"].reset_index()
print(verb_df)
verb_df=verb_df[verb_df.groupby('sense_id').sense_id.transform(len)>1]
print(verb_df)

#%%
from classes.classification import get_x_y
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

X_total, y_total = get_x_y(verb_df, "ft")

sss=StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
train_split = [train for train, test in sss.split(X_total, y_total)]
print(X_total[train_split])
train_split = train_split[0].tolist()
print(train_split)
print([y_total[i] for i in train_split])

# %%

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
from classes.kmeans import Kmeans
from data_preparation import get_data
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

from classes.classification import decrease_training_examples
from data_preparation import get_data

df = get_data()
scores = decrease_training_examples(df)

nb_examples = [50, 45, 40, 35, 30, 25, 20, 15, 10]

print(scores)

#%%

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df_decrease = pd.DataFrame({"Number of Examples": nb_examples, "F-Score": scores})
sns.barplot(df_decrease, x= "Number of Examples", y="F-Score")
plt.gca().invert_xaxis()

#%%

from data_preparation import get_data

data = get_data()

#%%
import pandas as pd

data = pd.read_csv("fse_data_w_embeddings")

print(data.head())



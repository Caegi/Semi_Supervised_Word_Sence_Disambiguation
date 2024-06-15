#%%
from classification import decrease_training_examples
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_json("src/fse_data_w_embeddings.json")
#%%

# count number of examples per lemma

counts = pd.DataFrame(df["lemma"].value_counts())
# print(counts)
ax = sns.countplot(counts, x="lemma")
ax.set(xlabel='Number of examples', ylabel='Count')
ax.bar_label(ax.containers[0])

#%%
# visualize classification with kmeans
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

nb_examples = [50, 45, 40, 35, 30, 25, 20, 15, 10]
results = [0.729, 0.722, 0.713, 0.708, 0.709, 0.685, 0.678, 0.685, 0.576]
kmeans = [0.689] * 9

df = pd.DataFrame({"examples": nb_examples, "classif": results, "kmeans": kmeans})


sns.set_theme(style='dark',rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

fig, ax = plt.subplots(figsize=(12,6))

l = sns.lineplot(data = df['kmeans'], marker='o', sort = False, ax=ax, label='K-Means', color='#00CED1')


c = sns.barplot(data = df, x='examples', y='classif', alpha=0.5, ax=ax, color="black", label="Classification")

c.set_xlabel("Number of Examples")
c.set_ylabel("F-Score")


plt.gca().invert_xaxis()
# for i in range(9):
#     ax.bar_label(ax.containers[i])

#%%
# visualize kmeans with classification
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

nb_examples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
results = [0.6946, 0.7138, 0.7274, 0.7414, 0.7387, 0.7501, 0.7548, 0.7637, 0.7679, 0.7666, 0.7704, 0.7739, 0.7803, 0.7787, 0.7808]
           
classif = [0.729] * 15

df = pd.DataFrame({"examples": nb_examples, "kmeans": results, "classif": classif})


sns.set_theme(style='dark',rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

fig, ax = plt.subplots(figsize=(12,6))

l = sns.lineplot(data = df['classif'], marker='o', sort = False, ax=ax,  label='Classification', color='#00CED1')


c = sns.barplot(data = df, x='examples', y='kmeans', alpha=0.5, ax=ax, color="black", label="K-Means")

c.set_xlabel("Number of Examples")
c.set_ylabel("F-Score")


#plt.gca().invert_xaxis()
# for i in range(15):
#     ax.bar_label(ax.containers[i])


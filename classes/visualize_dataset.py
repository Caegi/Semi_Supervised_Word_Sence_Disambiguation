#%%
from classification import decrease_training_examples
from data_preparation import get_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = get_data()
#%%

# count number of examples per lemma

counts = pd.DataFrame(df["lemma"].value_counts())
# print(counts)
ax = sns.countplot(counts, x="lemma")
ax.set(xlabel='Number of examples', ylabel='Count')
ax.bar_label(ax.containers[0])

#%%
# visualize f-score for decreasing examples

scores = decrease_training_examples(df)

nb_examples = [50, 45, 40, 35, 30, 25, 20, 15, 10]

print(scores)

#%%

df_decrease = pd.DataFrame({"Number of Examples": nb_examples, "F-Score": scores})
ax = sns.barplot(df_decrease, x= "Number of Examples", y="F-Score")
plt.gca().invert_xaxis()
ax.bar_label(ax.containers[0])
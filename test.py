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







#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df = pd.read_csv('../fse_data.csv')
list_lemma = list(set(df.lemma.tolist()))



#%%
# get table with nb examples per word sense
for lemma in list_lemma:
  print("lemma:",lemma)
  df_lemma = df[(df['lemma'] == lemma)].reset_index()
  print(df_lemma["word_sense"].value_counts())
  print("")

#%%

# try to visualize, not very clear

df["word_sense"] = df["word_sense"].str.split("_").str[1]
cols = 2
rows = len(list_lemma)//2
fig, axes = plt.subplots(rows, cols, figsize = (10, 30)) # type: ignore
axes = axes.flat

for i in range(len(list_lemma)):
  df_lemma = df[(df['lemma'] == list_lemma[i])].reset_index()
  #labels = df_lemma['word_sense']
  plot = sns.countplot(df_lemma, x="word_sense", ax = axes[i])
#   plot.set_xticklabels(labels, rotation=20)
#   plot.set_yticklabels(labels, rotation=20)
  axes[i].set_title(f'{list_lemma[i]}')


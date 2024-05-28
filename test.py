#%%
import pandas as pd
from classes.Classification import id2sense



df = pd.read_csv('fse_data.csv')
id_2_sense, sense_2_id = id2sense(df)
print(sense_2_id)   

#print(df[df['lemma'] == "conclure"]["word_sense"].to_string())

df["sense_id"] = sense_2_id[df.get("lemma").][df.get("word_sense")]

print(df["sense_id"])

# test_list = df["word_sense"].str.split("_").str[1]

# print(test_list[2260])

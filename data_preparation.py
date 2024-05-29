import pandas as pd

# id_to_sense: dictionary --> key = lemma, value = list of senses
# sense_to_id: dictionary --> key = lemma, values = dictionary --> key = sense, value = index of sense in list id_to_sense[lemma]

df = pd.read_csv('fse_data.csv')

senses = set(df.word_sense.tolist())

id_to_sense = {}
sense_to_id = {}

# compute id_to_sense
for s in senses:
  lemma = s.split("_")[-1]
  if lemma not in id_to_sense:
    id_to_sense[lemma] = []

  id_to_sense[lemma].append(s)


# compute sense_to_id
for l in id_to_sense:
  if l not in sense_to_id:
    sense_to_id[l] = {}
  for count, s in enumerate(id_to_sense[l]):
    sense_to_id[l][s] = count

def lookup(row):
    
    return sense_to_id[row['lemma']][row['word_sense']]

def get_data():
  
  df['sense_id'] = df.apply(lookup, axis=1)

  return df
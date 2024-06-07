import pandas as pd
from gensim.models import KeyedVectors
from spacy.lang.fr import French
import fasttext

# initialize tokenizer
nlp = French()

# load data
df = pd.read_csv('./fse_data.csv')

# load static embeddings
w2v = KeyedVectors.load_word2vec_format("../frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", binary=True, unicode_errors="ignore")
ft = fasttext.load_model('../cc.fr.300.bin') 

# id_to_sense: dictionary --> key = lemma, value = list of senses
# sense_to_id: dictionary --> key = lemma, values = dictionary --> key = sense, value = index of sense in list id_to_sense[lemma]

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



def get_sense_id(row):
    '''returns sense id for word sense'''
    
    return sense_to_id[row['lemma']][row['word_sense']]


def tokenize(row):
    '''returns tokenized sentence'''
    
    return nlp(row.sentence.lower()).text


def get_data():
  '''returns the prepared data for the tasks'''
  
  # add sense id to every sense
  df['sense_id'] = df.apply(get_sense_id, axis=1)

  # tokenize sentences
  df['sentence'] = df.apply(tokenize, axis=1)

  # add column with word2vec embeddings
  w2v_embed_column = [w2v.get_mean_vector(row['sentence']) for _, row in df.iterrows()]
  df['w2v_embeddings'] = w2v_embed_column

  # add column with fasttext embeddings
  ft_embed_column = [ft.get_sentence_vector(row['sentence']) for _, row in df.iterrows()]
  df['ft_embeddings'] = ft_embed_column

  return df

# export data frame to csv file
#get_data().to_csv('./fse_data_w_embeddings.csv')

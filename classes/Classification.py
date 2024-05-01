"""## Classification"""

"""Import"""
from spacy.lang.fr import French
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import fasttext
from gensim.models import KeyedVectors
from DataExtraction import *
import gzip
import shutil


# get the pre-trained fast text embeddings for French
input_file = 'content/cc.fr.300.bin.gz'
output_file = 'content/cc.fr.300.bin'

# unzip gzip file
with gzip.open(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

"""Pre-trained Embeddings"""
# load the different embeddings

# fasttext
ft = fasttext.load_model('../../cc.fr.300.bin') 
# word2vec
w2v = KeyedVectors.load_word2vec_format("../../frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", binary=True, unicode_errors="ignore")
# glove ???

"""Data Preparation"""
# prepare tokenizer
nlp = French()
tokenizer = nlp.tokenizer

# id_to_sense: dictionary --> key = lemma, value = list of senses
# sense_to_id: dictionary --> key = lemma, values = dictionary --> key = sense, value = index of sense in list id_to_sense[lemma]


# extractor = DataExtraction()
# df = extractor.load_saved_file()
df = pd.read_csv('../fse_data.csv')
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



print("id_to_sense", id_to_sense)
print("sense_to_id", sense_to_id)


def get_x_y(data, lemma, embedding_size, embedding_type):

  # initialize X and y as zero array and empty list
  X = np.zeros((len(data), embedding_size))
  y = []

  # iterate through the sentences and update X and y
  for count, doc in enumerate(tokenizer.pipe(data.sentence.tolist())): # type: ignore
    if embedding_type == "ft":
      X[count, :] = ft.get_sentence_vector(doc.text)
    elif embedding_type == "w2v":
      X[count, :] = w2v.get_mean_vector(doc.text.lower().split(" "), ignore_missing = True)

    y.append(sense_to_id[lemma][data.word_sense[count]])

  return X, y



# def get_x_y_glove(data, lemma):

#   X = np.zeros((len(data), 200))
#   y = []

#   for count, sentence in enumerate(tokenizer.pipe(data.sentence.tolist())): #type:ignore

#     X[count, :] = w2v.get_mean_vector(sentence.text.lower().split(" "), ignore_missing = True)

#     y.append(sense_to_id[lemma][data.word_sense[count]])

#   return (X, y)


"""Classification functions"""
# Cross validation Classification

def cv_classification(X, y, lemma):

  # train classifier
  if len(set(y)) > 1:
    cv_classif = LogisticRegression(solver="liblinear").fit(X, y)
    scores = cross_val_score(cv_classif, X, y, cv=2)
    return round(scores.mean(),2)

  else:
    return float("nan")

# Traditional Classification
def traditional_classification(X_train,X_test,y_train, y_test, lemma):

  # train classifier
  if len(set(y_train)) > 1:
    classifier = LogisticRegression(solver = "liblinear").fit(X_train, y_train)
    return round(classifier.score(X_test, y_test), 2)

  else:
    return float("nan")


"""Compare functions"""

# compare 70/30 splitting with cross validation (fasttext embeddings)
def compare_split_method():

  trad_classif = []
  cv = []

  for lemma in id_to_sense.keys():

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()
      
    # split data into test and train
    X, y = get_x_y(data, lemma, 300, "ft")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) # shuffles data by default before splitting

    # train classifiers with different methods
    trad_classif.append(traditional_classification(X_train, X_test, y_train, y_test, lemma))
    cv.append(cv_classification(X, y, lemma))

  return (trad_classif, cv)


# compare fasttext embeddings with word2vec embeddings (cross validation)
def compare_embeddings():

  fast_emb = []
  w2v_emb = []

  for lemma in id_to_sense.keys():

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()

    # classification with fasttext
    X_fast, y_fast = get_x_y(data, lemma, 300, "ft")
    fast_emb.append(cv_classification(X_fast, y_fast, lemma))

    # classification with word2vec
    X_w2v, y_w2v = get_x_y(data, lemma, 200, "w2v")
    w2v_emb.append(cv_classification(X_w2v, y_w2v, lemma))

  return (fast_emb, w2v_emb)

fast_emb, w2v_emb = compare_embeddings()

trad_classification, cv = compare_split_method()

print("Embedding results")
print("Fasttext:", fast_emb)
print("Word2Vec:", w2v_emb)

print("\nClassification results")
print("70/30 split:", trad_classification)
print("cross valid:", cv)
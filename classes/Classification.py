"""## Classification"""

"""Import"""
from spacy.lang.fr import French
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import pandas as pd
import fasttext
from gensim.models import KeyedVectors

"""Pre-trained Embeddings"""
# load the different embeddings

# fasttext
#ft = fasttext.load_model('../cc.fr.300.bin') 
# word2vec
w2v = KeyedVectors.load_word2vec_format("../frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin", binary=True, unicode_errors="ignore")
# glove ???


"""Data Preparation"""
# prepare tokenizer
nlp = French()
tokenizer = nlp.tokenizer

# id_to_sense: dictionary --> key = lemma, value = list of senses
# sense_to_id: dictionary --> key = lemma, values = dictionary --> key = sense, value = index of sense in list id_to_sense[lemma]

df = pd.read_csv('fse_data.csv')

def id2sense(df):
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

  return (id_to_sense, sense_to_id)



# print("id_to_sense", id_to_sense)
# print("sense_to_id", sense_to_id)

def get_x_y_w2v(data):

  # initialize X and y as zero array and empty list
  X = np.zeros((len(data), 200))
  y = []

  for count, doc in enumerate(tokenizer.pipe(data.sentence.tolist())): # type: ignore
    X[count, :] = w2v.get_mean_vector(doc.text.lower().split(" "), ignore_missing = True)

    y.append(data["word_sense"])

  return X, y


# def get_x_y(data, lemma, embedding_size, embedding_type):

#   # initialize X and y as zero array and empty list
#   X = np.zeros((len(data), embedding_size))
#   y = []

#   # iterate through the sentences and update X and y
#   for count, doc in enumerate(tokenizer.pipe(data.sentence.tolist())): # type: ignore
#     if embedding_type == "ft":
#       X[count, :] = ft.get_sentence_vector(doc.text)
#     elif embedding_type == "w2v":
#       X[count, :] = w2v.get_mean_vector(doc.text.lower().split(" "), ignore_missing = True)

#     y.append(sense_to_id[lemma][data.word_sense[count]])

#   return X, y

"""Classification functions"""
# Cross validation Classification
from sklearn.model_selection import StratifiedKFold
def cv_classification(df, lemma, nb_splits):

  X, y = get_x_y_w2v(df)
  
  sk_fold=StratifiedKFold(n_splits=nb_splits)
  # train classifier
  if len(set(y)) > 1:
    cv_classif = LogisticRegression(solver="liblinear")
    mod_score4=cross_val_score(cv_classif,X,y,cv=sk_fold, scoring="f1_micro")
    return round(mod_score4.mean(),2)

  else:
    return float(0)

# Traditional Classification
def traditional_classification(X_train,X_test,y_train, y_test):

  # train classifier
  if len(set(y_train)) > 1:
    classifier = LogisticRegression(solver = "liblinear").fit(X_train, y_train)
    
    pred = classifier.predict(X_test)
    score = f1_score(y_test, pred, average="micro")
    return round(float(score), 2)

  else:
    return float(0)


"""Compare functions"""

# compare 70/30 splitting with cross validation (fasttext embeddings)
# def compare_split_method():

#   trad_classif = []
#   cv = []

#   for lemma in id_to_sense.keys():

#     # compute data frame with only one lemma
#     data = df[(df['lemma'] == lemma)].reset_index()
      
#     # split data into test and train
#     X, y = get_x_y(data, lemma, 200, "w2v")
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) # shuffles data by default before splitting

#     # train classifiers with different methods
#     trad_classif.append(traditional_classification(X_train, X_test, y_train, y_test))
#     cv.append(cv_classification(X, y, 5))

#   return (trad_classif, cv)


# compare fasttext embeddings with word2vec embeddings (cross validation)
# def compare_embeddings():

#   fast_emb = []
#   w2v_emb = []

#   for lemma in id_to_sense.keys():

#     # compute data frame with only one lemma
#     data = df[(df['lemma'] == lemma)].reset_index()

#     # classification with fasttext
#     X_fast, y_fast = get_x_y(data, lemma, 300, "ft")
#     fast_emb.append(cv_classification(X_fast, y_fast, 5))

#     # classification with word2vec
#     X_w2v, y_w2v = get_x_y(data, lemma, 200, "w2v")
#     w2v_emb.append(cv_classification(X_w2v, y_w2v, 5))

# return (fast_emb, w2v_emb)


# computes the mean socre over all lemma
def get_best(x1, x2, names):

  count = [round(np.mean(np.asarray(x1)), 3), round(np.mean(np.asarray(x2)),3)]
  
  print(f"Mean f-score over all lemma for: \n{names[0]}: {count[0]} \n{names[1]}: {count[1]}")

# gives mean score over all lemma for decreasing number of examples (always around 10 test examples)
# def decrease_training_examples():
#   scores = []
#   nb_examples = 50
#   t_size=0.0

#   while nb_examples >= 20: 

#     scores_per_lemma = []
    
#     for lemma in id_to_sense.keys():

#       # compute data frame with only one lemma
#       data = df[(df['lemma'] == lemma)].reset_index()

#       X_total, y = get_x_y_w2v(data, lemma)
      
#       if nb_examples < 50:
#         X, X_left_out, y, y_left_out = train_test_split(X_total, y, test_size=t_size, random_state=5)
#       else: X = X_total

#       print(X.shape)
#       scores_per_lemma.append(cv_classification(X, y, nb_examples//10))

#     scores_np = np.nan_to_num(np.asarray(scores_per_lemma))    
#     scores.append(round(np.mean(scores_np),3))
#     t_size += 0.1
#     nb_examples -= 5
#   print(scores)

# decrease_training_examples()

# fast_emb, w2v_emb = compare_embeddings()

# trad_classification, cv = compare_split_method()
# print(trad_classification)
# print(cv)
# print("Embedding results\n")
# get_best(fast_emb, w2v_emb, ["Fasttext", "Word2Vec"])

# print("\nClassification results\n")
# get_best(trad_classification, cv, ["70/30 split", "Cross Validation"])

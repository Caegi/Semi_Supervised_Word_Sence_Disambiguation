"""## Classification"""

"""Import"""
from spacy.lang.fr import French
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from spacy.tokenizer import Tokenizer

"""Pre-trained Embeddings"""
# load the different embeddings

# fasttext

# word2vec

# glove ???


"""Data Preparation"""
# prepare tokenizer
nlp = French()
tokenizer : Tokenizer = nlp.tokenizer

W2V_EMBED_SIZE = 200 # word2vec embedding size = 200

def get_x_y_w2v(data):
  """
  returns sentence embeddings and gold classes for the given sentence data
  """
  # initialize X and y as zero array and empty list
  X = np.zeros((len(data), W2V_EMBED_SIZE)) 
  y = []

  tokenized_sentences = tokenizer.pipe(data.sentence.tolist()) 

  for count, doc in enumerate(tokenized_sentences): 
    
    # ignore_missing avoids error for missing words in w2v vocab
    X[count, :] = w2v.get_mean_vector(doc.text.lower().split(" "), ignore_missing = True)

    y.append(data["sense_id"].loc[count])

  return X, y


def get_x_y(data, embedding_size, embedding_type):

  # initialize X and y as zero array and empty list
  X = np.zeros((len(data), embedding_size))
  y = []

  # iterate through the sentences and update X and y
  for count, doc in enumerate(tokenizer.pipe(data.sentence.tolist())):
    if embedding_type == "ft":
      X[count, :] = ft.get_sentence_vector(doc.text)
    elif embedding_type == "w2v":
      X[count, :] = w2v.get_mean_vector(doc.text.lower().split(" "), ignore_missing = True)

    y.append(data["sense_id"].loc[count])

  return X, y

"""Classification functions"""
# Cross validation Classification

def cv_classification(X, y, nb_splits):
  """performs cross validation classification and outputs the average over all folds"""
  
  sk_fold=StratifiedKFold(n_splits=nb_splits)

  
  if len(set(y)) > 1:
    cv_classif = LogisticRegression(solver="liblinear")
    scores=cross_val_score(cv_classif,X,y,cv=sk_fold, scoring="f1_micro")
    return scores.mean()

  else:
    return 0

# Traditional Classification
def traditional_classification(X_train,X_test,y_train, y_test):

  # train classifier
  if len(set(y_train)) > 1:
    classifier = LogisticRegression(solver = "liblinear").fit(X_train, y_train)
    
    pred = classifier.predict(X_test)
    score = f1_score(y_test, pred, average="micro")
    return round(float(score), 2)

  else:
    return 0


"""Compare functions"""

# compare 70/30 splitting with cross validation (fasttext embeddings)
def compare_split_method(df):

  trad_classif = []
  cv = []
  list_of_verbs = df['lemma'].unique()

  for lemma in list_of_verbs:

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()
      
    # split data into test and train
    X, y = get_x_y(data, lemma, 200, "w2v")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) # shuffles data by default before splitting

    # train classifiers with different methods
    trad_classif.append(traditional_classification(X_train, X_test, y_train, y_test))
    cv.append(cv_classification(X, y, 5))

  return (trad_classif, cv)


# compare fasttext embeddings with word2vec embeddings (cross validation)
def compare_embeddings(df):

  fast_emb = []
  w2v_emb = []

  list_of_verbs = df['lemma'].unique()

  for lemma in list_of_verbs:

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()

    # classification with fasttext
    X_fast, y_fast = get_x_y(data, lemma, 300, "ft")
    fast_emb.append(cv_classification(X_fast, y_fast, 5))

    # classification with word2vec
    X_w2v, y_w2v = get_x_y(data, lemma, 200, "w2v")
    w2v_emb.append(cv_classification(X_w2v, y_w2v, 5))

  return (fast_emb, w2v_emb)


# computes the mean socre over all lemma
def get_best(x1, x2, names):

  count = [round(np.mean(np.asarray(x1)), 3), round(np.mean(np.asarray(x2)),3)]
  
  print(f"Mean f-score over all lemma for: \n{names[0]}: {count[0]} \n{names[1]}: {count[1]}")

# gives mean score over all lemma for decreasing number of examples (always around 10 test examples)
def decrease_training_examples(df):
  scores = []
  nb_examples = 50
  t_size=0.0

  while nb_examples >= 10: 

    scores_per_lemma = []
    list_of_verbs = df['lemma'].unique()

    for lemma in list_of_verbs:

      # compute data frame with only one lemma
      data = df[(df['lemma'] == lemma)].reset_index()

      # get embeddings and gold labels
      X_total, y = get_x_y_w2v(data)
      
      # exception for case where all training examples are used
      # decrease training examples with train_test_split and only use test set
      if nb_examples < 50:
        X, X_left_out, y, y_left_out = train_test_split(X_total, y, test_size=t_size, random_state=5)
      else: X = X_total
      
      # exception for case when split is smaller than 2
      # always size test set = 10 except for 15 and 10 examples (test set = 7 and 5 examples)
      if nb_examples // 10 > 1:
        split = nb_examples // 10
      else : split = 2

      scores_per_lemma.append(cv_classification(X, y, split))
    
    scores_np = np.nan_to_num(np.asarray(scores_per_lemma))    
    scores.append(round(np.mean(scores_np),3))
    t_size += 0.1
    nb_examples -= 5
    
  return scores

# decrease_training_examples()

# fast_emb, w2v_emb = compare_embeddings()

# trad_classification, cv = compare_split_method()
# print(trad_classification)
# print(cv)
# print("Embedding results\n")
# get_best(fast_emb, w2v_emb, ["Fasttext", "Word2Vec"])

# print("\nClassification results\n")
# get_best(trad_classification, cv, ["70/30 split", "Cross Validation"])

"""## Classification"""

"""Import"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning

"""get X and y for classification"""

def get_x_y_w2v(data):
  """
  returns word2vec sentence embeddings and gold classes for the given sentence data
  """
  # initialize X and y as zero array and empty list
  X = np.asarray(data['w2v_embeddings'].to_list()) 
  y = data['sense_id'].to_list()

  return X, y


def get_x_y(data, embedding_type):

  """
  return w2v or fasttext sentence embeddings and gold classes for given data
  """

  if embedding_type == "ft":
    X = np.asarray(data['ft_embeddings'].to_list()) 
  elif embedding_type == "w2v":
    X = np.asarray(data['w2v_embeddings'].to_list()) 
  elif embedding_type == "tf_idf":
    X = np.asarray(data['tf_idf_embeddings'].to_list())
    X = np.vstack(X)

  y = data['sense_id'].to_list()

  return X, y

"""Classification functions"""
# Cross validation Classification

def cv_classification(X, y, nb_splits):
  """performs cross validation classification and outputs the average over all folds"""
  
  sk_fold=StratifiedKFold(n_splits=nb_splits)

  warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")

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
  """
  returns the mean f-score for each lemma for cross validation and 70/30 split with w2v embeddings
  """

  trad_classif = []
  cv = []
  list_of_verbs = df['lemma'].unique()

  for lemma in list_of_verbs:

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()
      
    # split data into test and train
    X, y = get_x_y(data, "w2v")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) # shuffles data by default before splitting

    # train classifiers with different methods
    trad_classif.append(traditional_classification(X_train, X_test, y_train, y_test))
    cv.append(cv_classification(X, y, 5))

  return (trad_classif, cv)


# compare fasttext embeddings with word2vec embeddings (cross validation)
def compare_embeddings(df):
  """
  returns mean score for each lemma for fasttext embeddings and word2vec embeddings (with cross validation)
  """

  fast_emb = []
  w2v_emb = []
  tf_idf_emb = []

  list_of_verbs = df['lemma'].unique()

  for lemma in list_of_verbs:

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()

    # classification with fasttext
    X_fast, y_fast = get_x_y(data, "ft")
    fast_emb.append(cv_classification(X_fast, y_fast, 5))

    # classification with word2vec
    X_w2v, y_w2v = get_x_y(data, "w2v")
    w2v_emb.append(cv_classification(X_w2v, y_w2v, 5))

    # classification with tf_idf
    X_tf_idf, y_tf_idf = get_x_y(data, "tf_idf")
    tf_idf_emb.append(cv_classification(X_tf_idf, y_tf_idf, 5))

  return (fast_emb, w2v_emb, tf_idf_emb)


# computes the mean socre over all lemma
def get_best(x1, x2, x3, names):
  """prints the scores for better understanding"""

  count = [round(np.mean(np.asarray(x1)), 3), round(np.mean(np.asarray(x2)),3), round(np.mean(np.asarray(x3)),3)]
  
  print(f"Mean f-score over all lemma for: \n{names[0]}: {count[0]} \n{names[1]}: {count[1]} \n{names[2]}: {count[2]}")


# gives mean score over all lemma for decreasing number of examples (always around 10 test examples)
def decrease_training_examples(df):
  """trains the classifier with decreasing training examples. Returns array with scores."""
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


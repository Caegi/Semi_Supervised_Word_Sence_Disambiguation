"""## Classification"""

"""Import"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
<<<<<<< HEAD
=======
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedShuffleSplit
>>>>>>> 34ffb0d (merged main and master)

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
<<<<<<< HEAD
  elif embedding_type == "w2v":
    X = np.asarray(data['w2v_embeddings'].to_list()) 
=======

  elif embedding_type == "w2v":
    X = np.asarray(data['w2v_embeddings'].to_list())

  elif embedding_type == "glov":
    X = np.asarray(data['glove_embeddings'].to_list()) 

  elif embedding_type == "tf_idf":
    X = np.asarray(data['tf_idf_embeddings'].to_list())
    X = np.vstack(X)
>>>>>>> 34ffb0d (merged main and master)

  y = data['sense_id'].to_list()

  return X, y

"""Classification functions"""
# Cross validation Classification

def cv_classification(X, y, nb_splits):
  """performs cross validation classification and outputs the average over all folds"""
  
  sk_fold=StratifiedKFold(n_splits=nb_splits)

<<<<<<< HEAD
  
  if len(set(y)) > 1:
    cv_classif = LogisticRegression(solver="liblinear")
=======
  warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")

  if len(set(y)) > 1:
    cv_classif = LogisticRegression(multi_class='multinomial')
>>>>>>> 34ffb0d (merged main and master)
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


<<<<<<< HEAD
"""Compare functions"""
=======
'''
Compare functions for the classification. 
We compared: 
- the evaluation: 70 / 30 split with cross validation (5 folds)
- the static embeddings: FastText, Word2Vec and (self-trained) GloVe
'''
>>>>>>> 34ffb0d (merged main and master)

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

<<<<<<< HEAD
  return (trad_classif, cv)
=======
  return (round(np.mean(np.asarray(trad_classif)),3), round(np.mean(np.asarray(cv)),3))
>>>>>>> 34ffb0d (merged main and master)


# compare fasttext embeddings with word2vec embeddings (cross validation)
def compare_embeddings(df):
  """
  returns mean score for each lemma for fasttext embeddings and word2vec embeddings (with cross validation)
  """

  fast_emb = []
  w2v_emb = []
<<<<<<< HEAD
=======
  glov_emb = []
  tf_idf_emb = []
>>>>>>> 34ffb0d (merged main and master)

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

<<<<<<< HEAD
  return (fast_emb, w2v_emb)


# computes the mean socre over all lemma
def get_best(x1, x2, names):
  """prints the scores for better understanding"""

  count = [round(np.mean(np.asarray(x1)), 3), round(np.mean(np.asarray(x2)),3)]
  
  print(f"Mean f-score over all lemma for: \n{names[0]}: {count[0]} \n{names[1]}: {count[1]}")
=======
    # classification with glove
    X_glov, y_glov = get_x_y(data, "glov")
    glov_emb.append(cv_classification(X_glov, y_glov, 5))

    # # classification with tf_idf
    # X_tf_idf, y_tf_idf = get_x_y(data, "tf_idf")
    # tf_idf_emb.append(cv_classification(X_tf_idf, y_tf_idf, 5))

  return (round(np.mean(np.asarray(fast_emb)),3), round(np.mean(np.asarray(w2v_emb)),3), round(np.mean(np.asarray(glov_emb)),3))


# computes the mean socre over all lemma
def get_best(scores, names):
  """prints the scores for better understanding"""
  
  print(f"Mean f-score over all lemma for:")
  for i in range(len(scores)):
    print(f"{names[i]}: {scores[i]}")
>>>>>>> 34ffb0d (merged main and master)


# gives mean score over all lemma for decreasing number of examples (always around 10 test examples)
def decrease_training_examples(df):
  """trains the classifier with decreasing training examples. Returns array with scores."""
  scores = []
<<<<<<< HEAD
  nb_examples = 50
  t_size=0.0

  while nb_examples >= 10: 
=======
  should_be_nb = 50
  t_size=0.0

  while should_be_nb >= 10: 
>>>>>>> 34ffb0d (merged main and master)

    scores_per_lemma = []
    list_of_verbs = df['lemma'].unique()

    for lemma in list_of_verbs:
<<<<<<< HEAD
=======
      print(f"Lemma: {lemma} with {(1-t_size) * 100}% of examples")
>>>>>>> 34ffb0d (merged main and master)

      # compute data frame with only one lemma
      data = df[(df['lemma'] == lemma)].reset_index()

<<<<<<< HEAD
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

=======
      # make sure every class occurs at least 2 times
      # so that every class occurs in our decreased train set
      #data=data[data.groupby('sense_id').sense_id.transform(len)>1]
      
      # get embeddings and gold labels
      X_total, y_total = get_x_y(data, "ft")
      
      # exception for case where all training examples are used
      # decrease training examples with train_test_split and only use test set
      if should_be_nb < 50:
        X, X_left_out, y, y_left_out = train_test_split(X_total, y_total, test_size=t_size, random_state=5)
        
        # try to get all classes in the decreasing k-fold
        # sss=StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        # train_split = [train for train, test in sss.split(X_total, y_total)]
        # X = X_total[train_split]
        # train_split = train_split[0].tolist()
        # y = [y_total[i] for i in train_split]


      else: 
        X = X_total
        y = y_total

      real_nb = X.shape[0]
      #print(f"Number of senses in total: {len(set(y_total))}\nNumber of senses in smaller set: {len(set(y))}")

      # exception for case when split is smaller than 2
      # always size test set = 10 except for 15 and 10 examples (test set = 7 and 5 examples)

      if real_nb // 10 > 1:
        split = real_nb // 10
      else : split = 2

      print(f"{real_nb} examples.\nK-fold with {split} splits")
      

>>>>>>> 34ffb0d (merged main and master)
      scores_per_lemma.append(cv_classification(X, y, split))
    
    scores_np = np.nan_to_num(np.asarray(scores_per_lemma))    
    scores.append(round(np.mean(scores_np),3))
    t_size += 0.1
<<<<<<< HEAD
    nb_examples -= 5
=======
    should_be_nb -= 5
>>>>>>> 34ffb0d (merged main and master)
    
  return scores


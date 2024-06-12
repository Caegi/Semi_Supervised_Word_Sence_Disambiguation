"""## Classification"""

"""Import"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

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
  elif embedding_type == "glov":
    X = np.asarray(data['glove_embeddings'].to_list())

  y = data['sense_id'].to_list()

  return X, y

"""Classification functions"""
# Cross validation Classification

def cv_classification(X, y, nb_splits):
  """performs cross validation classification and outputs the average over all folds"""
  
  sk_fold=StratifiedKFold(n_splits=nb_splits)

  
  if len(set(y)) > 1: # make sure that the lemma has at least two different sense
    cv_classif = LogisticRegression(multi_class='multinomial')
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


'''
Compare functions for the classification. 
We compared: 
- the evaluation: 70 / 30 split with cross validation (5 folds)
- the static embeddings: FastText, Word2Vec and (self-trained) GloVe
'''

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
    X, y = get_x_y(data, "ft")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) # shuffles data by default before splitting

    # train classifiers with different methods
    trad_classif.append(traditional_classification(X_train, X_test, y_train, y_test))
    cv.append(cv_classification(X, y, 5))

  return (round(np.mean(np.asarray(trad_classif)),3), round(np.mean(np.asarray(cv)),3))


# compare fasttext embeddings with word2vec embeddings (cross validation)
def compare_embeddings(df):
  """
  returns mean score for each lemma for fasttext embeddings and word2vec embeddings (with cross validation)
  """

  fast_emb = []
  w2v_emb = []
  glov_emb = []

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

    # classification with glove
    X_glov, y_glov = get_x_y(data, "glov")
    glov_emb.append(cv_classification(X_glov, y_glov, 5))


  return (round(np.mean(np.asarray(fast_emb)),3), round(np.mean(np.asarray(w2v_emb)),3), round(np.mean(np.asarray(glov_emb)),3))


# computes the mean socre over all lemma
def get_best(scores, names):
  """prints the scores for better understanding"""
  
  print(f"Mean f-score over all lemma for:")
  for i in range(len(scores)):
    print(f"{names[i]}: {scores[i]}")


# gives mean score over all lemma for decreasing number of examples (always around 10 test examples)
def decrease_training_examples(df):
  """trains the classifier with decreasing training examples. Returns array with scores."""
  scores = []
  should_be_nb = 50
  t_size=0.0

  while should_be_nb >= 10: 

    scores_per_lemma = []
    list_of_verbs = df['lemma'].unique()

    for lemma in list_of_verbs:
      print(f"Lemma: {lemma} with {(1-t_size) * 100}% of examples")

      # compute data frame with only one lemma
      data = df[(df['lemma'] == lemma)].reset_index()

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
      

      scores_per_lemma.append(cv_classification(X, y, split))
    
    scores_np = np.nan_to_num(np.asarray(scores_per_lemma))    
    scores.append(round(np.mean(scores_np),3))
    t_size += 0.1
    should_be_nb -= 5
    
  return scores


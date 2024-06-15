"""## Classification"""

"""Import"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from joblib import dump

"""get X and y for classification"""

def get_x_y(data, embedding_type):

  """
  return w2v or fasttext sentence embeddings and gold classes for given data
  """

  # FastText
  if embedding_type == "ft":
    X = np.asarray(data['ft_embeddings'].to_list()) 

  # Word2Vec
  elif embedding_type == "w2v":
    X = np.asarray(data['w2v_embeddings'].to_list()) 

  # GloVe
  elif embedding_type == "glov":
    X = np.asarray(data['glove_embeddings'].to_list())

  y = data['sense_id'].to_list()

  return X, y

"""Classification functions"""

# Cross validation Classification
def cv_classification(X, y, nb_splits):
  """performs cross validation classification and outputs the average over all folds
  
  y: list[int]
  """
  
  sk_fold = StratifiedKFold(n_splits=nb_splits)

  warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")

  if len(set(y)) > 1:
    cv_classif = MLPClassifier(max_iter=500, hidden_layer_sizes=(300,))
    scores = cross_val_score(cv_classif, X, y, cv=sk_fold, scoring="f1_micro")
    return scores.mean()

  else:
    return 0
  
# Cross validation Classification for tf idf
def cv_classification_tf_idf(l_sentences, y, nb_splits):
  """performs cross validation classification and outputs the average over all folds"""
  
  sk_fold = StratifiedKFold(n_splits=nb_splits)
  tfidf_vectorizer = TfidfVectorizer()
  cv_clf = MLPClassifier(max_iter=500, hidden_layer_sizes=(300,))

  warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._split")

  pipeline = make_pipeline(tfidf_vectorizer, cv_clf)

  if len(set(y)) > 1:
    scores = cross_val_score(pipeline, l_sentences, y, cv=sk_fold, scoring='f1_micro')
    return scores.mean()

  else:
    return 0
  
  # # Iterate through each fold
  # for train_index, test_index in sk_fold.split(l_sentences, y):
  #     X_train, X_test = [l_sentences[i] for i in train_index], [l_sentences[i] for i in test_index]
  #     y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

  #     X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
  #     X_test_tfidf = tfidf_vectorizer.transform(X_test)

  #     if len(set(y)) > 1:
  #       cv_classif = MLPClassifier(max_iter=300).fit(X_train_tfidf, y_train)

  #       scores = cross_val_score(cv_classif, X, y, cv=sk_fold, scoring="f1_micro")
  #       return scores.mean()

  #     else:
  #       return 0

# Traditional Classification
def traditional_classification(X_train,X_test,y_train, y_test):

  # train classifier
  if len(set(y_train)) > 1:
    classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(300,)).fit(X_train, y_train)
    
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

  for count,  lemma in enumerate(list_of_verbs):
    print(f"Lemma {count+1} out of {len(list_of_verbs)}: {lemma}")

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()
      
    # split data into test and train
    X, y = get_x_y(data, "ft")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) # shuffles data by default before splitting

    # train classifiers with different methods
    print("trad classif")
    trad_classif.append(traditional_classification(X_train, X_test, y_train, y_test))
    print("Score:", trad_classif[-1])
    print("cv classif")
    cv.append(cv_classification(X, y, 5))
    print("Score:", cv[-1], "\n")

  return (round(np.mean(np.asarray(trad_classif)),3), round(np.mean(np.asarray(cv)),3))


# compare fasttext embeddings with word2vec embeddings (cross validation)
def compare_embeddings(df):
  """
  returns mean score for each lemma for fasttext embeddings and word2vec embeddings (with cross validation)
  """

  fast_emb = []
  w2v_emb = []
  glov_emb = []
  tf_idf_emb = []

  list_of_verbs = df['lemma'].unique()

  for count,  lemma in enumerate(list_of_verbs):
    print(f"Lemma {count+1} out of {len(list_of_verbs)}: {lemma}")

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()

    # classification with fasttext
    X_fast, y = get_x_y(data, "ft")
    fast_emb.append(cv_classification(X_fast, y, 5))
    print("Fasttext classification:", fast_emb[-1])

    # classification with word2vec
    X_w2v, y = get_x_y(data, "w2v")
    w2v_emb.append(cv_classification(X_w2v, y, 5))
    print("Word2Vec classification:", w2v_emb[-1])

    # classification with glove
    X_glov, y = get_x_y(data, "glov")
    glov_emb.append(cv_classification(X_glov, y, 5))
    print("GloVe classification:", glov_emb[-1])

    # tfidf
    l_sentences = data["sentence"].to_list()
    tf_idf_emb.append(cv_classification_tf_idf(l_sentences, y, 5))
    print("TF-IDF classification:", tf_idf_emb[-1], "\n")

  return (round(np.mean(np.asarray(fast_emb)),3), round(np.mean(np.asarray(w2v_emb)),3), round(np.mean(np.asarray(glov_emb)),3), round(np.mean(np.asarray(tf_idf_emb)), 3))


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

    for count, lemma in enumerate(list_of_verbs):

      print(f"Lemma {count+1} out of 66: {lemma} with {(1-t_size) * 100}% of examples")

      # compute data frame with only one lemma
      data = df[(df['lemma'] == lemma)].reset_index()

      # get embeddings and gold labels
      X_total, y_total = get_x_y(data, "ft")
      
      # exception for case where all training examples are used
      # decrease training examples with train_test_split and only use test set
      if should_be_nb < 50:
        X, X_left_out, y, y_left_out = train_test_split(X_total, y_total, test_size=t_size, random_state=5)
        
      else: 
        X = X_total
        y = y_total

      real_nb = X.shape[0]

      # exception for case when split is smaller than 2
      # always size test set = 10 except for 15 and 10 examples (test set = 7 and 5 examples)

      if real_nb // 10 > 1:
        split = real_nb // 10
      else : split = 2

      print(f"{real_nb} examples.\nK-fold with {split} splits")
      

      scores_per_lemma.append(cv_classification(X, y, split))
      print(f"Score for {lemma}: {scores_per_lemma[-1]}\n")
    
    scores_np = np.nan_to_num(np.asarray(scores_per_lemma))    
    scores.append(round(np.mean(scores_np),3))
    print(f"Scores for {(1-t_size) * 100}% of examples: {scores[-1]}\n")
    t_size += 0.1
    should_be_nb -= 5
    
  return scores

'''Save trained classifiers as joblib'''

def save_trained_classif(df): 
  '''Saves the models to a joblib file'''
  list_of_verbs = df['lemma'].unique()

  for count,  lemma in enumerate(list_of_verbs):
    print(f"Lemma {count+1} out of {len(list_of_verbs)}: {lemma}")

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()

    X, y = get_x_y(data, "ft")
    classifier = MLPClassifier(max_iter=500, hidden_layer_sizes=(300,))
    print("Train classifier")
    classifier.fit(X, y)

    print("Save model\n")
    file_name = f"../trained_models/{lemma}.joblib"
    dump(classifier, file_name)

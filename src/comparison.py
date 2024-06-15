from src.kmeans import Kmeans, KmeansConstraint
from src.classification import cv_classification, get_x_y, cv_classification_tf_idf
import numpy as np

#comparison between WSI and WSD

def compare(df, n):
  # List of verbs in the Dataset: 66 verbs
  list_of_verbs = df['lemma'].unique()

  scores_kmeans = []
  scores_kmeans_constr = []
  scores_classif = []
  scores_classif_tfidf = []

  # loop through the verbs, we classify and cluster for each verb
  for count,  verb in enumerate(list_of_verbs):
    print(f"Lemma {count+1} out of {len(list_of_verbs)}: {verb}")

    verb_df = df[df['lemma'] == verb].reset_index()

    # number of clusters
    k = len(verb_df['word_sense'].unique())

    # instantiate KMeans Clustering
    my_kmeans = Kmeans(verb_df, k, "ft_embeddings")
    my_kmeans.fit()

    # instantiate KMeans Clustering with constraints
    my_kmeans_constraint = KmeansConstraint(verb_df, k, "ft_embeddings", n)
    my_kmeans_constraint.fit()

    # evaluate clustering
    scores_kmeans.append(my_kmeans.evaluate()) # type: ignore
    scores_kmeans_constr.append(my_kmeans_constraint.evaluate())
    print("K-means:", scores_kmeans[-1])
    print("Constrained k-means:", scores_kmeans_constr[-1])

    # evaluate classification
    X, y = get_x_y(verb_df, "ft")
    scores_classif.append(cv_classification(X, y, 5))
    print("Classification:", scores_classif[-1])

    # evaluate tf_idf
    l_sentences = verb_df["sentence"].to_list()
    scores_classif_tfidf.append(cv_classification_tf_idf(l_sentences, y, 5))
    print("TF-IDF:", scores_classif_tfidf[-1], "\n")


  print(f"mean score k-means: {np.mean(np.asarray(scores_kmeans))}")
  print(f"mean score k-means constraint: {np.mean(np.asarray(scores_kmeans_constr))}")
  print(f"mean score classif: {np.mean(np.asarray(scores_classif))}")
  print(f"mean score tf_idf classif: {np.mean(np.asarray(scores_classif_tfidf))}")

def print_comparison_kmeans_clf(df, max_ex4constraint=10):
  # List of verbs in the Dataset: 66 verbs
  list_of_verbs = df['lemma'].unique()

  scores_kmeans = [[] for i in range(max_ex4constraint)]
  scores_classif = []

  # loop through the verbs, we classify and cluster for each verb
  for count,  verb in enumerate(list_of_verbs):
    print(f"Lemma {count+1} out of {len(list_of_verbs)}: {verb}")

    verb_df = df[df['lemma'] == verb].reset_index()

    # number of clusters
    k = len(verb_df['word_sense'].unique())

    for nb_ex4constraint in range(max_ex4constraint):
      # instantiate KMeans Clustering with constraints
      my_kmeans_constraint = KmeansConstraint(verb_df, k, "ft_embeddings", nb_ex4constraint)
      my_kmeans_constraint.fit()
      scores_kmeans[nb_ex4constraint].append(my_kmeans_constraint.evaluate()) # type: ignore
      print(f"K-means constrained on {nb_ex4constraint+1} examples: {scores_kmeans[nb_ex4constraint][-1]}")

    # evaluate classification
    X, y = get_x_y(verb_df, "ft")
    scores_classif.append(cv_classification(X, y, 5))
    print("Classification:", scores_classif[-1], "\n")


  final_score_classif = np.mean(np.asarray(scores_classif))

  for nb_ex4constraint, score in enumerate(scores_kmeans):
    print(f"mean score k-means constrained on {nb_ex4constraint+1} examples: {np.mean(np.asarray(score))}")
    print(f"mean score classif: {final_score_classif} \n")


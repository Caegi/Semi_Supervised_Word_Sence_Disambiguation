from classes.kmeans import Kmeans, KmeansConstraint
from classes.classification import cv_classification, get_x_y, cv_classification_tf_idf
import numpy as np

#comparison between WSI and WSD

#if __name__ == "__main__":
def compare(df):
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
    my_kmeans_constraint = KmeansConstraint(verb_df, k, "ft_embeddings")
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


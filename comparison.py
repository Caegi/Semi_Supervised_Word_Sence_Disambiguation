from classes.kmeans import Kmeans, KmeansConstraint
from classes.classification import cv_classification, get_x_y
import numpy as np


# get dataset
#df = pd.read_json("fse_data_w_embeddings.json")

#comparison between WSI and WSD

#if __name__ == "__main__":
def compare(df):
  # List of verbs in the Dataset: 66 verbs
  list_of_verbs = df['lemma'].unique()

  scores_kmeans = []
  scores_kmeans_constr = []
  scores_classif = []

  # loop through the verbs, we classify and cluster for each verb
  for verb in list_of_verbs:
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

    # evaluate classification
    X, y = get_x_y(verb_df, "ft")
    scores_classif.append(cv_classification(X, y, 5))


  print(f"mean score k-means: {np.mean(np.asarray(scores_kmeans))}")
  print(f"mean score k-means constraint: {np.mean(np.asarray(scores_kmeans_constr))}")
  print(f"mean score classif: {np.mean(np.asarray(scores_classif))}")


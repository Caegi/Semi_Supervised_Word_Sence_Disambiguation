import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import contingency_matrix

class Kmeans:

  global nlp
  nlp = spacy.load("fr_core_news_sm")

  def __init__(self, df: pd.DataFrame, k: int, embeddings):
    """
    Initialize KMeansClustering object.
    Args:
        df (pd.DataFrame): Input DataFrame containing text data.
        k (int): Number of clusters.
    """
    self.df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    self.k = k
    self.centroids = None
    self.embeddings = embeddings

  def _init_centroids(self):
    random_indices = np.random.choice(range(len(self.df)), size=self.k, replace=False)
    centroids_df = self.df.iloc[random_indices]
    centroids_array = np.vstack(centroids_df[self.embeddings].values)
    return centroids_array

  # Method to calculate new centroids
  def _get_new_centroids(self, cs):
    new_centroids = []
    for c in cs:
      vectors = np.vstack(self.df.loc[self.df['cluster'] == c, self.embeddings].values) # type: ignore
      centroid = np.mean(vectors, axis=0)
      new_centroids.append(centroid)
    return np.array(new_centroids)

  # Method to calculate cosine similarity between centroids and all examples
  def _get_new_clusters(self, iris_array, centroids):
    clusters = []
    for vector in iris_array:
      similarities = []
      for i in range(len(centroids)):
        dot_product = np.dot(vector, centroids[i])
        m_vector = np.linalg.norm(vector)
        m_centroid = np.linalg.norm(centroids[i])
        similarity = dot_product / (m_vector * m_centroid)
        similarities.append(similarity)
      max_value = max(similarities)
      max_index = similarities.index(max_value)
      clusters.append(max_index)
    return clusters

  def fit(self):
    self.df.loc[:, 'cluster'] = 0
    # Excluding last column which is cluster label
    # self.df'embedding'] = self.df['sentence'].apply(lambda x: nlp(x).vector) # type: ignore
    embeddings_array = np.asarray(self.df[self.embeddings].to_list())
    self.centroids = self._init_centroids()
    is_changed = True

    while is_changed:
      # Making a copy of current clusters for comparison later
      clusters_before = self.df['cluster'].copy()
      new_clusters = self._get_new_clusters(embeddings_array, self.centroids)
      self.df.loc[:, 'cluster'] = new_clusters

      if not np.array_equal(np.array(new_clusters), clusters_before):
        class_unique = self.df['cluster'].unique()
        self.centroids = self._get_new_centroids(class_unique)
      else:
        is_changed = False

  def get_centroids(self):
    return self.centroids

  def get_dataframe(self):
    return self.df

  def get_distribution(self):
    return self.df['cluster'].value_counts()
  
  def evaluate(self):

    # get gold labels and predicted cluster
    y = self.df["sense_id"]
    y_1 = self.get_dataframe()["cluster"]
    
    # compute contingency matrix
    # get class index with most examples per cluster
    m1 = contingency_matrix(y, y_1)
    max_indices = np.argmax(m1, axis=0) # type: ignore

    # "replace" (in a new list) the cluster names with predicted class indices
    y_pred = []
    for c in y_1.to_list():
      y_pred.append(max_indices[c])

    # compute f-score
    score = f1_score(y, y_pred, average="micro")

    # Purity au cas où on veut l'utiliser...
    # score = np.sum(np.max(M1, axis=0)) / np.sum(M1) # type: ignore

    return score

    

class KmeansConstraint(Kmeans):
  
  # Override
  # Method to initialize centroids with first example of each unique word sense in the verb dataframe
  def _init_centroids(self):
    prototypes = self.df.groupby('word_sense').apply(lambda x: x.iloc[0, x.columns != 'word_sense'])
    centroids_array = np.vstack(prototypes[self.embeddings].values) # type: ignore
    return centroids_array


def wsi_compare_embeddings(df):
    # List of verbs in the Dataset: 66 verbs
  list_of_verbs = df['lemma'].unique()

  scores_kmeans_ft = []
  scores_kmeans_w2v = []
  scores_kmeans_glov = []

  scores_kmeans_constr_ft = []
  scores_kmeans_constr_w2v = []
  scores_kmeans_constr_glov = []

  # loop through the verbs, we classify and cluster for each verb
  for verb in list_of_verbs:
    verb_df = df[df['lemma'] == verb].reset_index()

    # number of clusters
    k = len(verb_df['word_sense'].unique())

    # K Means
    # Test fasttext vectors
    ft_kmeans = Kmeans(verb_df, k, "ft_embeddings")
    ft_kmeans.fit()
    scores_kmeans_ft.append(ft_kmeans.evaluate())

    # Test word2vec vectors
    w2v_kmeans = Kmeans(verb_df, k, "w2v_embeddings")
    w2v_kmeans.fit()
    scores_kmeans_w2v.append(w2v_kmeans.evaluate())

    # Test glove vectors
    glov_kmeans = Kmeans(verb_df, k, "glove_embeddings")
    glov_kmeans.fit()
    scores_kmeans_ft.append(glov_kmeans.evaluate())

    # Constraint K-Means
    # Test fasttext vectors
    ft_kmeans_constraint = KmeansConstraint(verb_df, k, "ft_embeddings")
    ft_kmeans_constraint.fit()
    scores_kmeans_constr_ft.append(ft_kmeans_constraint.evaluate())

    # Test word2vec vectors
    w2v_kmeans_constraint = KmeansConstraint(verb_df, k, "w2v_embeddings")
    w2v_kmeans_constraint.fit()
    scores_kmeans_constr_w2v.append(w2v_kmeans_constraint.evaluate())

    # Test glove vectors
    glov_kmeans_constraint = KmeansConstraint(verb_df, k, "glove_embeddings")
    glov_kmeans_constraint.fit()
    scores_kmeans_constr_glov.append(glov_kmeans_constraint.evaluate())

  # replace NaN values with 0
  scores_kmeans_ft = np.nan_to_num(np.asarray(scores_kmeans_ft))
  scores_kmeans_w2v = np.nan_to_num(np.asarray(scores_kmeans_w2v))
  scores_kmeans_glov = np.nan_to_num(np.asarray(scores_kmeans_glov))

  scores_kmeans_constr_ft = np.nan_to_num(np.asarray(scores_kmeans_constr_ft))
  scores_kmeans_constr_w2v = np.nan_to_num(np.asarray(scores_kmeans_constr_w2v))
  scores_kmeans_constr_glov = np.nan_to_num(np.asarray(scores_kmeans_constr_glov))

  print(f"K-Means:\nFastText: {round(np.mean(scores_kmeans_ft), 3)}\nWord2Vec: {round(np.mean(scores_kmeans_w2v), 3)}\nGloVe: {round(np.mean(scores_kmeans_glov), 3)}")
  print(f"\nConstraint K-Means:\nFastText: {round(np.mean(np.asarray(scores_kmeans_constr_ft)))}\nWord2Vec: {round(np.mean(np.asarray(scores_kmeans_constr_w2v)))}\nGloVe: {round(np.mean(np.asarray(scores_kmeans_constr_glov)))}")
  
  

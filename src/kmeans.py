import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import contingency_matrix
from joblib import dump

class Kmeans:

  #global nlp
  #nlp = spacy.load("fr_core_news_sm")

  def __init__(self, df: pd.DataFrame, k: int, emb_src: str):
    """
    Initialize KMeansClustering object.
    Args:
        df (pd.DataFrame): Input DataFrame containing text data.
        k (int): Number of clusters.
    """
    self.df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    self.k = k
    self.centroids = None
    self.emb_src = emb_src

  def _init_centroids(self):
    random_indices = np.random.choice(range(len(self.df)), size=self.k, replace=False)
    centroids_df = self.df.iloc[random_indices]
    centroids_array = np.vstack(centroids_df[self.emb_src].values)
    return centroids_array

  # Method to calculate new centroids
  def _get_new_centroids(self, cs):
    new_centroids = []
    for c in cs:
      vectors = np.vstack(self.df.loc[self.df['cluster'] == c, self.emb_src].values) # type: ignore
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
    embeddings_array = np.asarray(self.df[self.emb_src].to_list())
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
    def __init__(self, df: pd.DataFrame, k: int, emb_src: str, nb_ex4constraint: int):
        super().__init__(df, k, emb_src)
        self.nb_ex4constraint = nb_ex4constraint

    # Override
    # Method to initialize centroids with the first example of each unique word sense in the DataFrame
    def _init_centroids(self):

        nb_ex4constraint = self.nb_ex4constraint + 1 # to avoid index errors
        all_mean_embeddings = []
        unique_senses = self.df['word_sense'].unique()

        for sense in unique_senses:

            mean_embedding = None  # Initialize mean_embedding

            indices = self.df['word_sense'] == sense
            sense_df = self.df[indices]  # DataFrame that contains all the examples of a unique sense

            if len(sense_df) >= nb_ex4constraint:
              embeddings = np.array(sense_df.head(nb_ex4constraint)[self.emb_src].tolist())
              mean_embedding = embeddings.mean(axis=0)

            elif 1 < len(sense_df) < nb_ex4constraint:
              embeddings = np.array(sense_df[self.emb_src].tolist())
              mean_embedding = embeddings.mean(axis=0)

            else:
              embedding = sense_df[self.emb_src].values[0]
              mean_embedding = np.array(embedding)

            all_mean_embeddings.append(mean_embedding)

        centroids = np.vstack(all_mean_embeddings)

        return centroids

def wsi_compare_embeddings(df, nb_ex4constraint):
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
    scores_kmeans_glov.append(glov_kmeans.evaluate())

    # Constraint K-Means
    # Test fasttext vectors
    ft_kmeans_constraint = KmeansConstraint(verb_df, k, "ft_embeddings", nb_ex4constraint)
    ft_kmeans_constraint.fit()
    scores_kmeans_constr_ft.append(ft_kmeans_constraint.evaluate())

    # Test word2vec vectors
    w2v_kmeans_constraint = KmeansConstraint(verb_df, k, "w2v_embeddings", nb_ex4constraint)
    w2v_kmeans_constraint.fit()
    scores_kmeans_constr_w2v.append(w2v_kmeans_constraint.evaluate())

    # Test glove vectors
    glov_kmeans_constraint = KmeansConstraint(verb_df, k, "glove_embeddings", nb_ex4constraint)
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
  print(f"\nConstraint K-Means:\nFastText: {round(np.mean(scores_kmeans_constr_ft), 3)}\nWord2Vec: {round(np.mean(scores_kmeans_constr_w2v), 3)}\nGloVe: {round(np.mean(scores_kmeans_constr_glov), 3)}")
  
  
def save_trained_kmeans(df):
  '''Saves the models to a joblib file'''
  list_of_verbs = df['lemma'].unique()

  for count,  lemma in enumerate(list_of_verbs):
    print(f"Lemma {count+1} out of {len(list_of_verbs)}: {lemma}")

    # compute data frame with only one lemma
    data = df[(df['lemma'] == lemma)].reset_index()

    # number of clusters
    k = len(data['word_sense'].unique())

    # instantiate KMeans Clustering
    print("Fit K-Means")
    my_kmeans = KmeansConstraint(data, k, "ft_embeddings", 9)
    my_kmeans.fit()

    print("Save model\n")
    file_name = f"../trained_kmeans/{lemma}.joblib"
    dump(my_kmeans, file_name)

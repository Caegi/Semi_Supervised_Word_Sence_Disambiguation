import pandas as pd
import numpy as np
import spacy

class kmeans:

  def __init__(self, df: pd.DataFrame, k: int):
    """
    Initialize KMeansClustering object.
    Args:
        df (pd.DataFrame): Input DataFrame containing text data.
        k (int): Number of clusters.
    """
    self.df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    self.k = k
    self.centroids = None
   

  # Method to initialize centroids randomly
  def _init_centroids(self):
    
    prototypes = self.df.groupby('word_sense').apply(lambda x: x.iloc[0, x.columns != 'word_sense'])
    centroids_array = np.vstack(prototypes['embedding'].values)
    return centroids_array

  # Method to calculate new centroids
  def _get_new_centroids(self, cs):
    new_centroids = []
    for c in cs:
      vectors = np.vstack(self.df.loc[self.df['cluster'] == c, 'embedding'].values)
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
    self.df.loc[:, 'embedding'] = self.df['sentence'].apply(lambda x: nlp(x).vector)
    embeddings_array = np.vstack(self.df['embedding'].values)
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
    print(self.df)

  def get_distribution(self):
    return self.df['cluster'].value_counts()

if __name__ == "__main__":
  global nlp
  nlp = spacy.load("fr_core_news_sm")
 
  # Loading Dataset
  df = pd.read_csv('/home/raymond/Bureau/WSB/fse_data.csv')

  # List of verbs in the Dataset: 66 verbs
  list_of_verbs = df['lemma'].unique()

  # Testing with the verb "aboutir"
  for verb in list_of_verbs:
    verb_df = df[df['lemma'] == verb]
    print(f"Verb : {verb}")

    # Number of clusters
    k = len(verb_df['word_sense'].unique())

    # Instantiate KMeans Clustering
    my_kmeans = kmeans(verb_df, k)
    my_kmeans.fit()

    print(my_kmeans.get_distribution())


  '''
    the number of senses for each word is the number of k 
    pick a prototype for each sense or (mean) not randomly
  '''

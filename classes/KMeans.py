import pandas as pd
import numpy as np
import spacy

class kmeans:

  def __init__(self,df:pd.DataFrame,k:int):

    self.df = df
    self.k = k
    self.centroids = None

    print("classe created successfuly")

  # Methode to initialize centroids randomly
  def _init_centroids(self,iris_array):
    print('initalizing centroids en cours')
    random_indices = np.random.choice(range(len(iris_array)), size=self.k, replace=False)
    centroids = iris_array[random_indices]
    return centroids

  #  to calculate new centroids
  def _get_new_centroids(self, cs):

      new_centroids = []

      for c in cs:

        vectors = np.vstack((self.df[self.df['cluster'] ==c])['embedding'].values)
        centroid = np.mean(vectors, axis=0)
        new_centroids.append(centroid)

      return (np.array(new_centroids))

  # Methode  to calculate cosine similarity between centroids and all examples:
  def _get_new_clusters(self,iris_array,centroids):

    clusters = []
    for vector in iris_array:
      similarities = []
      for i in range(len(centroids)):

        dot_product = np.dot(vector,centroids[i])
        m_vector = np.linalg.norm(vector)
        m_centroid = np.linalg.norm(centroids[i])
        similarity = dot_product / (m_vector * m_centroid)
        similarities.append(similarity)

      max_value = max(similarities) ;  max_index = similarities.index(max_value)
      clusters.append(max_index)

    return clusters



  def fit(self):

    self.df['cluster'] = 0

     # Excluding last column which is cluster label
  
    self.df['embedding'] = self.df['sentence'].apply(lambda x : nlp(x).vector)
    embeddings_array = np.vstack(self.df['embedding'].values)
    self.centroids = self._init_centroids(embeddings_array)

    is_changed = True

    while is_changed:

      # Making a copy of current clusters for comparison late
      clusters_before = self.df['cluster'].copy()
      new_clusters = self._get_new_clusters(embeddings_array,self.centroids)
      self.df['cluster'] = new_clusters

      if not np.array_equal(np.array(new_clusters), clusters_before):
            class_unique = self.df['cluster'].unique()
            self.centroids =  self._get_new_centroids(class_unique)
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
  verbe_df = df[df['lemma']=="aboutir"]


  # Number of cluster
  k = 4
  # Instanciate Kmeans Clustering
  my_kmeans = kmeans(verbe_df,k)
  my_kmeans.fit()

  print(my_kmeans.get_distribution())
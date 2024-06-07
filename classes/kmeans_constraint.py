import pandas as pd
import numpy as np
import spacy
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import contingency_matrix
from classes.kmeans import Kmeans

class KmeansConstraint(Kmeans):
  
  # Override
  # Method to initialize centroids with first example of each unique word sense in the verb dataframe
  def _init_centroids(self):
    prototypes = self.df.groupby('word_sense').apply(lambda x: x.iloc[0, x.columns != 'word_sense'])
    centroids_array = np.vstack(prototypes['w2v_embeddings'].values) # type: ignore
    return centroids_array
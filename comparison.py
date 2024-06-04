from classes.KMeans import kmeans
from classes.Classification import cv_classification, get_x_y_w2v
from data_preparation import get_data
import pandas as pd
import numpy as np


# get dataset
df = get_data()


# comparison between WSI and WSD

if __name__ == "__main__":

  # List of verbs in the Dataset: 66 verbs
  list_of_verbs = df['lemma'].unique()

  scores_kmeans = []
  scores_classif = []

  # Testing with the verb "aboutir"
  for verb in list_of_verbs:
    verb_df = df[df['lemma'] == verb].reset_index()
    print(f"Verb : {verb}")

    # Number of clusters
    k = len(verb_df['word_sense'].unique())

    # Instantiate KMeans Clustering
    my_kmeans = kmeans(verb_df, k)
    my_kmeans.fit()

    # evaluate clustering
    scores_kmeans.append(my_kmeans.evaluate_kmeans()) # type: ignore

    # evaluate classification
    X, y = get_x_y_w2v(verb_df)
    scores_classif.append(cv_classification(X, y, 5))


  # print(f"k-means: {scores_kmeans}")
  # print(f"classif: {scores_classif}\n")

  print(f"mean score k-means: {np.mean(np.asarray(scores_kmeans))}")
  print(f"mean score classif: {np.mean(np.asarray(scores_classif))}")


    

  '''
    the number of senses for each word is the number of k 
    pick a prototype for each sense or (mean) not randomly
  '''


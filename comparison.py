from classes.KMeans import kmeans
from classes.Classification import cv_classification
from data_preparation import get_data
import pandas as pd
import numpy as np

df = get_data()
# comparison between WSI and WSD

if __name__ == "__main__":
 
  # Loading Dataset
  #df = pd.read_csv('/home/raymond/Bureau/WSB/fse_data.csv')
  #df = pd.read_csv('fse_data.csv')

  # adding column with id for every sense to the dataframe
  # df['sense_id'] = df.apply(lookup, axis=1)

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
    scores_kmeans.append(my_kmeans.evaluate_kmeans())

    # evaluate classification
    scores_classif.append(cv_classification(verb_df, verb, 5))

  # print(f"k-means: {scores_kmeans}")
  # print(f"classif: {scores_classif}\n")

  print(f"mean score k-means: {np.mean(np.asarray(scores_kmeans))}")
  print(f"mean score classif: {np.mean(np.asarray(scores_classif))}")


    

  '''
    the number of senses for each word is the number of k 
    pick a prototype for each sense or (mean) not randomly
  '''


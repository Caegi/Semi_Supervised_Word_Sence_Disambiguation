from classes.data_preparation import get_data
from classes.kmeans import Kmeans
from classes.classification import cv_classification, get_x_y_w2v, compare_embeddings, compare_split_method, get_best, decrease_training_examples
import pandas as pd
import numpy as np
from classes.kmeans_constraint import KmeansConstraint

# get dataset
df = get_data()


# comparison between WSI and WSD

# if __name__ == "__main__":

#   # List of verbs in the Dataset: 66 verbs
#   list_of_verbs = df['lemma'].unique()

#   scores_kmeans = []
#   scores_kmeans_constraint = []
#   scores_classif = []

#   # Testing with the verb "aboutir"
#   for i_verb in range(len(list_of_verbs)):
#     print(f"Verb {i_verb+1} / {len(list_of_verbs)}")
#     verb_df = df[df['lemma'] == list_of_verbs[i_verb]].reset_index()

#     # Number of clusters
#     k = len(verb_df['word_sense'].unique())

#     # Instantiate KMeans Clustering without constraints
#     my_kmeans = Kmeans(verb_df, k)
#     my_kmeans.fit()

#     # Instantiate KMeans Clustering with constraints
#     my_kmeans_constraint = KmeansConstraint(verb_df, k)
#     my_kmeans_constraint.fit()

#     # evaluate clustering
#     scores_kmeans.append(my_kmeans.evaluate_kmeans()) # type: ignore
#     scores_kmeans_constraint.append(my_kmeans_constraint.evaluate_kmeans())

#     # evaluate classification
#     X, y = get_x_y_w2v(verb_df)
#     splits_cross_validation = 5
    
#     scores_classif.append(cv_classification(X, y, splits_cross_validation))

#   print(f"mean score k-means without constraints: {np.mean(np.asarray(scores_kmeans))}")
#   print(f"mean score k-means with constraints: {np.mean(np.asarray(scores_kmeans_constraint))}")
#   print(f"mean score classif: {np.mean(np.asarray(scores_classif))}")


# print("Compare split method:")
# trad_classification, cv = compare_split_method(df)
# get_best(trad_classification, cv, ["70/30 split", "Cross Validation"])

print("\nCompare embeddings")
fast_emb, w2v_emb, tf_idf_emb = compare_embeddings(df)
get_best(fast_emb, w2v_emb, tf_idf_emb, ["Fasttext", "Word2Vec", "TF_IDF"])

# print(decrease_training_examples(df))
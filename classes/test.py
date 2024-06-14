
from kmeans import Kmeans
from comparison import print_comparison_kmeans2clf
import pandas as pd
import numpy as np
from numpy.linalg import norm
import fasttext

# ft = fasttext.load_model('../cc.fr.300.bin')
# sentence_emb = ft.get_sentence_vector("En mai 1929 , il accomplit avec le croiseur Trento un voyage de La Spezia à Barcelone")

# verb_df = df[df['lemma'] == "aboutir"].reset_index()

# # number of clusters
# k = len(verb_df['word_sense'].unique())
# print("Fit k-means")
# # instantiate KMeans Clustering
# my_kmeans = Kmeans(verb_df, k, "ft_embeddings")
# my_kmeans.fit()

# centroides = my_kmeans.get_centroids()

# def cosine_similarity(a, b):
#     return np.dot(a, b)/(norm(a)* norm(b))

# print("compute distance")
# dist = [cosine_similarity(c, sentence_emb) for c in centroides]

# print(np.argmin(dist))
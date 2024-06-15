from classification import get_best, compare_embeddings, compare_split_method, decrease_training_examples
#from classes.classification import save_trained_classif
from joblib import load
from kmeans import wsi_compare_embeddings
#from kmeans import save_trained_kmeans
from comparison import compare, print_comparison_kmeans_clf
import arg_parser as a
import pandas as pd
import fasttext
import numpy as np
from numpy.linalg import norm

df = pd.read_json("src/fse_data_w_embeddings.json")

#save_trained_classif(df)
#save_trained_kmeans(df)

def same_length(verb):
    
    spaces = [" "] * (10 - len(verb))
    return verb + "".join(spaces)

# show tests on WSD
if a.online_help().wsd:

    print("WARNING: This will take around 1h to run!\n")

    # get comparison
    print("Compute comparison for split methods")
    trad_classification, cv = compare_split_method(df)
    print("compute comparison for embeddings")
    fast_emb, w2v_emb, glov_emb, tf_idf = compare_embeddings(df)

    # print results
    print("Compare split method:")
    get_best([trad_classification, cv], ["70/30 split", "Cross Validation"])
    print("\nCompare embeddings")
    get_best([fast_emb, w2v_emb, glov_emb, tf_idf], ["FastText", "Word2Vec", "GloVe", "TF-IDF"])

# show tests on WSI
elif a.online_help().wsi:
    exs_2add_as_constraint = 2
    wsi_compare_embeddings(df, exs_2add_as_constraint)

# show k-means, constraint k-means, classification and classification with TF-IDF
elif a.online_help().compare:
    exs_2add_as_constraint = 2
    compare(df, exs_2add_as_constraint)

# compare the constrained kmeans and the classifier to see how many examples
#  should be added as constraint for the constrained kmeans to get a better score
elif a.online_help().compare_km_clf:
    print_comparison_kmeans_clf(df)

# get word sense for a new sentence 
elif a.online_help().sentence:
    if a.online_help().lemma:
        
        # get sentence embedding with fasttext vector
        print(f"Your input sentence is: {a.online_help().sentence}")
        print("Your sentence is being processed...")
        ft = fasttext.load_model('../cc.fr.300.bin')
        sentence_embedding = ft.get_sentence_vector(a.online_help().sentence)

        # for WSD Classification
        if a.online_help().mode == "wsd":
            print("You chose to get the word sense with our WSD model.")
            # get file to load the model
            file_path = f"../trained_models/{a.online_help().lemma}.joblib"
            model = load(file_path)

            # model predicts sense-id, get actual sense of the word
            sense = df.loc[(df["sense_id"] == model.predict(sentence_embedding.reshape(1, -1))[0]) & (df['lemma'] == a.online_help().lemma)].reset_index()
            print(f"The word {a.online_help().lemma} has sense {sense['word_sense'][0]} in the provided sentence.")

        # for WSI Clustering
        else: 
            print("You chose to get the word sense with our WSI model.")

            # get file to load pretrained k-means
            file_path = f"../trained_kmeans/{a.online_help().lemma}.joblib"
            model = load(file_path)

            # predict cluster of the sentence
            centroids = model.get_centroids()

            # compute cosine similarity
            normalized_centroids = centroids / norm(centroids)
            normalized_embeddings = sentence_embedding / norm(sentence_embedding)

            cosine = np.matmul(normalized_centroids, normalized_embeddings.T)

            print(f"The word {a.online_help().lemma} has the sense of cluster {np.argmin(cosine)} in the provided sentence")


    else: print("Please provide a sentence and a lemma to execute the code.")

elif a.online_help().verbs:
    
    list_of_verbs = sorted(df['lemma'].unique())

    for i in range(0, len(list_of_verbs)-3, 4):

        print(f"{same_length(list_of_verbs[i])}\t{same_length(list_of_verbs[i+1])}\t", \
              f"{same_length(list_of_verbs[i+2])}\t{same_length(list_of_verbs[i+3])}")


elif a.online_help().lemma:
    print("Please provide a sentence with your lemma.")

elif a.online_help().decrease:
    decrease_training_examples(df)
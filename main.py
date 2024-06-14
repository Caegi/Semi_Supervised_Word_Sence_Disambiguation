from classes.classification import get_best, compare_embeddings, compare_split_method
#from classes.classification import save_trained_classif
from joblib import load
from classes.kmeans import wsi_compare_embeddings, Kmeans
#from classes.kmeans import save_trained_kmeans
from classes.comparison import compare
import classes.arg_parser as a
import pandas as pd
import fasttext
import numpy as np
from numpy.linalg import norm

df = pd.read_json("src/fse_data_w_embeddings.json")

#save_trained_classif(df)
#save_trained_kmeans(df)


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
    wsi_compare_embeddings(df)

# show k-means, constraint k-means, classification and classification with TF-IDF
elif a.online_help().compare:
    compare(df)

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
            print("You cose to get the word sense with our WSI model.")

            # define cosine similaritiy
            def cosine_similarity(a, b):
                return np.dot(a, b)/(norm(a)* norm(b))

            # get file to load pretrained k-means
            file_path = f"../trained_kmeans/{a.online_help().lemma}.joblib"
            model = load(file_path)

            # predict cluster of the sentence
            centroids = model.get_centroids()
            dist = [cosine_similarity(c, sentence_embedding) for c in centroids]

            print(f"The word {a.online_help().lemma} has the sense of cluster {np.argmin(dist)} in the provided sentence")


    else: print("Please provide a sentence and a lemma to execute the code.")

# elif a.online_help().lemma:
#     if a.online_help().sentence:

#       # get file to load the model
        # file_path = f"../trained_models/{a.online_help().lemma}.joblib"
        # model = load(file_path)

        # # get sentence embedding with fasttext vector
        # ft = fasttext.load_model('../cc.fr.300.bin')
        # sentence_embedding = ft.get_sentence_vector(a.online_help().sentence).reshape(1, -1)
        # print(f"The word {a.online_help.lemma} has sense number {model.predict(sentence_embedding)} in the provided sentence.")

#     else: print("Please provide a sentence and a lemma to execute the code.")
from classes.classification import get_best, compare_embeddings, compare_split_method
from classes.classification import save_trained_classif
from joblib import load
from classes.kmeans import wsi_compare_embeddings
from classes.comparison import compare
import classes.arg_parser as a
import pandas as pd
import fasttext

df = pd.read_json("src/fse_data_w_embeddings.json")

#save_trained_classif(df)


if a.online_help().wsd:
    print("WARNING: This will take around 1h to run!\n")
    print("Compute comparison for split methods")
    trad_classification, cv = compare_split_method(df)
    print("compute comparison for embeddings")
    fast_emb, w2v_emb, glov_emb, tf_idf = compare_embeddings(df)
    print("Compare split method:")
    get_best([trad_classification, cv], ["70/30 split", "Cross Validation"])
    print("\nCompare embeddings")
    get_best([fast_emb, w2v_emb, glov_emb, tf_idf], ["FastText", "Word2Vec", "GloVe", "TF-IDF"])

elif a.online_help().wsi:
    wsi_compare_embeddings(df)

elif a.online_help().compare:
    compare(df)

if a.online_help().sentence:
    if a.online_help().lemma:
        # get file to load the model
        file_path = f"../trained_models/{a.online_help().lemma}.joblib"
        model = load(file_path)

        # get sentence embedding with fasttext vector
        ft = fasttext.load_model('../cc.fr.300.bin')
        sentence_embedding = ft.get_sentence_vector(a.online_help().sentence).reshape(1, -1)

        if a.online_help().mode == "wsd":
            sense = df.loc[df["sense_id"] == model.predict(sentence_embedding)[0]].reset_index()
            print(f"The word {a.online_help().lemma} has sense {sense['word_sense'][0]} in the provided sentence.")

        #else: insert here same for k-means! Default is k-means


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
#from comparison import compare
from classes.classification import get_best, compare_embeddings, compare_split_method
from classes.kmeans import wsi_compare_embeddings
from classes.comparison import compare
import classes.arg_parser as a
import pandas as pd

df = pd.read_json("fse_data_w_embeddings.json")

if a.online_help().wsd:
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

elif a.online_help().sentence:
    if a.online_help().lemma:
        print(f"You want to analyze the word {a.online_help().lemma} in the sentence {a.online_help().sentence}")
    else: print("Please provide a sentence and a lemma to execute the code.")

elif a.online_help().lemma:
    if a.online_help().sentence:
        print(f"You want to analyze the word {a.online_help().lemma} in the sentence {a.online_help().sentence}")
    else: print("Please provide a sentence and a lemma to execute the code.")
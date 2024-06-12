#from comparison import compare
from classes.classification import get_best, compare_embeddings, compare_split_method
from classes.kmeans import wsi_compare_embeddings
from comparison import compare
import arg_parser as a
import pandas as pd

df = pd.read_json("fse_data_w_embeddings.json")

if a.online_help().wsd:
    trad_classification, cv = compare_split_method(df)
    fast_emb, w2v_emb, glov_emb = compare_embeddings(df)
    print("Compare split method:")
    get_best([trad_classification, cv], ["70/30 split", "Cross Validation"])
    print("\nCompare embeddings")
    get_best([fast_emb, w2v_emb, glov_emb], ["FastText", "Word2Vec", "GloVe"])

elif a.online_help().wsi:
    wsi_compare_embeddings(df)

elif a.online_help().compare:
    compare(df)
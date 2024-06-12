'''Import libraries'''
import pandas as pd
from spacy.lang.fr import French
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

'''Set up tokenizer'''
nlp = French()
tokenizer = nlp.tokenizer

def tokenize(row):
    '''returns tokenized sentence'''
    
    return " ".join([t.text for t in tokenizer(row.lower())])


'''Load corpus'''
df_tot = pd.read_parquet("../gallica_presse_1.parquet")
# only use the firs 1500 documents (vocab size: 163 115 tokens (only those appearing 5 times or more))
df = df_tot.head(1500) 

'''Tokenize text'''
df.loc[:, "complete_text"] = df["complete_text"].apply(tokenizer)
print(df['complete_text'].head(1))

'''Create txt file with tokenized text'''
i = 0
j = 1
with open("../corpus_glove_sentence_split.txt", "w") as f:
    for sentence in df['complete_text'].to_list():
        print(f"file {j} out of 1500")
        for t in tokenizer(sentence.lower()):
            f.write(f"{t.text} ")
            i+=1
        j+=1

print(f"Number of examples: {i}")


'''load trained vectors'''
# change to word2vec format
glove2word2vec('../vectors.txt', '../vectors_glov_w2v.txt')# Load pre-trained GloVe embeddings

# load the model so we can use it
glove_model = KeyedVectors.load_word2vec_format('../vectors_glov_w2v.txt', binary=False)

# test with most similar words to "difficile"
print(glove_model.most_similar('difficile'))
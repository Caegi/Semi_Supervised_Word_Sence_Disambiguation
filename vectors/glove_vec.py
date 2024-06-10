
from glove import Corpus, Glove
import pandas as pd
from spacy.lang.fr import French

# print("load corpus")
# df_tot = pd.read_parquet("vectors/gallica_presse_1.parquet")

# df = df_tot.head(10000)

# sentence_list = df["complete_text"].str.split("\n")
# print(sentence_list[0][1])

# i = 0
# with open("vectors/corpus_glove_sentence_split.txt", "w") as f:
#     for sentence in sentence_list:
#         for w in sentence:
#             f.write(f"{w}\n")
#             i+=1

# print(f"Number of examples: {i}")

nlp = French()
tokenizer = nlp.tokenizer

print("load corpus")

with open ("vectors/corpus_glove_sentence_split.txt", "r") as f:
    text = []   
    line = f.readline().lower()
    i = 0
    while line != "" and i < 1200000:
        print(f"Line {i} out of 1.200.000")
        text.append([t.text for t in tokenizer(line.strip())])
        line = f.readline()
        i+=1

print("corpus loaded successfully\n\nfit corpus")
corpus = Corpus()
corpus.fit(corpus=text, window=5)


print("save corpus")
corpus.save('vectors/glove_corpus.model')

print("corpus saved.\n\nTrain model")
seed = 1
glove = Glove(no_components=300, 
              learning_rate=0.05, 
              random_state=seed,
              alpha=0.75,
              max_count=100)
glove.fit(corpus.matrix, epochs=100, no_threads=4, verbose=False)

print("model trained.\n\nadd dictionary")

glove.add_dictionary(corpus.dictionary)

print("dictionary added.\n\nsave vectors")
glove.save('vectors/glove_300D.model')
print("vectors saved\n\nprocess finished")


glovy = Glove().load("vectors/glove_300D.model")
copry = Corpus().load("vectors/glove_corpus.model")

len(glovy.dictionary)

glovy.most_similar('difficile')

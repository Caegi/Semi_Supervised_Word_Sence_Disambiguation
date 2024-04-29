"""## Classification"""

#import fasttext
from spacy.lang.fr import French
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import sparknlp
from sparknlp import base
from sparknlp import annotator
from sparknlp.base import *
from sparknlp.annotator import *
import pandas as pd
import fasttext
from DataExtraction import *
import gzip
import shutil

"""## Fasttext"""

# get the pre-trained fast text embeddings for French
input_file = 'content/cc.fr.300.bin.gz'
output_file = 'content/cc.fr.300.bin'

# unzip gzip file
with gzip.open(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# load the model of unzipped file
ft = fasttext.load_model('content/cc.fr.300.bin')

"""## GloVe"""

# !wget https://nlp.stanford.edu/data/glove.840B.300d.zip
# !unzip glove.840B.300d.zip -d /content

# !unzip glove_840B_300.zip -d /content

# embeddings = WordEmbeddings.pretrained("glove_6B_300", "xx") \
# .setInputCols("sentence", "token") \
# .setOutputCol("embeddings")

# embeddings_dict = {}
# with open("/content/glove_840B_300.zip", 'r') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vector = np.asarray(values[1:], "float32")
#         embeddings_dict[word] = vector

# id_to_sense: dictionary --> key = lemma, value = list of senses
# sense_to_id: dictionary --> key = lemma, values = dictionary --> key = sense, value = index of sense in list id_to_sense[lemma]

extractor = DataExtraction()
df = extractor.load_saved_file()
senses = set(df.word_sense.tolist())

id_to_sense = {}
sense_to_id = {}

# compute id_to_sense
for s in senses:
  lemma = s.split("_")[-1]
  if lemma not in id_to_sense:
    id_to_sense[lemma] = []

  id_to_sense[lemma].append(s)


# compute sense_to_id
for l in id_to_sense:
  if l not in sense_to_id:
    sense_to_id[l] = {}

  for count, s in enumerate(id_to_sense[l]):
    sense_to_id[l][s] = count



print("id_to_sense", id_to_sense["aboutir"])
print("sense_to_id", sense_to_id["aboutir"])

def get_x_y(data, lemma):

  # initialize X and y as zero array and empty list
  X = np.zeros((len(data), 300))
  y = []

  # iterate through the sentences and update X and y
  for count, doc in enumerate(tokenizer.pipe(data.sentence.tolist())):
    X[count, :] = ft.get_sentence_vector(doc.text)
    y.append(sense_to_id[lemma][data.word_sense[count]])

  return X, y

# Cross validation Classification

def cv_classification(X, y, lemma):

  # train classifier
  if len(set(y)) > 1:
    cv_classif = LogisticRegression(solver="liblinear").fit(X, y)
    scores = cross_val_score(cv_classif, X, y, cv=2)
    return round(scores.mean(),2)

  else:
    return float("nan")

# Traditional Classification
def traditional_classification(X_train,X_test,y_train, y_test, lemma):

  # train classifier
  if len(set(y_train)) > 1:
    classifier = LogisticRegression(solver = "liblinear").fit(X_train, y_train)
    return round(classifier.score(X_test, y_test), 2)

  else:
    return float("nan")

# prepare tokenizer
nlp = French()
tokenizer = nlp.tokenizer

trad_classif = []
cv = []

for lemma in id_to_sense.keys():

  # compute data frame with only one lemma
  data = df[(df['lemma'] == lemma)].reset_index()

  # split data into test and train
  X, y = get_x_y(data, lemma)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) # shuffles data by default before splitting

  # train classifiers with different methods
  trad_classif.append(traditional_classification(X_train, X_test, y_train, y_test, lemma))
  cv.append(cv_classification(X, y, lemma))

order = ["trad", "cv"]
highest = {"trad" :0, "cv":0}
for i in range(len(cv)):
  max_value = np.argmax(np.array([trad_classif[i], cv[i]]))
  highest[order[max_value]] += 1

print(highest)
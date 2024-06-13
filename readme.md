# Semi Supervised Word Sense Disambiguation

What is the project about?

## How to use
Every functionality can be used with a command in the terminal. Before you start, make sure to add two files to the parent directory of our project: 

1. The file containing FastText embeddings: It is called cc.fr.300.bin and can be downloaded <a href="https://fasttext.cc/docs/en/crawl-vectors.html">here</a>. Just search for French and click on bin. 
2. The folder with the trained models: It is named trained_models and is included in the zip file we handed in. 

With this done, everything else you need to know are the commands for the terminal. Here is an overview over all the options: 

|Command | Action |
|--------|--------|
help, -h |Display the online help we provided|
|--wsi, -i| Show our clustering results for the different embeddings|
|--wsd, -d|Show our classification results for the different embeddings and the split method. WARNING: This takes around 1h to execute!|
|--compare, -c|Show the clustering results and the classification results to compare them. WARNING: This takes around 15 minutes to execute!|

The next three options have to be used together. With them, you can provide the model with a sentence and a lemma and it predicts the lemma's sense used in the sentence: 

|Command | Action |
|--------|--------|
|--sentence, -s| Provide a the sentence to be evaluated between quotes|
|--lemma, -l| Provide the lemma that should be looked at in the sentence. This is used to select the right pretrained model|
|--mode, -m| Provide either "wsd" or "wsi" depending on which method you want to use to predict the sense|

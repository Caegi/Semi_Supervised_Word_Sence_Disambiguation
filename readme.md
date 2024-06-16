# Semi Supervised Word Sense Disambiguation

This project implements Word Sense Disambiguation and Word Sense Induction. It compares both ways of disambiguating words within a context.

Link to our project on <a href="https://github.com/Caegi/Semi_Supervised_Word_Sence_Disambiguation">GitHub</a> 

## How to use
Every functionality can be used with a command in the terminal. Before you start, make sure to add two files to the parent directory of our project: 

1. The file containing FastText embeddings: It is called cc.fr.300.bin and can be downloaded <a href="https://fasttext.cc/docs/en/crawl-vectors.html">here</a>. Just search for French and click on bin. 
2. The two folders with the trained models and the trained clusters: The folder containing the classification models is named trained_models and that for the clustering models is named trained_kmeans. Both are included in the zip file we handed in. 

With this done, everything else you need to know are the commands for the terminal. Here is an overview over all the options: 

|Command | Action |
|--------|--------|
help, -h |Display the online help we provided|
|--wsi, -i| Show our clustering results for the different embeddings|
|--wsd, -d|Show our classification results for the different embeddings and the split method. WARNING: This takes around 1h to execute!|
|--compare, -c|Show the clustering results and the classification results to compare them.|
|--decrease, -dv| Shows the results of the classification with decreasing training examples. WARNING: This takes around 1h to execute!|
|--increase, -ic| See how many examples should be added as constraints in order for Kmeans to achieve a better quality than a WSD classifier|
|--verbs, -v|Show all available verbs we trained our models on. There are 66 verbs in total.|

The next three options have to be used together. With them, you can provide the model with a sentence and a lemma and it predicts the lemma's sense used in the sentence: 

|Command | Action |
|--------|--------|
|--sentence, -s| Provide a the sentence to be evaluated between quotes|
|--lemma, -l| Provide the lemma that should be looked at in the sentence. This is used to select the right pretrained model|
|--mode, -m| Provide either "wsd" or "wsi" depending on which method you want to use to predict the sense. The default mode is WSI.|

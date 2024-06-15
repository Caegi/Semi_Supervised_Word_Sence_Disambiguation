import argparse

def online_help():
    help_description = ("This program implements Word Sense Disambiguation (WSD) and Word Sense Induction (WSI)." +
                        "You can either look at our results or ask the model to predict the sense by providing it with a sentence and a lemma.")
    
    parser = argparse.ArgumentParser(description=help_description)

    parser.add_argument("-d", "--wsd", required=False, action = "store_true", help="Execute only WSD tests")
    parser.add_argument("-i", "--wsi", required=False, action = "store_true", help="Execute only WSI tests")
    parser.add_argument("-c", "--compare", required=False, action = "store_true", help="Execute the comparison")
    parser.add_argument("-v", "--verbs", required=False, action="store_true", help="Shows all available verbs for the test sentences.")
    parser.add_argument("-dc", "--decrease", required=False, action="store_true", help="Shows the result of decreasing training examples for classification.")
    parser.add_argument("-ic", "--increase", required=False, action = "store_true", help="See how many examples should be added as constraints in order for Kmeans to achieve better quality than a WSD classifier")
    parser.add_argument("-s", "--sentence", type=str, help="Sentence you want to analyze. Can only be used with lemma")
    parser.add_argument("-l", "--lemma", type=str, help="Word in the sentence you want to analyze. Can only be used with sentence")
    parser.add_argument("-m", "--mode", type=str, help="indicate which method to use to analyze the sentence. Available are: 'wsd' and 'wsi'.")
    

    args = parser.parse_args()

    return(args)
        

online_help()
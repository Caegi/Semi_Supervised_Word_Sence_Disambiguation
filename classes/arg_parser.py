import argparse

def online_help():
    help_description = ("This program implements Word Sense Disambiguation (WSD) and Word Sense Induction (WSI).")
    parser = argparse.ArgumentParser(description=help_description)

    parser.add_argument("-d", "--wsd", required=False, action = "store_true", help="Execute only WSD tests")
    parser.add_argument("-i", "--wsi", required=False, action = "store_true", help="Execute only WSI tests")
    parser.add_argument("-c", "--compare", required=False, action = "store_true", help="Execute the comparison")
    parser.add_argument("-s", "--sentence", type=str, help="Sentence you want to analyze. Can only be used with lemma")
    parser.add_argument("-l", "--lemma", type=str, help="Word in the sentence you want to analyze. Can only be used with sentence")

    args = parser.parse_args()

    return(args)
        

online_help()
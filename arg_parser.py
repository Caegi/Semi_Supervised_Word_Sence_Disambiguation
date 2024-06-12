import argparse

def aide_en_ligne():
    help_description = ("This program implements Word Sense Disambiguation (WSD) and Word Sense Induction (WSI).")
    parser = argparse.ArgumentParser(description=help_description)

    #parser.add_argument("-t", "--WSI", required=False, action = "store_true", help="Commande pour exécuter les tests")


    args = parser.parse_args()

    return(args)
        

aide_en_ligne()
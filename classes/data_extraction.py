# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET # to extract data from XML file
import pandas as pd
import re
import tarfile

class DataExtraction: 

  def extract_files():
    file = tarfile.open('content/FSE-1.1-10_12_19.tar.gz')  # source: http://www.llf.cnrs.fr/dataset/fse/FSE-1.1-10_12_19.tar.gz 
    # extract file 
    file.extractall('./content') 
    file.close() 

  def get_df_from_xml(self, filepath_sent, filepath_ws):
      """
      Get a pandas dataframe containing sentences, their target words (words with multiple meanings),
      their lemma, their position in the sentence, and their word sense in the current context

      Parameters:
        filepath_sent: file with the xml file containing the sentences
          and the targets words to disambiguate
        filepath_ws: text file containing the gold clases, i.e. the words with
          multiples meaning and what word sense is used in the sentence

      Returns:
          pandas dataframe
      """
      tree = ET.parse(filepath_sent)
      root = tree.getroot()  # root = corpus
      l_sent = []
      l_target = []
      l_pos = []

      for sentence_block in root.iter("sentence"):
          s, targets = self.get_sentence_and_targets(sentence_block)
          self.update_lists(l_pos, l_sent, l_target, s, targets)

      l_word_sense, l_lemma = self.get_word_sense_and_lemma(filepath_ws)

      return pd.DataFrame({"sentence": l_sent, "target": l_target, "lemma": l_lemma,
                          "position": l_pos, "word_sense": l_word_sense})

  def get_sentence_and_targets(self, sentence_block):
      """
      get the sentence and the target words from a sentence xml "block"

      Parameters:
        sentence_block: "block" of xml containing a tag(its name),
        a text(what it contains), and children

      Returns:
        s: str
        targets: list[str]
      """
      s = []
      targets = [] # this is a list to account for sentences with multiple targets
      for child in sentence_block:
          s.append(child.text)

          if (child.tag == "instance"):
              targets.append(child.text)

      s = " ".join(s)
      return s, targets

  def update_lists(self, l_pos, l_sent, l_target, s, targets):
      """
      add values to l_pos, l_sent, l_target with the values s and targets

      Parameters:
        l_pos: list[ tuple(int, int) ], list of the indices of the target words in the sentences they are in
        l_sent: list[str], list of sentences
        l_target: list[str] list of target words (words to disambiguate)
        s: str sentence
        targets: list[str]

      Returns:
        None
      """
      for target in targets:
          l_sent.append(s)

          l_target.append(target)

          target_start = s.index(target)
          target_end = target_start + len(target) - 1
          l_pos.append( (target_start, target_end) )

  def get_word_sense_and_lemma(self, filepath_ws):
    """
    Parameters:
      filepath_ws: textfile containing the target words appearing in the xml file
      in order and their gold labels (word sense in the context of the sentence)

    Returns:
      l_word_sense: list[str]´
      l_lemma: list[str]
    """
    with open(filepath_ws) as file:
      content = file.read()

      pattern_word_sense = re.compile("ws_[0-9]_.*?(?=_)|ws_[0-9]\w_.*?(?=_)")
      l_word_sense = re.findall(pattern_word_sense, content, flags=0)

      pattern_lemma = re.compile("(?<=ws_[0-9]_).*?(?=_)|(?<=ws_[0-9]\w_).*?(?=_)")
      l_lemma = re.findall(pattern_lemma, content, flags=0)

      return l_word_sense, l_lemma

  def extract_data(self):
    # get_df_from_xml("/content/FSE-1.1.data.xml", "/content/FSE-1.1.gold.key.txt")
    df = self.get_df_from_xml("content/FSE-1.1-191210/FSE-1.1.data.xml", "content/FSE-1.1-191210/FSE-1.1.gold.key.txt")
    df.to_csv('fse_data.csv', index=False)

  def load_saved_file(self):
    return pd.read_csv('fse_data.csv')  
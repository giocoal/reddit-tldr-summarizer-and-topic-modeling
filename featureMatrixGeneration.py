#----------------------------- library import ----------------------------

from pandarallel import pandarallel
import time
import pandas as pd
import numpy as np
import os
import text_summarization_utility as tsu

#---------------------------------- Main ---------------------------------
if __name__ == "__main__":

#--------------------------- parallelization setup------------------------
  pandarallel.initialize(
    nb_workers=4,
    progress_bar=True
  )

#----------------------------- path definition ---------------------------
  root_path = 'Processed Data For Summarization'
  jsons_name = os.listdir(root_path)

  saving_path = 'Feature Matrices'
  csvs_name = os.listdir(saving_path)

  broken_files = ['train_4_2.json', 'train_6_0.json']


#----------------------------- matrix generation -------------------------
  for file_name in jsons_name:
    print(file_name)
    if (''.join([file_name[:-4],'csv']) not in csvs_name) and file_name not in broken_files:

      print('/'.join([root_path, file_name]))
      st = time.time()
      data = pd.read_json('/'.join([root_path, file_name]), orient="records", lines=True)
      ed = time.time()
      print(f'time elapsed {ed-st}')

      print('\nremoving non summarized element')
      st = time.time()
      data_ = data[data.apply(tsu.sum_is_not_zero, axis=1)]
      ed = time.time()
      print(f'time elapsed {ed-st}')

      print('\nremoving one sentence documents')
      st = time.time()
      data_ = data_[data_.apply(tsu.more_than_one_sentence, axis=1)]
      ed = time.time()
      print(f'time elapsed {ed-st}')

      print('\ncreating feature matrix')
      st = time.time()
      dfs = data_[['document_normalized','pos_tags', 'ext_labels']].parallel_apply(tsu.sentence_feature_matrix, axis=1)
      ed = time.time()
      print(f'time elapsed {ed-st}')

      print('\nsaving csv')
      st = time.time()
      pd.concat(list(dfs), ignore_index=True).to_csv('/'.join([saving_path, ''.join([file_name[:-4],'csv'])]))
      ed = time.time()
      print(f'time elapsed {ed-st}'+'\n')

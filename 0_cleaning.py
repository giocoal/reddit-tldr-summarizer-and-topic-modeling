# base
import os
import json
import pandas as pd
#import pprint
import seaborn as sns
import numpy as np 
from collections import Counter 
from matplotlib import pyplot as plt
#import statistics
import gc

def import_corpus(files):
    
    test = pd.DataFrame()
    train = pd.DataFrame()
    val = pd.DataFrame()
    
    for file in files:
        temp = pd.read_json(file, lines=True)
        temp.set_index("id", inplace=True)
            
        if "test" in file:
            test = pd.concat([test, temp])
        elif "train" in file:
            train = pd.concat([train, temp])
        else:
            val = pd.concat([val, temp])
    
    corpus = {'train': train.copy(deep=True),
                'val': val.copy(deep=True),
                    'test': test.copy(deep=True)}
    return corpus

if __name__ == "__main__":
    n_split = 10
    print("Cleaning ...")
    # Importazione
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk("Dataset_TLDRHQ/") for f in filenames]
    corpus = import_corpus(files)
    
    # Reset indici
    for key in list(corpus.keys()):
        # Roporta id come colonna
        corpus[key].reset_index(inplace=True)
        
    # Cleaning
    
    for key in list(corpus.keys()):
        # remove duplicates from train, val and test
        count = corpus[key].duplicated(subset = ["document"]).sum()
        print(f"{key} - number of duplicated documents:{count}/{corpus[key].shape[0]}")
        corpus[key].drop_duplicates(subset = ["document"], inplace=True)
        
    for key in list(corpus.keys()):
        count = corpus[key].duplicated(subset = ["summary"]).sum()
        print(f"{key} - number of duplicated sumarries:{count}/{corpus[key].shape[0]}")
        # remove duplicates from train, val and test
        corpus[key].drop_duplicates(subset = ["summary"], inplace=True)
        
    # Saving
    print("Saving ...")
    corpus["val"].to_json(f"Dataset_splitted/val.json", orient="records", lines=True)
    corpus["test"].to_json(f"Dataset_splitted/test.json", orient="records", lines=True)
    
    chunk_size = int(corpus["train"].shape[0] / n_split)
    for i in range(n_split):
        start_index = i*chunk_size
        end_index = (i+1)*chunk_size
        if i == (n_split - 1):
            # For last chunk, use the remaining rows
            end_index = corpus["train"].shape[0]
        chunk = corpus["train"][start_index:end_index]
        file_name = "train_{}.json".format(i+1)
        # Save chunk to a JSON file
        chunk.to_json(f"Dataset_splitted/train_{i+1}.json", orient="records", lines=True)
    
# base
import os
import json
import pandas as pd
import pprint
import seaborn as sns
import numpy as np 
from collections import Counter 
from matplotlib import pyplot as plt
import statistics
from IPython.display import clear_output

import multiprocessing
from functools import partial
from multiprocessing import  Pool

from tqdm import tqdm

# nlp
from num2words import num2words

import string

from bs4 import BeautifulSoup 
import time

from textblob import TextBlob

from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from pygments.lexers import guess_lexer

import emoji
import demoji

import re
import contractions
from contractions import contractions_dict

import nltk
from nltk.tokenize import word_tokenize 
#nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
#nltk.download('wordnet')
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora

# download

# Scarica il corpus delle stopwords inglesi
#nltk.download('stopwords')
# Ottieni la lista delle stopwords inglesi
stop_words_en = stopwords.words('english')

# TEXT NORMALIZATION

def split_string(string):
    return string.split("</s><s>")

def remove_html_tags(sentence):
    pattern = re.compile("<.*?>") 
    cleaned_sentence = re.sub(pattern,'',sentence).strip()
    return cleaned_sentence

def remove_html_entities(sentence):
    pattern = re.compile("&[a-z0-9]+|&#[0-9]{1,6}|&#x[0-9a-f]{1,6}")
    cleaned_sentence = re.sub(pattern,'',sentence).strip()
    return cleaned_sentence

def remove_extra_whitespaces(text):
    cleaned_sentence = re.sub(r'^\s*|\s\s*', ' ', text).strip()
    return cleaned_sentence

def remove_urls(sentence):
    http_pattern = re.compile(r"http\S+") 
    cleaned_sentence = re.sub(http_pattern,'',sentence).strip()
    www_pattern = re.compile(r"www\S+") 
    cleaned_sentence = re.sub(www_pattern,'',cleaned_sentence)
    return cleaned_sentence

def remove_emoji(phrase):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    cleaned_sentence = emoji_pattern.sub(r'', phrase)
    return cleaned_sentence

# Conversione ordinali 1st 2nd etc in words
def replace_ordinal_numbers(text):
    re_results = re.findall('(\d+(st|nd|rd|th))', text)
    for enitre_result, suffix in re_results:
        num = int(enitre_result[:-len(suffix)])
        text = text.replace(enitre_result, num2words(num, ordinal=True))
    return text

# Replace numbers with words (lone numbers)
def replace_numbers(phrase):
    cleaned_sentence = re.sub(r"\s\d+\s", lambda x: f" {num2words(x.group())} ", phrase)
    return cleaned_sentence

# Rimuovi i rimanenti
def remove_numbers(phrase):
  cleaned_sentence=re.sub(r'\d+', '',phrase)
  return cleaned_sentence

# Funzione finale
def process_numbers(phrase):
    phrase = replace_ordinal_numbers(phrase)
    phrase = replace_numbers(phrase)
    phrase = remove_numbers(phrase)
    return phrase

def remove_reddit_tags(phrase):
  cleaned_sentence = re.sub(r'(\\ u \\|u \/|r \/|\\ r \\)', '',phrase)
  return cleaned_sentence

def remove_control_chars(phrase):
    cleaned_sentence = re.sub(r'(\\ u |\\ n |\\ v |\\ m | \\ )', ' ',phrase)
    return cleaned_sentence

def process_age(phrase):
    cleaned_sentence = re.sub(r'(\\ u |\\ n |\\ v |\\ m | \\ )', ' ',phrase)
    return cleaned_sentence

# WORD PROCESSING

def to_lower(phrase):
  phrase = phrase.lower()
  return phrase

def character_repeatation(text):
    # Pattern matching for all case alphabets
    # \1   It refers to the first capturing group.
    # {2,} It means we are matching for repetition that occurs more than two times (or equal).
    # r’\1\1' → It limits all the repetition to two characters.
    Pattern_alpha = re.compile(r"([A-Za-z])\1{2,}", re.DOTALL)
    # Limiting all the  repeatation to two characters.
    Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
    # Pattern matching for all the punctuations that can occur
    Pattern_Punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    # Limiting punctuations in previously formatted string to only one.
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    return Combined_Formatted

def fix_and_expand_eng_contradictions(phrase):
    # Remove whitespaces before '
    phrase = re.sub(r"\s+'", "'", phrase)
    # Avvicina parole tipo "was n't" a "wasn't" 
    phrase = re.sub(r"\s+n't", "n't", phrase)
    phrase = contractions.fix(phrase, slang=True)
    return phrase

def correct_me(text):
  textBlb = TextBlob(text)        
  textCorrected = str(textBlb.correct())   # Correcting the text
  return textCorrected

def remove_special_characters_punctuations(sentence):
    cleaned_text  = re.sub(r"[^a-zA-Z]+",' ',sentence).strip()
    return cleaned_text

def text_normalization(sentences):
  # Text Cleaning
  sentences = list(map(remove_html_tags, sentences))
  sentences = list(map(remove_html_entities, sentences))
  sentences = list(map(remove_extra_whitespaces, sentences))
  sentences = list(map(remove_urls, sentences))
  sentences = list(map(remove_emoji, sentences))
  sentences = list(map(process_numbers, sentences))
  sentences = list(map(remove_reddit_tags, sentences))
  sentences = list(map(remove_control_chars, sentences))
  sentences = list(map(process_age, sentences))
  sentences = list(map(remove_extra_whitespaces, sentences))
  # Word Processing
  sentences = list(map(to_lower, sentences))
  sentences = list(map(character_repeatation, sentences))
  sentences = list(map(fix_and_expand_eng_contradictions, sentences))
  sentences = list(map(correct_me, sentences))
  sentences = list(map(remove_special_characters_punctuations, sentences))
  return sentences

# WORD ANALYSIS

def stop_words_1char_removal(text, stopwords_list = stopwords.words('english')):
  return ([word for word in text if (word not in stopwords_list) and (len(word) > 1)])

def lemmaSentence(token_words):
    lemma_text=[]
    for word in token_words:
        lemma_text.append(WordNetLemmatizer().lemmatize(word))
    return lemma_text

def text_Tokenization(sentences):
  # Tokenization
  return list(map(word_tokenize, sentences))

def text_stop_words_1char_removal(sentences):
  # Stopwords removal
  return list(map(stop_words_1char_removal, sentences))

def text_lemmatizer(senteces):
  return list(map(lemmaSentence, senteces))

### Multiprocessing ###

# def parallelize(data, func, num_of_processes=8):
#     data_split = np.array_split(data, num_of_processes)
#     pool = Pool(num_of_processes)
#     data = pd.concat(pool.map(func, data_split))
#     pool.close()
#     pool.join()
#     return data

# def run_on_subset(func, data_subset):
#     return data_subset.apply(func)

# def parallelize_on_rows(data, func, num_of_processes=8):
#     return parallelize(data, partial(run_on_subset, func), num_of_processes)

def parallelize_dataframe(df, func, num_processes = 8):
    df_split = np.array_split(df, num_processes)
    with multiprocessing.Pool(num_processes) as pool:
        df = pd.concat(pool.map(func, df_split))
    pool.close()
    return df

def parallelize_normalizer(df):
    df["document_normalized"] = df["document"].apply(text_normalization)
    return df

if __name__ == "__main__":
    # Definizione parametri multiprocessing
    cores = multiprocessing.cpu_count()
    num_process = cores - 3
    
    ### Import ###
    print("Importing, parsing, cleaning, sentences splitting ...\n")
    
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk("Dataset_TLDRHQ/") for f in filenames]
    #files = files[0:100] # subset composed of train, val, test

    test = pd.DataFrame()
    train = pd.DataFrame()
    val = pd.DataFrame()

    for file in files:
        temp = pd.read_json(file, lines=True)
        temp.set_index("id", inplace=True)
            
        if "test" in file:
            test = pd.concat([test, temp])
        if "train" in file:
            train = pd.concat([train, temp])
        else:
            val = pd.concat([val, temp])
        
    corpus = {'train': train,
                'val': val,
                    'test': test}
    
    ### Reset index ###
    
    for key,value in corpus.items():
        # Roporta id come colonna
        value.reset_index(inplace=True)
        # Rimozione colonne inutili
        # columns = ds.columns.tolist()
        # columns = [col for col in columns if col not in ["document", "id"]]
        # ds.drop(columns, axis=1, inplace=True)
        
    ### Cleaning ###
    
    for key, value in corpus.items():
        # remove duplicates from train, val and test
        value.drop_duplicates(subset = ["document"], inplace=True)
        
    ### Senteces Splitting ###
    
    for key, value in corpus.items():
        value["document"] = value["document"].apply(split_string)
    
    ### Test ###
    
    for key, value in corpus.items():
        value = value.head(10)
    
    ### Text Normalization ###
    
    for key, value in corpus.items():
        print(f"Text normalization for {key} ...\n")
        
        # Splittind corpus e multiprocessing
        
        value = parallelize_dataframe(df = value, 
                                        func = parallelize_normalizer, 
                                        num_processes = num_process) 


        value.to_json(f"ProcessedData/{key}_normalized.json", orient="records", lines=True)
        

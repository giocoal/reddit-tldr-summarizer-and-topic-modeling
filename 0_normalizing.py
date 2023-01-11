################# LIBRERIE #################

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
#from IPython.display import clear_output

import multiprocessing
#from functools import partial
from multiprocessing import  Pool

#from pandarallel import pandarallel

import swifter
# from swifter import set_defaults
# cores = multiprocessing.cpu_count()
# num_process = cores
# set_defaults(
#     npartitions=cores,
#     dask_threshold=1,
#     scheduler="processes",
#     progress_bar=True,
#     progress_bar_desc=None,
#     allow_dask_on_strings=False,
#     force_parallel=False,
# )

from tqdm import tqdm, tqdm_pandas

# nlp
from num2words import num2words

#import string

#from bs4 import BeautifulSoup 
#import time

from textblob import TextBlob

#from pygments import highlight
#from pygments.lexers import get_lexer_by_name
#from pygments.formatters import HtmlFormatter
#from pygments.lexers import guess_lexer

#import emoji
#import demoji

import re
import contractions
#from contractions import contractions_dict

import nltk
from nltk.tokenize import word_tokenize 
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
#nltk.download('wordnet')
from nltk.corpus import stopwords
#import gensim
#from gensim.utils import simple_preprocess
#import gensim.corpora as corpora

# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
# nltk.download('universal_tagset')

# download

# Scarica il corpus delle stopwords inglesi
#nltk.download('stopwords')
# Ottieni la lista delle stopwords inglesi
# stop_words_en = stopwords.words('english')

################# FUNZIONI #################


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
    try:
        phrase = replace_numbers(phrase)
        phrase = remove_numbers(phrase)
    except:
        phrase = remove_numbers(phrase)
    return phrase

def remove_reddit_tags(phrase):
  cleaned_sentence = re.sub(r'(\\ u \\|u \/|r \/|\\ r \\)', '',phrase)
  return cleaned_sentence

def remove_control_chars(phrase):
    cleaned_sentence = re.sub(r'(\\ u |\\ n |\\ v |\\ m | \\ )', ' ',phrase)
    return cleaned_sentence

def process_age(phrase):
    cleaned_sentence = re.sub(r'(\b\d{2})f\b', r'\1 female',phrase)
    cleaned_sentence = re.sub(r'(\b\d{2})m\b', r'\1 male',phrase)
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
  #sentences = list(map(remove_reddit_tags, sentences))
  #sentences = list(map(remove_control_chars, sentences))
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

def text_normalization_full(sentences):
  # Text Cleaning
  try: 
    sentences = list(map(remove_html_tags, sentences))
    sentences = list(map(remove_html_entities, sentences))
    sentences = list(map(remove_extra_whitespaces, sentences))
    sentences = list(map(remove_urls, sentences))
    sentences = list(map(remove_emoji, sentences))
    sentences = list(map(process_numbers, sentences))
    #sentences = list(map(remove_reddit_tags, sentences))
    #sentences = list(map(remove_control_chars, sentences))
    sentences = list(map(process_age, sentences))
    sentences = list(map(remove_extra_whitespaces, sentences))
    # Word Processing
    sentences = list(map(to_lower, sentences))
    sentences = list(map(character_repeatation, sentences))
    sentences = list(map(fix_and_expand_eng_contradictions, sentences))
    #sentences = list(map(correct_me, sentences))
    sentences = list(map(remove_special_characters_punctuations, sentences))
    # Word analysis
    sentences = list(map(word_tokenize, sentences))
    sentences = list(map(stop_words_1char_removal, sentences))
    sentences = list(map(lemmaSentence, sentences))
  except:
    print(sentences)
  return sentences

def extract_tags(document_tags: list):
  doc_tags = pd.Series(document_tags)
  doc_tags = doc_tags.apply(lambda subList: pd.Series(subList))
  doc_tags = doc_tags.applymap(lambda wordTagTuple: wordTagTuple[1] if type(wordTagTuple)==tuple else '')
  return doc_tags.values

def POS_tagging(document: list, tagset:str = 'universal', lang:str='eng'):
  POS_tags = nltk.tag.pos_tag_sents(document, tagset=tagset, lang=lang)
  POS_tags = extract_tags(POS_tags)
  return [list(filter(None, l.tolist())) for l in POS_tags]

################# MAIN #################
if __name__ == "__main__":
    # Definizione parametri multiprocessing
    #print(f"Core utilizzati:{num_process} \n")
    
    ### Import ###
    print(" Starting processing ...\n")
    prefix = "./Dataset_splitted/"
    #files = [prefix+f for f in os.listdir(prefix)]
    files = sorted(os.listdir(prefix))
    
    print(files)

    for file in files:
        print(f"Importing, parsing, sentences splitting, text normalization, tokenization, stop words removal, lemmatization and POS for {file} ...\n")
        ds = pd.read_json(f'./Dataset_splitted/{file}', orient="records", lines=True)
        print(f"Dimensioni: {ds.shape}")

        # Sentence splitting
        ds["document"] = ds["document"].apply(split_string)

        # Normalizing in multiprocessing
        ds["document_normalized"] = ds["document"].swifter.apply(text_normalization_full)
        
        # POS
        print(f"POS tagging {file} ...\n")
        ds["pos_tags"] = ds["document_normalized"].swifter.apply(POS_tagging)
        
        # SAVING
        print(f"Saving: ./Dataset_splitted/{file} \n")
        ds.to_json(f"ProcessedData/{file}", orient="records", lines=True)
        
        
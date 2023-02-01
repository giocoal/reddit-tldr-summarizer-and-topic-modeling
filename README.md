# Extreme Extractive Text Summarization and Topic Modeling over Reddit Posts from TLDRHQ dataset.

<img src="https://img.shields.io/badge/Complete-Done-red" alt="Soon"/> 
<img src="https://img.shields.io/badge/NLP-Topic%20Modeling-blue" alt="Soon"/>
<img src="https://img.shields.io/badge/NLP-Text%20Summarization-blue" alt="Soon"/>
<img src="https://img.shields.io/github/issues/giocoal/reddit-tldr-summarizer-and-topic-modeling" alt="issu"/>
<img src="https://img.shields.io/github/license/giocoal/reddit-tldr-summarizer-and-topic-modeling" alt="lisen"/>

## Table of contents
* [Abstract](#abstract)
* [Requirements](#requirements)
* [TLDRHQ: Data and Text Pre-processing](#tldrhq-data-and-text-pre-processing)
* [Extreme Extractive Text Summarization](#tools)
* [Screenshots](#screenshots)
* [Status](#status)
* [Contact](#contact)
* [License](#license)
* [Contributing](#contributing)

## Abstract

Reddit is a social news aggregation and discussion website where users post content (such as links, text posts,
images, and videos) on a wide range of topics in domain-specific boards called ”communities” or ”subreddits.”
The following project aims to implement text summarization and topic modelling pipelines on the textual
content of Redditors’ posts. The dataset used to achieve these goals is the Reddit-based TL;DR summarization
dataset TLDRHQ[1] containing approximately 1.7 million Reddit posts (submissions as well as comments)
obtained through scraping techniques. Each instance in the dataset includes a Reddit post and its TL;DR, which
is an acronym for ”Too Long; Didn’t Read” and is an extremely short summary of the post’s content that is good
practice for users to leave at the end of a post. While the Reddit usage has increased, the practice of write TL;DR
didn’t keep the pace. In this context, a system (such as a bot) capable to automatically generate the TL;DR of
a post could improve Reddit usability. However, the high abstractivity, heterogeneity and noisy of posts make
the text summarization task challenging. In this work a supervised extreme extractive summarization model
is developed. Despite its lower complexity, results show that its performance are not so different with respect
to the state of the art BertSumExt. Moreover the topic modeling analysis of the posts could be really useful in
identifying the hidden topics in a post and evaluate if it’s published in the right subreddit. In this project LSA and
LDA techniques are used. On this dataset LSA outperformed LDA and identified 20 well defined topics providing
the respective document-topics and topic-terms matrices

## Requirements

- python 3.10.7
- contractions==0.1.73
- gensim==4.3.0
- ipython==8.8.0
- matplotlib==3.6.3
- nltk==3.8.1
- num2words==0.5.12
- numpy==1.22.1
- pandas==1.3.5
- seaborn==0.12.2
- simplemma==0.9.0
- spacy==3.4.4
- swifter==1.3.4
- textblob==0.17.1
- tqdm==4.64.1
- scikit-learn==1.2.0
- rouge-score==0.1.2
- imbalanced-learn==0.10.1
- wordcloud==1.8.2.2
- pyLDAvis==3.3.1

## TLDRHQ: Data and Text Pre-processing

### Step 0. Prepare Folders

First of all, create three empty folders: `./DatasetTLDRHQ`,`./ProcessedData` and `./Dataset_splitted`.

### Step 1. Download and extract the dataset

Download annotations from the [official Google Drive Folder](https://drive.google.com/file/d/1jCi0Mn0k-pid5SSTafov11-e1A9LEZed/view?usp=sharing) and extract them in `./DatasetTLDRHQ`, resulting in a folder tree like this:

```
project_folder
└───Dataset_TLDRHQ
    ├───dataset-m0
    ├───dataset-m1
    ├───dataset-m2
    ├───dataset-m2021
    ├───dataset-m3
    ├───dataset-m4
    └───dataset-m6
```

### Step 2. Perform data cleaning and splitting of the dataset
Run the `0_cleaning.py` script which will perform data cleaning (removing duplicates), splits the dataset into training/validation and test sets (splitting the training set so that it is easier to manage) and then save it splitted into `.JSON` files in `./Dataset_splitted`. You get a directory tree like this:
```
project_folder
└───Dataset_splitted
    ├───test.json
    ├───train_1.json
    ├───train_10.json
    ├───train_2.json
    ├───train_3.json
    ├───train_4.json
    ├───train_5.json
    ├───train_6.json
    ├───train_7.json
    ├───train_8.json
    ├───train_9.json
    └───val.json
```


### Step 3. Perform text pre-processing on the dataset
Run the `0_normalizing.py` script which will perform senteces splitting, text normalization, tokenization, stop-words removal, lemmatization and POS tagging on `document` variable, containing reddit posts. Then save it splitted into various `.JSON` files in `./ProcessedData`. You get a directory tree like this:
```
project_folder
└───ProcessedData
    ├───test.json
    ├───train_1.json
    ├───train_10.json
    ├───train_2.json
    ├───train_3.json
    ├───train_4.json
    ├───train_5.json
    ├───train_6.json
    ├───train_7.json
    ├───train_8.json
    ├───train_9.json
    └───val.json
```
The text normalisation operations performed include, in order: Sentence Splitting, HTML tags and entities removal, Extra White spaces Removal, URLs Removal, Emoji Removal, User Age Processing (e.g. 25m becomes 25 male), Numbers Processing, Control Characters Removal, Case Folding, Repeated characters processing (e.g. reallllly becomes really), Fix and Expand English contradictions, Special Characters and Punctuation Removal, Tokenization (Uni-Grams), Stop-Words and 1-character tokens, Lemmatization and POS tagging.

## Extreme Extractive Summarization task

### Step 0. Split and clean'ProcessedData' for easy management
Run notebook 'Preprocessing for summarization.ipynb' in order to:
- remove document without summary
- remove document with a single sentence
- split train dataset 

```
project_folder
└───Processed Data For Summarization
    ├───test_0.json
    ├───test_1.json
    ├───test_2.json
    ├───train_1_0.json
    ├───train_1_1.json
    ├───train_1_2.json
    ├───train_2_0.json
    ├───train_2_1.json
    ├───train_2_2.json
    ├───  ...
    ├───train_8_0.json
    ├───train_8_1.json
    ├───train_8_2.json
    ├───train_9_0.json
    ├───val_0.json
    ├───val_1.json
    └───val_2.json
```

### Step 1. Create a feature matrix for each of the JSON in 'Processed Data For Summarization'
Run featureMatrixGeneration.py obtaining feature matrices (sentences x features). You get a directory tree like this:

```
project_folder
└───Feature Matrices
    ├───test_0.csv
    ├───test_1.csv
    ├───test_2.csv
    ├───train_1_0.csv
    ├───train_1_1.csv
    ├───train_1_2.csv
    ├───train_2_0.csv
    ├───train_2_1.csv
    ├───train_2_2.csv
    ├───  ...
    ├───train_8_0.csv
    ├───train_8_1.csv
    ├───train_8_2.csv
    ├───train_9_0.csv
    ├───val_0.csv
    ├───val_1.csv
    └───val_2.csv
```
    
 Run the notebook featureMatrixGeneration2.ipynb to join train, val and test datasets. You get a directory tree like this:
 
 ```
 project_folder
 └───Feature Matrices
    ├───test_0.csv
    ├───test_1.csv
    ├───test_2.csv
    ├───train_1_0.csv
    ├───train_1_1.csv
    ├───train_1_2.csv
    ├───train_2_0.csv
    ├───train_2_1.csv
    ├───train_2_2.csv
    ├───  ...
    ├───train_8_0.csv
    ├───train_8_1.csv
    ├───train_8_2.csv
    ├───train_9_0.csv
    ├───val_0.csv
    ├───val_1.csv
    ├───val_2.csv
    ├───test.csv
    ├───train.csv
    └───val.csv
```
    
Features generated at this step are the following:
- sentence_relative_positions
- sentence_similarity_score_1_gram
- word_in_sentence_relative
- NOUN_tag_ratio
- VERB_tag_ratio
- ADJ_tag_ratio
- ADV_tag_ratio
- TF_ISF
 
    
### Step 2. Perform CUR undersampling
Run notebook featureMatrixUndersampling.ipynb in order to perform CUR undersampling on both train and validation data sets. You get a directory tree like this:

```
project_folder
└───Undersampled Data
    ├───trainAndValMinorityClass.csv
    └───trainAndValMajorityClassUndersampled.csv
```

Majority and minority class are splitted because CUR undersampling works only on the majority class

### Step 3. Perform EditedNearestNeighbours(ENN) undersamplig
Run notebook featureMatrixAnalysis.ipynb to perform EEN undersampling. You get a directory tree like this:

```
project_folder
└───Undersampled Data
    ├───trainAndValUndersampledENN3.csv
    ├───trainAndValMinorityClass.csv
    └───trainAndValMajorityClassUndersampled.csv
```

### Step 4. Machine Learning model selection and evaluation
Run notebook featureMatrixAnalysis.ipynb to perform a RandomizedSearcCV over the following models
- RandomForestClassifier
- LogisticRegression
- HistGradientBoostingClassifier

with a few possible parameters configuration. 

Then, evaluate the resulting best model on the test set with respect to:
- ROC curve
- Recall
- Precision
- Accuracy

### Step 5. Perform Maximal Marginal Relevance(MMR) selection
Run notebook featureMatrixAnalysis.ipynb to perform MMR and obtain an extractive summary for each document in the test set.

### Step 6. Summary Evaluation
Run notebook featureMatrixAnalysis.ipynb to measure summaries quality by means of 
- Rouge1
- Rouge2 
- RougeL 

## Topic modeling task
### Step 0. Perform preprocessing
Run the `Preprocessing for topic modeling.ipynb` script to process and extract only the useful data. The output is saved here:

```
project_folder
└───processed_dataset
    ├───test.json
```

### Step 1. Perform topic modeling on the test set
Run the `topic_modeling.ipynb` script which will perform LDA (with grid search of the best hyper-parameters) and LSA. The script saves 9 CSV files, 3 for LSA and 6 for LDA (UMass and CV coherence measures), containing: document-topic matrix, topic-term matrix and a table with topic insights.

```
project_folder
└───Results_topic_modeling
    ├───lda_doc_topic.csv
    ├───lda_doc_topic_CV.csv
    ├───lda_top_terms.csv
    ├───lda_top_terms_CV.csv
    ├───lda_topic_term.csv
    ├───lda_topic_term_cv.csv
    ├───lsa_doc_topic.csv
    ├───lsa_top_terms.csv
    ├───lsa_topic_term.csv
```
Saves images regarding the number of words per document and wordcloud in

```
project_folder
└───Images
```

Saves hyperparameters grid search results for UMass and CV coherence in
```
project_folder
└───Hyperparameters
    ├───tuning.csv
    ├───tuning_CV.csv
```

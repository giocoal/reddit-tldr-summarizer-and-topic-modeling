# Extractive Text Summarization and Topic Modeling over Reddit Posts

## Requirements

- python 3.10.7
- ipython 8.6.0
- beautifulsoup4==4.11.1
- contractions==0.1.73
- gensim==4.3.0
- ipython==8.8.0
- matplotlib==3.6.3
- nltk==3.8.1
- num2words==0.5.12
- numpy==1.22.1
- pandas==1.3.5
- Pygments==2.11.2
- seaborn==0.12.2
- simplemma==0.9.0
- spacy==3.4.4
- swifter==1.3.4
- textblob==0.17.1
- tqdm==4.64.1

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

## Model

## Testing

## Results
|Network     | One frame | Whole video  |
-------------|:--------------:|:----:|
|Spatial ResNet-50     |39.0%           |45.0% |
|Temporal    |25.8%           |NA% |
|Fusion      |38.0%           |NA% |

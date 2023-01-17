# Extractive Text Summarization and Topic Modeling over Reddit Posts

## Requirements

- python 3.10.7
- ipython 8.6.0
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

## Summarization task

### Step 0. Split and clean'ProcessedData' for easy management
Run notebook 'Preprocessing for summarization.ipynb' in order to:
- remove document without summary
- remove document with a single sentence
- split train dataset 

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
    
### Step 1. Create a feature matrix for each of the JSON in 'Processed Data For Summarization'
Run featureMatrixGeneration.py obtaining feature matrices (sentences x features). You get a directory tree like this:
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
    
 Run the notebook featureMatrixGeneration2.ipynb to join train, val and test datasets. You get a directory tree like this:
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
project_folder
└───Undersampled Data
    ├───trainAndValMinorityClass.csv
    └───trainAndValMajorityClassUndersampled.csv

Majority and minority class are splitted because CUR undersampling works only on the majority class

### Step 3. Perform EditedNearestNeighbours(ENN) undersamplig
Run notebook featureMatrixAnalysis.ipynb to perform EEN undersampling. You get a directory tree like this:
project_folder
└───Undersampled Data
    ├───trainAndValUndersampledENN3.csv
    ├───trainAndValMinorityClass.csv
    └───trainAndValMajorityClassUndersampled.csv

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

## Testing

## Results
|Network     | One frame | Whole video  |
-------------|:--------------:|:----:|
|Spatial ResNet-50     |39.0%           |45.0% |
|Temporal    |25.8%           |NA% |
|Fusion      |38.0%           |NA% |

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm, pinv
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import random
from scipy.linalg import interpolative

#------------------------------ sentence_relative_positions -----------------------
def sentence_relative_positions(sentence: list, document: list):
  doc = pd.Series(document)
  sentence_index = doc[doc.apply(lambda x: x==sentence)].index.to_list()[0]
  if len(doc)-1 == 0:
    return 0
  else:
    relative_index = sentence_index/(len(doc)-1)
    return relative_index

#------------------------------- word_in_sentence_relative ------------------------
def word_in_sentence_relative(sentence: list, document:list):
  doc = pd.Series(document)
  word_in_sentence = len(sentence)
  word_in_document = doc.apply(len).sum()
  return word_in_sentence/word_in_document

#------------------------------------ POS_tag_ratio --------------------------------
def POS_tag_ratio(sentence: list, tag_name: str, document: list, document_tags: list):
  doc = pd.Series(document)

  sentence_index = doc[doc.apply(lambda x: x==sentence)].index.to_list()[0]
  tags = document_tags[sentence_index]

  sentence_length = len(sentence)

  if sentence_length == 0:
    return 0
  else:
    tag_count = (pd.Series(tags)==tag_name).sum()
    return tag_count/sentence_length


#-------------------------- TF_ISF_and_Sentence_similarity_score_n_gram ----------------
def TF_ISF_and_Sentence_similarity_score_n_gram(document: list, n: int=1):
  doc = pd.Series(document)
  doc = doc.apply(lambda wordList: ' '.join(wordList))
  
  vectorizer = TfidfVectorizer(ngram_range=(n,n))
  X = vectorizer.fit_transform(doc)

  tf_isf_absolute = X.sum(axis=1)
  
  tf_isf_relative = tf_isf_absolute/max(tf_isf_absolute)

  sentence_similarity_absolute = np.array(list(map(lambda x: x.sum(), cosine_similarity(X))))
  sentence_similarity_relative = sentence_similarity_absolute/len(doc)

  return (np.array(tf_isf_relative).flatten(), sentence_similarity_relative)

#--------------------------------- sentence_feature_matrix --------------------------------
def sentence_feature_matrix(document_data: pd.Series, doc_col_names: str = 'document_normalized', tag_col_names: str = 'pos_tags', label_col_name: str = 'ext_labels'):
  doc = pd.Series(document_data[doc_col_names])
  doc_tag = document_data[tag_col_names]
  
  s = pd.DataFrame()
  s['document_index'] = [document_data.name]*len(doc)
  s['sentence_relative_positions'] = doc.apply(sentence_relative_positions, document = doc)
  s['word_in_sentence_relative'] = doc.apply(word_in_sentence_relative, document = doc)

  for tag_name in ['NOUN','VERB','ADJ', 'ADV']:
    s['POS_tag_ratio_'+tag_name] = doc.apply(POS_tag_ratio, tag_name = tag_name, document = doc, document_tags = doc_tag)
  
  for n in [1]:
    (s['tf_isf_'+str(n)+'_gram'], s['sentence_similarity_'+str(n)+'_gram']) = TF_ISF_and_Sentence_similarity_score_n_gram(doc, n)

  s[label_col_name] = document_data[label_col_name]

  return s


#-------------------------------- CUR Decomposition ----------------------------------
def ColumnSelect(A, k, c, v):
    # {
    # Input
    # - A: una matrice m x n.
    # - k: un parametro di rango k.
    # - c: numero di colonne che vogliamo selezionare da A.
    # - v: matrice n x k dei primi k vettori singolari destri di A.
    # Output
    # - C: matrice m x c' con c' colonne da A, c' <= c.
    # }

    random.seed(30)

    m,n = A.shape
    #------- Calcolo del normalized leverage scores di eqn. 3 di [1]. ---------
    pi = []
    for j in range(n):
        pi.append((norm(v[j,:])**2) / k )
        
    #--------------------------selezione colonne ------------------------------
    indexA = []
    for j in range(n):
        # la j-esima colonna di A è selezionata con probabilità prob_j.
        prob_j = min([1,c*pi[j]])   #trova il minimo tra 1 e c*pi(j)

        if prob_j==1: # se prob_j=1 seleziona la j-esima colonna di A
            indexA.append(j)
        elif prob_j > random.random():  # se prob_j<1, genera un numero casuale in [0,1]
            indexA.append(j) # poi se prob_j > rand, seleziona la j-esima colonna di A

    # Al termine indexA conterrà gli indici delle colonne selezionate da A
    C = A[:, indexA]
    return C, indexA


def CUR(A, k, c, r):
    # {
    # Input:
    # - A: matrice m x n.
    # - k: parametro di rango, con k << min(m,n).
    # - c: numero di colonne che vogliamo selezionare da A.
    # - r: numero di righe che vogliamo slezionare da A.
    # Ouput:
    # - C: matrice m x c' con c' colonne da A, c' <= c.)
    # - R: matrice r' x n con r' righe da A, r' <= r.
    # - U_pi: matrice c' x r'.
    # }

    # calcolo dei primi k vettori singolari destri di A and A'
    U, s, Vt = svd(A, full_matrices = False)
    V = Vt.transpose() 
    Vk = V[:,:k]  
    Uk = U[:,:k] 

    #-------------------------- Algoritmo CUR ---------------------------------
    C, indexAc = ColumnSelect(A, k, c, Vk)             # Sceglie c' colonne da A.
    R, indexAr = ColumnSelect(A.transpose(), k, r, Uk) #  Sceglie r' righe da A.
    R=R.transpose()
    U=A[indexAr, :][:,indexAc]  #costruisce U come in teorema [2.1.1] 
    U_pi=pinv(U);               #determina la pseudoinversa di U
    return C, U_pi, R, indexAr, indexAc






def CUR_undersampling(training_dataset: pd.DataFrame, features, ratio: float=0.5, analysis: bool=True):
  A_0 = training_dataset[features]

  k = int(min(A_0.shape[1], int(A_0.shape[0]*ratio))/2)

  C, U_pi, R, indexAr, indexAc = CUR(np.matrix(A_0), k, A_0.shape[1], int(A_0.shape[0]*ratio))

  A_undersampled = A_0.iloc[indexAr,:]

  if analysis:
    A_0_reconstructed = np.round(np.matmul((np.matmul(C, U_pi)),R),5)
    error = norm(A_0-A_0_reconstructed)/(norm(A_0))
    print('reconstruction error',error)

    print('Initial rows', (len(A_0)))
    print('After sampling rows', len(A_undersampled))

  return A_undersampled, indexAr

#-------------------------------- other -----------------------
def sum_is_not_zero(row):
  return sum(row['ext_labels']) != 0

def more_than_one_sentence(row):
  return len(row['document_normalized']) > 1


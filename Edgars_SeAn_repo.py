import json
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import re
import spacy
from matplotlib import pyplot as plt
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split



nlp = spacy.load('en_core_web_sm')
stopwords = spacy.lang.en.stop_words.STOP_WORDS

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def get_df(file):
    df = pd.read_csv(file)
    return df
    
def get_vectorizers():
    ct_vectorizer = CountVectorizer(lowercase=True,
                                strip_accents='ascii',
                                ngram_range=(1, 2),
                                stop_words=['english', 'german'])
    tf_vectorizer = TfidfVectorizer()
    return ct_vectorizer, tf_vectorizer

def tokenize(df, text_col):
    word_count = []
    gna1_list, gna2_list, gna3_list = [], [], []
    for content in df[text_col]:
        match_digits_and_words = ('(\d+|\w+)')
        approach_1 = re.findall(match_digits_and_words, content)
        approach_2 = word_tokenize(content)
        approach_3 = re.match('[a-z0-9 ]+', content) # whole sentences until comma
        word_count.append(len(content))
        approach_1_list.append(approach_1)
        approach_2_list.append(approach_2)
        approach_3_list.append(approach_3)
    print(len(word_count))
    plt.hist(word_count)
    df['text_len'] = word_count
    return df, approach_1_list, approach_2_list, approach_3_list

def tokenize2(string):
    doc = nlp(string)
    tokens = [token.text for token in doc]
    print(tokens)
    return tokens
    
def lemmatize(string):
    doc = nlp(string)
    lemmas = [token.lemma_ for token in doc]
    print(lemmas)
    return lemmas

def word_count(string):
    words = string.split()
    return len(words)

def remove_stop_words(string):
    doc = nlp(string)
    lemmas = [token.lemma_ for token in doc]
    a_lemmas = [lemma for lemma in lemmas 
            if lemma.isalpha() and lemma not in stopwords]
    print(' '.join(a_lemmas))
    return ' '.join(a_lemmas)

def pos_tagging(string):
    doc = nlp(string)
    pos = [(token.text, token.pos_) for token in doc]
    print(pos)
    return pos

def entity_tagging(string):
    doc = nlp(string)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def vectorize(df, col, vectorizer):
    series = df[col]
    bow_matrix = vectorizer.fit_transform(series)
    print(bow_matrix.shape)
    return pd.DataFrame(bow_matrix.toarray())

def split_me(train_df, test_df):
    y_train = train_df['target']
    X_train = train_df.drop(['target'], axis = 1)
    y_test = test_df['target']
    X_test = test_df.drop(['target'], axis = 1)
    return X_train, X_test, y_train, y_test

def split_me_regular(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                y,
                                                test_size=.25,
                                                random_state=42,
                                                stratify=y)
    return X_train, X_test, y_train, y_test

def plot_me(y, X_test_pred):
    plt.subplot(2, 1, 1)
    sns.countplot(X_test_pred)
    plt.subplot(2, 1, 2)
    sns.countplot(y)
    plt.show()


def apply_func(players_df, new_col, used_col, func):
    players_df[new_col] = players_df[used_col].apply(func)
    

if __name__ == '__main__':        
    print('Wrong script, buddy! ;)')
import spacy
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from Edgars_SeAn_repo import get_df, tokenize, vectorize, plot_me, tokenize2, lemmatize, \
            pos_tagging, entity_tagging, split_me_regular, entity_tagging


stopwords = spacy.lang.en.stop_words.STOP_WORDS


def get_vectorizers():
    ct_vectorizer = CountVectorizer(lowercase=True,
                                strip_accents='unicode',
                                ngram_range=(1, 2),
                                stop_words=None,
                                analyzer='word')
    tf_vectorizer = TfidfVectorizer()
    return ct_vectorizer, tf_vectorizer


def sentiment_analyze(X, y, vectorizer, X_t):
    X_train, X_test, y_train, y_test = split_me_regular(X, y)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    #X_train_bow = vectorizer.fit_transform(X_train.values)
    #X_test_bow = vectorizer.transform(X_test.values)
    clf = MultinomialNB()
    clf.fit(X_train_bow, y_train)
    pred = clf.predict(X_test_bow)
    accuracy = clf.score(X_test_bow, y_test)
    accuracy2 = metrics.accuracy_score(y_test, pred)
    print('accuracy: ', accuracy, accuracy2)
    X_t_bow = vectorizer.transform(X_t)
    pred_t = clf.predict(X_t_bow)
    return pred_t


def main():
    ct_vectorizer, tf_vectorizer = get_vectorizers()
    train_df = get_df('train.csv')
    test_df = get_df('test.csv')
    #train_df = train_df.dropna()
    y = train_df['target']
    X = train_df.drop(['target'], axis = 1)
    y = np.asarray(y).astype(np.float32)
    X['id'] = X['id'].astype(np.float32)
    
    cols = X.select_dtypes(include=['object'])
    for col in cols.columns.values:
        X[col] = X[col].fillna('')
    print(X.isna().any())
    
    X_train, X_test, y_train, y_test = split_me_regular(X, y)

    tokenized_train_df, approach_1_list, approach_2_list, approach_3_list = tokenize(train_df, 'text')
    print(approach_1_list)
    #print(df.loc[df['text_len'] > 140, 'text'])
    X_train_vectorized = vectorize(train_df, 'text', ct_vectorizer)
    #X_test_pred = sentiment_analyze(train_df.text, y, tf_vectorizer, test_df.text)
    X_test_pred = sentiment_analyze(train_df.text, y, tf_vectorizer, test_df.text)
    #sns.pairplot(plt.bar(np.arange(len(X_test_pred)), X_test_pred), y.hist())
    plot_me(y, X_test_pred)
    test_df['target'] = X_test_pred
    test_df[['id', 'target']].to_csv('submit.csv', index=False)
    
    
    DL_model(train_df, X_train_vectorized, y)
    

def DL_model(train_df, X_train_vectorized, y):
    X = X_train_vectorized
    X_train, X_test, y_train, y_test = split_me_regular(X, y)
    print(X_train.shape)
    try:
        model = load_model('models/DL_model')
    except OSError:
        model = tf.keras.Sequential([Dense(32, input_shape=(7613, 90130), activation='relu'),
                                     Dense(32, activation='relu'),
                                     Dropout(.3),
                                     Dense(32, activation='relu'),
                                     Dropout(.3),
                                     Dense(32, activation='relu'),
                                     Dropout(.3),
                                     Dense(32, activation='relu'),
                                     Dropout(.3),
                                     Dense(32, activation='relu'),
                                     Dropout(.3),
                                     Dense(1, activation='sigmoid')])
        opt = Adam(learning_rate=.001)
        model.compile(loss='binary_crossentropy', metrics='accuracy', optimizer=opt)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=100)
        Model.save(model, 'models/DL_model')
    preds = model.predict(X_test)
    preds = (model.predict(X_test) > .5).astype('int32')
    print('f1 score: ', f1_score(preds, y_test))
    


if __name__ == '__main__':
    main()
    X = get_df('train.csv')
    X.nunique().sort_values(ascending=False)
    # some tweets are duplicates?
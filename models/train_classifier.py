# import libraries
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine

import re

import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def load_data(database_filepath):
    '''
    Loads data from database into a dataframe.
    Segregates and returns features data, target data and column names
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('Disaster_Response', engine)

    X = df.iloc[:, 1]
    Y = df.iloc[:, 4:]

    categories = list(Y.columns)

    return X, Y, categories


def tokenize(text):
    '''
    Custom tokenize function for text processing.
    Uses nltk to case normalize, lemmatize, and tokenize text.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    www_pat = r'www.[^ ]+'
    stop_words = set(stopwords.words("english"))

    # transform some of the urls that are not in standard format then remove urls
    text = re.sub('http ', 'http://', text)
    stripped = re.sub(url_regex, '', text)
    # clean urls not having http in it
    stripped = re.sub(www_pat, '', stripped)
    # transform text to lower case
    lower_case = stripped.lower()
    # only keep letters
    letters_only = re.sub("[^a-zA-Z]", " ", lower_case)
    # tokenize the text and remove single length words that dont add any value
    tokens = [x for x in word_tokenize(letters_only) if len(x) > 1]
    # remove stop words
    clean_tokens = [token for token in tokens if token not in stop_words]
    # lemmatize and stem tokens to get root words
    lemmer = WordNetLemmatizer()
    ps = PorterStemmer()

    clean_tokens = [lemmer.lemmatize(token.strip()) for token in clean_tokens]
    clean_tokens = [ps.stem(token) for token in clean_tokens]

    return clean_tokens


def build_model():
    '''
    Builds a machine learning pipeline.
    Uses grid search to find optimal hyperparameters
    '''
    tvec = TfidfVectorizer(tokenizer=tokenize)
    multi_clf = MultiOutputClassifier(RidgeClassifier(
        alpha=1, tol=1e-2, solver="sag", random_state=42), n_jobs=-1)

    pipeline = Pipeline([
        ('vect', tvec),
        ('clf', multi_clf)
    ])

    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'vect__use_idf': (True, False),
        'vect__max_features': (5000, 10000),
    }
    gs_clf_ridge = GridSearchCV(
        pipeline, param_grid=parameters, verbose=1, scoring='f1_micro', cv=3)

    return gs_clf_ridge


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Uses the model to predict on test data
    Print the accuracy precision and f1 score for each category_names
    '''
    Y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(category+':\n', classification_report(Y_test[category], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Exports the trained model as a pickle file
    The pickle file be used later on while using the webapp to predict categories for messages
    '''
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

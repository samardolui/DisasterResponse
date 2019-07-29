# import libraries
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('Disaster_Response', engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]

    df = df[df.related!=2]
    df['message_len'] = df['message'].apply(lambda x: len(x))
    df = df[(df.message_len>28) & (df.message_len<400)]
    df = df.drop(['message_len'], axis=1)

    return X,Y,list(Y.columns)

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    www_pat = r'www.[^ ]+'
    stop_words = set(stopwords.words("english"))

    text = re.sub('http ','http://', text)
    text = re.sub(url_regex,'', text)
    stripped = re.sub(www_pat, '', text)
    letters_only = re.sub("[^a-zA-Z]", " ", stripped)

    tokens = word_tokenize(letters_only)
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token.lower().strip()) for token in tokens]

    return clean_tokens

def build_model():
    
    pipeline = Pipeline([
    ('vect',TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RidgeClassifier (alpha=1, tol=1e-2, solver="sag", random_state=42)))
    ])

    parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'vect__use_idf': (True, False),
    'vect__max_features': (None, 5000, 10000),
    }
    gs_clf_ridge = GridSearchCV(pipeline, param_grid=parameters, verbose=2)

    return gs_clf_ridge

def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)
    i = 0
    f1_scores = []
    for category in category_names:
        print(category+':\n', classification_report(Y_test[category], Y_pred[:,i]))
        f1_scores.append(f1_score(Y_test[category], Y_pred[:,i], average = 'weighted'))
        i += 1
    print('Average f1 score ',np.mean(f1_scores, axis=0))

def save_model(model, model_filepath):

    with open(model_filepath,"wb") as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

# import libraries
import json
import plotly
import pandas as pd
import numpy as np
import re

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from sklearn.externals import joblib

app = Flask(__name__)


def tokenize(text):
    '''
    Custom tokenize function for text processing.
    Uses nltk to case clean, normalize, lemmatize, and tokenize text.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    www_pat = r'www.[^ ]+'
    stop_words = set(stopwords.words("english"))

    # transform some of the urls that are not in standard format, then remove urls
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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disaster_Response', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model


@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # extract data needed for visuals
    cat_val = np.round(df[df.columns[4:40]].mean(
        axis=0).sort_values(ascending=False)*100, 2)
    cat_ids = [category.replace('_', ' ').title()
               for category in cat_val.index]

    new_graph = {
        'data': [
            Bar(
                x=cat_ids,
                y=cat_val,
                text=cat_val,
                textposition='auto',
            )
        ],

        'layout': {
            'title': 'Distribution of Message by Category',
            'yaxis': {
                'title': "Percentage"
            },
            'xaxis': {
                'title': "Category",
                'tickangle': -35
            }
        }
    }
    graphs.append(new_graph)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    # sort the results to show the predicted categories on top of the search results
    classification_results = dict(
        sorted(classification_results.items(), key=lambda kv: kv[1], reverse=True))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=8080, debug=True)


if __name__ == '__main__':
    main()

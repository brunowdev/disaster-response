import json
import plotly
import pandas as pd
import numpy as np

import nltk

nltk.download('words')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

import sys

sys.path.insert(0, './models')

import nlp_extractors

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load metrics dataset
df_metrics = pd.read_csv('./models/linear_model_metrics.csv')
df_metrics = df_metrics.sort_values(by = ['f1-score' ], ascending = False)

# load model
model = joblib.load("./models/model.pkl")

# Configurations 
chart_font_size = 18
tick_rotation = -45

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Prepare the data for metrics visualization
    data_metrics = []
    for metric in ['precision', 'recall', 'f1-score']:
        data_metrics.append({
            'name' : metric.title(),
            'type' : 'bar',
            'x' : list(df_metrics.category),
            'y' : list(map(lambda x: np.around(x, 2), df_metrics[metric]))
        })
    
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
                'font': {
                    'size': chart_font_size
                },
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'categoryorder' : ['news', 'direct', 'social']
                }
            }
        },
        {
            'data' : data_metrics,
                'layout' : {
                    'title': 'Linear Regression Model Performance',
                    'font': {
                        'size': chart_font_size
                    },
                    'xaxis': {
                        'tickangle': tick_rotation,
                        'ticks': 'outside',
                        'automargin': True
                    },
                    'yaxis': {
                        'ticks': 'outside'
                    },
                    'barmode': 'group'
                }
        }
    ]
    
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
    _result = model.predict([query])

    classification_labels = _result[0]
    _possible_labels = list(df.columns[4:])
    _possible_labels.pop(_possible_labels.index('child_alone'))

    classification_results = dict(zip(_possible_labels, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()

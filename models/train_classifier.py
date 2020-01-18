#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import pandas as pd
import nlp_extractors

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

import pickle

expected_arguments_length = 2
data_argument_index = 0
model_output_argument_index = 1
metrics_filename = 'linear_model_metrics.csv'

random_state = 42
test_size = .3

def load_data(db_filename):

    print('Loading data...')

    engine = create_engine(f'sqlite:///{db_filename}')
    df = pd.read_sql_table('messages', engine)

    labels = list(df.columns[4:])

    X = df.message
    Y = df[labels]

    Y = Y.drop(['child_alone'], axis = 1)
    labels.pop(labels.index('child_alone'))

    return X, Y, labels

def create_model():

    print('Creating model...')

    model = LogisticRegression(random_state = random_state, solver = 'lbfgs', cv = 5, max_iter = 1000)

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=nlp_extractors.tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('numeric_digit', nlp_extractors.NumericDigitExtractor()),
            ('question', nlp_extractors.QuestionExtractor())
        ])),
    
        ('classifier', MultiOutputClassifier(model)) 
    ])

    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        # 'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        # 'features__text_pipeline__tfidf__use_idf': (True, False),
        'features__text_pipeline__tfidf__norm': ('l1', 'l2'),
        #'classifier__estimator__n_estimators': [50, 100, 200],
        #'classifier__estimator__C': [5, 10],
        #'classifier__estimator__penalty' : ['l1', 'l2']
    }

    return GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, verbose = 10)

def evaluate_model(model, X_test, y_test, labels):

    print('Evaluating model...')

    y_pred = model.predict(X_test) 

    report = classification_report(y_test.values, y_pred, target_names=labels, output_dict = True)

    df_metrics = pd.DataFrame(list(map(lambda label: map_with_key(report, label), labels)))
    df_metrics.to_csv(metrics_filename, index = False)


def map_with_key(report, key):
    result = report.get(key)
    result['category'] = key
    return result

def save_model(model, model_filename):
    print('Saving the model...')
    pickle.dump(model, open(model_filename, 'wb'))

def main(args):

    db_filename = args[data_argument_index]
    model_filename = args[model_output_argument_index]

    X, Y, labels = load_data(db_filename)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = random_state)

    model = create_model()    

    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, labels)
    
    save_model(model, model_filename)

    print('Training finished successfully.')


if __name__ == '__main__':
    main(sys.argv[1:])

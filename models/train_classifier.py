#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

import pandas as pd
import pickle
import nlp_extractors

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

expected_arguments_length = 2
data_argument_index = 0
model_output_argument_index = 1
metrics_filename = 'linear_model_metrics.csv'

random_state = 42
test_size = .3

def load_data(db_filename):
    """
    Loads data from the database file.
    Args:
        db_filename: path to file
    Returns:
        (DataFrame) X: The feature column
        (DataFrame) Y: Labels
        (List) labels: The list of target labels
    """
    print('Loading data...')

    engine = create_engine(f'sqlite:///{db_filename}')
    df = pd.read_sql_table('messages', engine)

    labels = list(df.columns[4:])

    X = df.message
    Y = df[labels]

    # Removes all the categories with less than one sample 
    df_filtered = df[labels].sum().gt(0).to_frame().reset_index()
    df_filtered.columns = [ 'category', 'non_zero' ]
    categories_to_remove = list(df_filtered[df_filtered.non_zero == False]['category'])
    for category in categories_to_remove:
        print(f'Warning: Removing the category {category} since it has no samples.')
        Y = Y.drop([category], axis = 1)
        labels.pop(labels.index(category))

    return X, Y, labels

def create_model():
    """
    Create a model with the pipeline.
    Returns:
        (DataFrame) X: The feature column
        (DataFrame) Y: Labels
        (List) labels: The list of target labels
    """
    print('Creating model...')

    model = LogisticRegression(random_state = random_state, solver = 'lbfgs', max_iter = 1000)

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

    return GridSearchCV(pipeline, cv = 5, param_grid = parameters, n_jobs = -1, verbose = 10)

def evaluate_model(model, X_test, y_test, labels):
    """
    Evaluates the model with the test data.
    Args:
        model: The model
        X_test: The feature column
        Y_test: Labels
        labels: The list of target labels
    """
    print('Evaluating model...')

    y_pred = model.predict(X_test) 

    report = classification_report(y_test.values, y_pred, target_names=labels, output_dict = True)
    print(report)

    print('Saving training report...')
    metrics = list(map(lambda label: extract_category(report, label), labels))
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(metrics_filename, index = False)


def extract_category(report, key):
    """
    Create a dict from a classification report for a given category.
    Args:
        report: The report
        key: The key to a category on the report
    """
    result = report.get(key)
    result['category'] = key
    return result

def save_model(model, model_filename):
    """
    Save the model to a pickle file.
    Args:
        model: The model
        model_filename: Name/path to save the model
    """
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

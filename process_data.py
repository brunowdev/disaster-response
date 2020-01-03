#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

from sqlalchemy import create_engine

import sys

supported_categories = 36
category_separator = ';'
category_value_separator = '-'
category_name_position = 0
category_value_position = 1
category_column = 'categories'

expected_arguments_length = 3
messages_argument_index = 0
categories_argument_index = 1
database_argument_index = 2

default_categories_file = 'categories.csv'
default_messages_file = 'messages.csv'
default_database_file = 'disaster_response.db'

def parse_arguments(arguments):
    
    """Parses the argument list to a dict with the required parameters.

    Args:
        arguments (list): The arguments passed to the script.
        
Parses the argument list to a dict with the required parameters.Parses the argument list to a dict with the required parameters.    Returns:
        Return a dict with all parameters except the script name itself.

    Examples:
        >>> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
        {
            'messages_filename' : 'disaster_messages.csv',
            'categories_filename' : 'disaster_categories.csv',
            'database_filename' : 'DisasterResponse.db'
        }

    """
    
    log_start(f'Parsing arguments: {arguments}')

    arguments = arguments
    have_arguments = len(arguments) == expected_arguments_length
    messages_filename = arguments[messages_argument_index] if have_arguments else default_categories_file
    categories_filename = arguments[categories_argument_index] if have_arguments else default_messages_file
    database_filename = arguments[database_argument_index] if have_arguments else default_database_file
    
    finish_log_success(f'Arguments parsing successfully: {database_filename}')
    return {
        'messages_filename' : messages_filename,
        'categories_filename' : categories_filename,
        'database_filename' : database_filename
    }

def split_category_column_to_list(text, position):
    return list(map(lambda cat: cat.split(category_value_separator)[position], text.split(category_separator)))

def get_categories_name_from_text(text):
    return split_category_column_to_list(text, category_name_position) 

def get_categories_value_from_text(text):
    return list(map(lambda value: int(value), split_category_column_to_list(text, category_value_position)))

def load_data_from_csv(filename, schema = None):
    log_start(f'Reading file: {filename}')
    df = pd.read_csv(filename)
    if schema is not None:
        assert list(df.columns) == schema
    finish_log_success(f'File read successfully:: {filename}')
    return df
    
def merge_datasets(df1, df2, key):
    return pd.merge(df1, df2, on = key)

def expand_categories_to_columns_with_values(df, category_column, categories):
    new_df = df.copy()
    new_df[categories] = df[category_column].apply(lambda c: pd.Series(get_categories_value_from_text(str(c))))
    return new_df

def create_db_file(df, database_filename, table_name = 'messages'):
    log_start(f'Creating database: {database_filename}')
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table_name, engine, index = False, if_exists = 'replace')
    finish_log_success(f'Database created: {database_filename}')

def main(args):
    
    print('Starting ETL script')
    
    arguments = parse_arguments(args)
    
    messages = load_data_from_csv(arguments['messages_filename'])
    categories = load_data_from_csv(arguments['categories_filename'])
    
    merged = merge_datasets(messages, categories, 'id')
    
    # Get the category string from the first row
    category_raw_text = merged.iloc[0].categories
    categories_name = get_categories_name_from_text(category_raw_text)
    
    full_dataset = expand_categories_to_columns_with_values(merged, category_column, categories_name)
    full_dataset = full_dataset.drop([category_column], axis = 1)
    
    print(f'Total of duplicated rows {bold_string("before")} drop: {bold_string(len(full_dataset[full_dataset.duplicated()]))}')
    full_dataset = full_dataset.drop_duplicates(keep = 'first')
    print(f'Total of duplicated {bold_string("after")} before drop: {bold_string(len(full_dataset[full_dataset.duplicated()]))}')
    
    create_db_file(full_dataset, arguments['database_filename'])
    
    print('')
    print('Finished ETL script')
    
def bold_string(text):
    return f'\033[1m{text}\033[0m'
    
def log_start(message):
    print(message)
    
def finish_log_success(message):
    print(message)
    print('')

if __name__ == '__main__':
    main(sys.argv[1:])
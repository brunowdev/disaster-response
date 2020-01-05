#!/usr/bin/python
# -*- coding: utf-8 -*-

import language_utils as lang_utils
import pandas as pd
from sqlalchemy import create_engine
import sys

supported_categories = 36
category_separator = ';'
category_value_separator = '-'
category_name_position = 0
category_value_position = 1
category_column = 'categories'
minimum_words_to_check_row = 10

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

    Returns:
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
    """A function to split the category string either as name
    (the first parameter) or the value (second parameter).

    Args:
        text (string): The text to split.
        position (int): The position to get a value (0 or 1).

    """
    return list(map(lambda cat: cat.split(category_value_separator)[position], text.split(category_separator)))

def get_categories_name_from_text(text):
    """Split the text with categories to a list of categories name.

    Args:
        text (string): The text to split. 
        
    Examples:
        >>> get_categories_value_from_text('related-1;request-0;offer-0')
        ['related', 'request', 'offer']

    """
    return split_category_column_to_list(text, category_name_position) 

def get_categories_value_from_text(text):
    """Split the text with categories to a list of categories value.

    Args:
        text (string): The text to split.
        
    Examples:
        >>> get_categories_value_from_text('related-1;request-0;offer-0')
        [1, 0, 0]

    """
    return list(map(lambda value: int(value if value == '0' else 1), split_category_column_to_list(text, category_value_position)))

def load_data_from_csv(filename, schema = None):
    """Merge two pandas dataframes based given a key.

    Args:
        filename (string): The name/path to file.
        schema (list): Columns to validate the dataframe.

    """
    log_start(f'Reading file: {filename}')
    df = pd.read_csv(filename)
    if schema is not None:
        assert list(df.columns) == schema
    finish_log_success(f'File read successfully:: {filename}')
    return df
    
def merge_datasets(df1, df2, key):
    """Merge two pandas dataframes based given a key.

    Args:
        df1 (dataframe): The first dataframe to merge.
        df2 (dataframe): The second dataframe to merge.
        key (string): The column used to join both dataframes.

    """
    return pd.merge(df1, df2, on = key)

def expand_categories_to_columns_with_values(df, category_column, categories):
    """Create a column for every category on categories column with the given value.

    Args:
        df (dataframe): The dataframe.
        category_column (string): The name of categories column.
        categories (list): The sorted list with the name of all existing categories.

    """
    new_df = df.copy()
    new_df[categories] = df[category_column].apply(lambda c: pd.Series(get_categories_value_from_text(str(c))))
    return new_df

def create_db_file(df, database_filename, table_name = 'messages'):
    log_start(f'Creating database: {database_filename}')
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(table_name, engine, index = False, if_exists = 'replace')
    finish_log_success(f'Database created: {database_filename}')
    
def remove_rows_with_invalid_words(df):
    
    copied_df = df.copy()
    print(f'Total of rows before cleaning: {len(copied_df)}')
    
    copied_df['length'] = copied_df.message.apply(lambda msg: len(msg.strip()))
    copied_df['words_length'] = copied_df.message.apply(lambda msg: len(msg.strip().split(' ')))
    
    possible_bad_rows = copied_df[(copied_df['words_length'] < minimum_words_to_check_row)]
    bad_data = possible_bad_rows[~possible_bad_rows.message.apply(lang_utils.has_any_word)]
    
    # Remove bad rows from the dataframe
    copied_df = copied_df[~copied_df.id.isin(list(bad_data.id))]
    
    print(f'Total of rows after cleaning: {len(copied_df)}')
    
    return copied_df.drop(['length', 'words_length'], axis = 1)
    
def remove_duplicated_rows(df, keep = 'first'):
    
    s_before = len(df[df.duplicated()])
    print(f'Total of duplicated rows {bold_string("before")} drop: { bold_string(s_before)}')
    
    new_df = df.drop_duplicates(keep = keep)
    
    s_after = len(new_df[new_df.duplicated()])
    print(f'Total of duplicated rows {bold_string("after")} before drop: { bold_string(s_after)}')
    
    return new_df 

def main(args):
    
    print('Starting ETL script')
    
    arguments = parse_arguments(args)
    
    # Loading data
    messages = load_data_from_csv(arguments['messages_filename'])
    categories = load_data_from_csv(arguments['categories_filename'])
    
    # Merge the data
    merged = merge_datasets(messages, categories, 'id')
    
    # Get the category string from the first row
    category_raw_text = merged.iloc[0].categories
    categories_name = get_categories_name_from_text(category_raw_text)
    
    # Create a column for each category
    full_dataset = expand_categories_to_columns_with_values(merged, category_column, categories_name)
    full_dataset = full_dataset.drop([category_column], axis = 1)
    
    # Remove duplicated rows
    full_dataset = remove_duplicated_rows(full_dataset)
    
    # Handles bad rows
    cleaned_df = remove_rows_with_invalid_words(full_dataset)

    # Store the file
    create_db_file(cleaned_df, arguments['database_filename'])
    
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
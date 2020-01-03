import pandas as pd

from sqlalchemy import create_engine

import sys

# Total of categories allowed on this version
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

# Get all the arguments, except the etl script name
# Given: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
# Will return: ['disaster_messages.csv', 'disaster_categories.csv', 'DisasterResponse.db']
arguments = sys.argv[1:]
have_arguments = len(arguments) == expected_arguments_length
messages_filename = arguments[messages_argument_index] if have_arguments else default_categories_file
categories_filename = arguments[categories_argument_index] if have_arguments else default_messages_file
database_filename = arguments[database_argument_index] if have_arguments else default_database_file

def split_category_column_to_list(text, position):
    return list(map(lambda cat: cat.split(category_value_separator)[position], text.split(category_separator)))

def get_categories_name_from_text(text):
    return split_category_column_to_list(text, category_name_position) 

def get_categories_value_from_text(text):
    return list(map(lambda value: int(value), split_category_column_to_list(text, category_value_position)))

def load_data_from_csv(filename, schema = None):
    df = pd.read_csv(filename)
    if schema is not None:
        assert list(df.columns) == schema
    return df

def load_categories(filename = categories_filename):
    return load_data_from_csv(filename)
    
def load_messages(filename = messages_filename):
    return load_data_from_csv(filename)
    
def merge_datasets(df1, df2, key):
    return pd.merge(df1, df2, on = key)

def expand_categories_to_columns_with_values(df, category_column, categories):
    new_df = df.copy()
    new_df[categories] = df[category_column].apply(lambda c: pd.Series(get_categories_value_from_text(str(c))))
    return new_df

def create_db_file(df, engine_name = f'sqlite:///{database_filename}', table_name = 'messages'):
    engine = create_engine(engine_name)
    df.to_sql(table_name, engine, index = False, if_exists = 'replace')
    
def main(args):
    
    categories = load_categories()
    messages = load_messages()
    
    merged = merge_datasets(messages, categories, 'id')
    
    # Get the category string from the first row
    category_raw_text = merged.iloc[0].categories
    categories_name = get_categories_name_from_text(category_raw_text)
    
    full_dataset = expand_categories_to_columns_with_values(merged, category_column, categories_name)
    full_dataset = full_dataset.drop([category_column], axis = 1)
    
    full_dataset = full_dataset.drop_duplicates(keep = 'first')
    
    create_db_file(full_dataset)

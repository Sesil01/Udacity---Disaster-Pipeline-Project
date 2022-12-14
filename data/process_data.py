#Import Libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
    load messages dataset
    input:
        messages_filepath:
        categories_filepath:
    output:
        df: dataframe, merge between messages and categories on id
    Return :
        True
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', on=['id'])

    return df

def clean_data(df):
    """
    clean dataframe

    Input:
        df: dataframe that containing messages and categories from load data

    Returns:
        df: dataframe has been clean
    """
    # Split the values in the categories column on the ; character so that each value becomes a separate column.
    categories = df['categories'].str.split(";", expand = True)

    # Use the first row of categories dataframe to create column names for the categories data.
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    # Rename columns of categories with new column names.
    row = categories.iloc[0]
    category_colnames = row.apply (lambda x: x[:-2])
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    # For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        categories['related'] = categories.related.map(lambda x: 1 if x == 2 else x)

    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join = "inner", axis = 1)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

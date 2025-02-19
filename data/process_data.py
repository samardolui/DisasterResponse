# import libraries
import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads the messages and categories dataset.
    Merges them into a single dataframe.
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''
    Cleans the data frame and transforms the categories into binaries
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames = [x.split('-')[0] for x in row]
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()

    # remove the noisy data in 'related'
    df = df[df.related != 2]
    # remove the top outliers that does not contribute much to the predictive model
    df['message_len'] = df['message'].apply(lambda x: len(x))
    df = df[(df.message_len > 25) & (df.message_len < 500)]
    df = df.drop(['message_len'], axis=1)

    return df


def save_data(df, database_filename):
    '''
    Saves the data into a database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(name='Disaster_Response', con=engine,
              if_exists='replace', index=False)


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

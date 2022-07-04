
import sys
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    disaster_categories = pd.read_csv(categories_filepath)
    disaster_messages = pd.read_csv(messages_filepath)

    return pd.merge(disaster_messages, disaster_categories, on='id')


def clean_data(df):

    df_categories = df.categories.str.split(';', expand=True)
    column_names = [x.split('-')[0] for x in df_categories.iloc[0]]
    df_categories.columns = column_names

    for col in df_categories.columns:
        df_categories[col] = df_categories[col].str.split('-').str.get(1)
        df_categories[col] = df_categories[col].astype(int)
    df_categories['id'] = df['id']
    
    df = pd.concat([df, df_categories], axis=1).drop('categories', axis=1).iloc[:, :-1].drop_duplicates()
    df['related'] = df['related'].replace({2: 1})
    print(df.head())
    
    return df


def save_data(df, database_filename):

    connection = create_engine(f'sqlite:///{database_filename}')
    try:
        connection.execute('SELECT 1 FROM database_messages')
        print('Table already exists, deleting files...')

    except OperationalError:
        pass

    df.to_sql('database_messages', connection, if_exists='replace', index=False)
    print('Table Created/Updated')

    return


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
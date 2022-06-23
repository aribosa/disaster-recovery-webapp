"""
Classification Model
Full ML Pipeline to detect disaster messages among social media, text or web articles.

Usage:
    Python Script
    > python3 train_classifier.py <path to sqllite db file> <path to the models pickle file>

    Python Script Sample
    > python3 train_classifier.py DisasterResponse.db ./model/classif_model.pkl

Args:
    db_file_source: directory for the Database model (.db file, containing sqlite schema and data)
    model_pickle: pickle object with pre-trained model
"""

# NLP libraries
import nltk
nltk.download('punkt')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('stopwords')


from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Data
import pandas as pd
import pickle
import numpy as np
from sqlalchemy import create_engine

import sys
import datetime
import os
import re
import pickle

# ML models
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    Returns both target and feature vectors from the messages database

    Args:
        database_filepath::str
            Database filepath to retrieve the data using SQLite

    Returns:
        X::pd.Series
            Pandas series containing the feature vector

        y::pd.DataFrame
            Dataframe with all the target values for all the observations

        categories::[str]
            List with all category names from the given dataset

    """
    engine = create_engine('sqlite:///'  + database_filepath)
    data_frame = pd.read_sql(f'SELECT * FROM database_messages', engine)

    return data_frame['message'], data_frame.iloc[:, 4:]


def tokenize(text):
    """
    Performs a tokenization to the provided text to that feature texts can be seen as vectors
    to our Machine Learning pipeline.

    Args:
        text:str
            message string to split

    Returns
        vector:[str]
            array with tokens
    """

    # Replace url hyperlinks with special place holders
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regex, text)

    # Crawl for each found URL and replace in the original text
    for url in urls:
        text = text.replace(url, 'url_placeholder')

    # Divide sentences into words (splitted by space)
    tokens = nltk.word_tokenize(text)

    # Lemmatize all the tokens to obtain the word's stem or root representation
    lem = nltk.stem.WordNetLemmatizer()
    clean_tokens = [lem.lemmatize(token).lower().strip() for token in tokens]
    clean_tokens = list(filter(lambda x: len(x) > 2, clean_tokens))  # Filter tokens with less than 2 characters
    
    # Discarded for performance issues
    # clean_tokens = [token for token in clean_tokens if token in stopwords.words('english')]

    return clean_tokens


class LoggerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X:pd.DataFrame, y=None):
        return X


class StartingVerbTransformer(BaseEstimator, TransformerMixin):

    @staticmethod
    def start_verb(text):
        try:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                pos_tags = nltk.pos_tag(tokenize(sentence))
                f_word, f_tag = pos_tags[0]
                if f_tag in ['VB', 'VBP'] or f_word == 'RT':
                    return True
        except:
            return False
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = pd.DataFrame(pd.Series(X).apply(self.start_verb))
        return X_new


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('pipeline-nlp', Pipeline([
                ('vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf-transformer', TfidfTransformer())
            ])),

            ('sentence-tagger', StartingVerbTransformer())
        ])),
        ('logging', LoggerTransformer()),

        ('classification', MultiOutputClassifier(RandomForestClassifier(n_estimators=45, n_jobs=-1)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names=None):
    Y_prediction = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_prediction, columns=Y_test.columns)

    # conf_matrix = confusion_matrix(Y_test, Y_prediction)
    # print('Confusion Matrix (Actual Values / Predicted')
    # print(conf_matrix)


    for i, column in enumerate(Y_test.columns):
        print(f'Model performance over feature {column}: {accuracy_score(Y_test[column], Y_pred[column])}')

    return


def save_model(model: Pipeline, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    
    print('Model Saved')
    return True


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        start = datetime.datetime.now()

        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        print(f'Total time using {n} rows: {datetime.datetime.now() - start}')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
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


URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
NOT_WORD_REGEX = re.compile('[^A-Za-z0-9]')
STOP_WORDS = stopwords.words('english')

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

    urls = re.findall(URL_REGEX, text)

    # Crawl for each found URL and replace in the original text
    for url in urls:
        text = text.replace(url, 'url_placeholder')

    # Divide sentences into words (splitted by space)
    tokens = nltk.word_tokenize(NOT_WORD_REGEX.sub(" ", text.lower()))
    tokens = [t for t in tokens if t not in STOP_WORDS]

    # Lemmatize all the tokens to obtain the word's stem or root representation
    lem = nltk.stem.WordNetLemmatizer()
    clean_tokens = [lem.lemmatize(t) for t in tokens]

    return clean_tokens


class StartingVerbTransformer(BaseEstimator, TransformerMixin):
    """
    Identifies wether the first word from the feature (message) contains a verb
    """
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
    """
    Returns a Sklearn Pipeline to perform an end-to-end transformation and training.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('pipeline-nlp', Pipeline([
                ('vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf-transformer', TfidfTransformer())
            ])),

            ('sentence-tagger', StartingVerbTransformer())
        ])),
        
        ('classification', MultiOutputClassifier(AdaBoostClassifier(n_estimators=200, learning_rate=0.3)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test):
    """
    Gathers the accuracy score for each target label

    Params:
        model: sklearn transformer with predict method
        X_test: feature vector
        Y_test: target vector (real values)
    """
    Y_prediction = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_prediction, columns=Y_test.columns)

    # Loop trough each column and test the model's accuracy
    for i, column in enumerate(Y_test.columns):
        print(f'Model performance over feature {column}: {accuracy_score(Y_test[column], Y_pred[column])}')

    return


def save_model(model: Pipeline, model_filepath):
    # Save the model into a pickle file, allowing reusability
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    
    print('Model Saved')
    return True


def main():
    if not len(sys.argv) == 3:
        print('No model and model file name provided, using defaults')

    database_filepath, model_filepath = sys.argv[1:] if len(sys.argv) > 1 else (None, None)

    # If no values are provided, default values are used
    database_filepath = database_filepath or '../data/DisasterResponse.db'
    model_filepath = model_filepath or 'model.pkl'

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))


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


if __name__ == '__main__':
    main()
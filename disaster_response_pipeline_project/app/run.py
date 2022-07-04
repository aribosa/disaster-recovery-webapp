import re
import sys
import nltk
import json
import plotly
import joblib
import pandas as pd
from flask import Flask
from plotly.graph_objs import Bar
from plotly import express as px
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from flask import render_template, request, jsonify
from sklearn.base import BaseEstimator, TransformerMixin


URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
NOT_WORD_REGEX = re.compile('[^A-Za-z0-9]')
STOP_WORDS = stopwords.words('english')

app = Flask(__name__)

def tokenize(text):
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


# Web App index page (metrics)
@app.route('/')
@app.route('/index')
def index():
    global df
    # Group the messages database by genre and obtain the unbalance ratio
    df_group = df.groupby('genre').size().to_frame('size').reset_index()
    unbalance = pd.DataFrame(df.iloc[:, 4:].sum() / len(df), columns=['unbalance']).reset_index().sort_values('unbalance')

    # Bar Plot n1 - Count of messages by Genre
    fig_1 = px.bar(
        x=df_group['genre'],
        y=df_group['size'],
        text=[f'{_/1000:.1f} k' if _ > 1000 else _ for _ in df_group['size']],
        color=df_group['genre'],
        color_discrete_sequence= ['lightblue', 'lightgreen', 'blue']
    )

    fig_1.update_xaxes(title='Origin')
    fig_1.update_yaxes(visible=False)
    
    # Bar Plot n2 - Unbalance ratio by genre
    fig_2 = px.bar(
        data_frame=unbalance,
        x='unbalance',
        y='index',
        orientation='h',
        height=800,
        color=['1' if _ > 0.2 else '0' for _ in unbalance.unbalance],
        color_discrete_map={'1': 'green', '0':'gray'}
    )

    fig_2.update_xaxes(title = 'Unbalance Ratio (higher is better)')
    fig_2.update_yaxes(title = '')
    fig_2.update_traces(showlegend=False)
    fig_2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    graphs = [fig_1, fig_2]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    global model

    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    tokens = tokenize(query)
    classification_labels = model.predict([query])    
    classification_labels = classification_labels[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render go.html jinja template with classification data 
    return render_template(
        'go.html',
        query=query,
        tokens=tokens,
        classification_result=classification_results
    )


def main():
    if len(sys.argv) >= 3:

        database_path = sys.argv[1]
        model_path = sys.argv[2]

        # Load data from Messages Database
        engine = create_engine(f'sqlite:///{database_path}')
        globals()['df'] = pd.read_sql_table('database_messages', engine)

        # load model
        globals()['model'] = joblib.load(f"{model_path}")

            # Run application on port 3001
        app.run(host='0.0.0.0', port=3001, debug=True)

    else:
        print("""
##########
Error: To execute the Disaste Recovery App please provide the database and model picke directory.
Usage:
        python run.py path_to_database path_to_pickle
##########
        """)

if __name__ == '__main__':
    main()
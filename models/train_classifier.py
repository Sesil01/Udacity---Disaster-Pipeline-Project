# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

import re
import sys
import pandas as pd
import numpy as np
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """ Load data from database
    input:
        df: dataset
    outputs:
        X: the text messages
        y: the training label
        category_names: name of columns
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = y.columns
    return X,y,category_names

def tokenize(text):
    """
    Tokenizes text data

    Returns:
        words list: Processed text after normalizing, tokenizing and lemmatizing
    """
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_regex)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stop_words = stopwords.words("english")
    tokens = [tok for tok in tokens if tok not in stop_words]

    #Lemmanitizer
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens

def build_model():
    """
    Build a machine learning Pipeline That takes in the message column as input and output classification results on the other 36 categories in the dataset.

    Input:
        Using vectorize and TfidfTransformer.
        Pipeline using MultiOutputClassifier for predicting multiple target variables pipeline function

    output:
        cv: the model

    """

    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    param_grid = {
    "clf__estimator__min_samples_split": [2, 6],
    "clf__estimator__max_depth": [4, 6, 8]}
    cv = GridSearchCV(model, param_grid, verbose=3)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate models
    input:
        y_pred: pipeline prediction from X_test
        class_report: classification report
    output:
        prediction the models
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, target_names=category_names)

    print(class_report)
    print('Accuracy: {}'.format(np.mean(y_test.values == y_pred)))

def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    return True

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

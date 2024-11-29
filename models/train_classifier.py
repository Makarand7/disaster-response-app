import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine


def load_data(database_filepath):
    """
    Load data from the SQLite database.
    
    Args:
        database_filepath (str): Path to the SQLite database.
        
    Returns:
        X (pd.Series): Messages (input features).
        y (pd.DataFrame): Categories of the messages (output labels).
        category_names (list): List of category names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    Tokenize and clean text data.
    
    Args:
        text (str): Input text to tokenize.
        
    Returns:
        clean_tokens (list): Tokenized and cleaned text.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline and set up grid search.
    
    Returns:
        model (GridSearchCV): Pipeline wrapped with grid search.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),  # Suppress warning by setting token_pattern=None
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'clf__estimator__learning_rate': [0.5, 1.0],
        'clf__estimator__n_estimators': [10, 20]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=3, n_jobs=-1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and print classification reports for each category.
    
    Args:
        model: Trained machine learning model.
        X_test (pd.Series): Test data (input features).
        Y_test (pd.DataFrame): True labels for test data.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
        model: Trained model.
        model_filepath (str): Path to save the model file.
    """
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Main function to load data, build and train model, evaluate, and save the model.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        
        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model.best_estimator_, model_filepath)
        
        print('Trained model saved!')
    else:
        print('Usage: python train_classifier_new.py <DATABASE_FILEPATH> <MODEL_FILEPATH>')
        sys.exit(1)


if __name__ == '__main__':
    main()

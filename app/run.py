import json
import pandas as pd
from sqlalchemy import create_engine
from flask import Flask, render_template, request, jsonify
from joblib import load
import plotly
from plotly.graph_objs import Bar

# Add the project root directory to the system path
import os
import sys
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_DIR)

# Ensure required NLTK data is downloaded
import nltk
nltk.download("punkt", download_dir="/opt/render/nltk_data")
nltk.download("punkt_tab", download_dir="/opt/render/nltk_data")
nltk.download("wordnet", download_dir="/opt/render/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="/opt/render/nltk_data")

# Import the tokenize function from train_classifier in models
from models.train_classifier import tokenize

app = Flask(__name__)

# Debugging logs
print(f"Project root directory: {PROJECT_DIR}")

# Define paths relative to the project directory
DATABASE_FILEPATH = os.path.join(PROJECT_DIR, "data", "DisasterResponse.db")
MODEL_FILEPATH = os.path.join(PROJECT_DIR, "models", "classifier.pkl")

print(f"Database file path: {DATABASE_FILEPATH}")
print(f"Model file path: {MODEL_FILEPATH}")

# Load data
try:
    engine = create_engine(f"sqlite:///{DATABASE_FILEPATH}")
    with engine.connect() as connection:
        df = pd.read_sql_table("disaster_messages", con=connection)
    print("Database connected successfully.")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit(1)

# Load model
try:
    model = load(MODEL_FILEPATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Index webpage displays visuals and receives user input text for the model
@app.route('/')
@app.route('/index')
def index():
    """Display the homepage with visuals."""
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_names = df.iloc[:, 4:].columns
    category_counts = (df.iloc[:, 4:] != 0).sum()

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Category"}
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render master.html with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    """Handle user query and display classification results."""
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render go.html with the query and classification results
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()

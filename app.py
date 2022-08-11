 # Importing essential libraries
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import re
from sklearn.metrics.pairwise import sigmoid_kernel
import pandas as pd






app = Flask(__name__)


df=pd.read_csv('books_new.csv')

df = df.drop(["Publisher"],axis =1 )

df = df.dropna()

df["overview"] = df["Genre"]+ "  " + df["SubGenre"]+ "  "  + df["Author"]
df["Title"] = df["Title"] +" " +"by"+" " + df["Author"]

books_cleaned_df = df.drop(["Author" ,"Genre","SubGenre","Height"],axis =1 )

books_cleaned_df = books_cleaned_df.reset_index()

from sklearn.feature_extraction.text import TfidfVectorizer


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3),
            stop_words = 'english')

# Filling NaNs with empty string
books_cleaned_df['overview'] = books_cleaned_df['overview'].fillna('')


# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(books_cleaned_df['overview'])



# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# Reverse mapping of indices and movie titles
indices = pd.Series(books_cleaned_df.index, index=books_cleaned_df['Title']).drop_duplicates()

def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores 
    sig_value = list(enumerate(sig[idx]))
    
    sig_scores = []
    for i in sig_value:
        if i[0] !=idx:
            sig_scores.append(i)

    # Sort the movies 
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 5 most similar movies
    sig_scores = sig_scores[1:4]

    # Movie indices
    books_indices = [i[0] for i in sig_scores if i[0] ]

    # Top 10 most similar movies
    return books_cleaned_df['Title'].iloc[books_indices]


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        broswer = request.form.get('browser')

        output = list(give_rec(broswer))
        

       
        return render_template('result.html', prediction=output)
        

if __name__ == '__main__':
     app.run(debug=True)

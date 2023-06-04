import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

# Load the movies dataset
movies_df = pd.read_csv(r"C:\Users\dhana\Downloads\ml-25m\ml-25m\movies.csv")

# Preprocess the movie titles for matching
movies_df['processed_title'] = movies_df['title'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))

# Create a TF-IDF vectorizer to convert movie genres into feature vectors
vectorizer = TfidfVectorizer(token_pattern=r'[^\s|,]+')

# Fit the vectorizer on the movie genres
genre_features = vectorizer.fit_transform(movies_df['genres'])

# Build the Nearest Neighbors model based on genre features
genre_model = NearestNeighbors(metric='cosine', algorithm='brute')
genre_model.fit(genre_features)

# Get input from the user for a movie title
user_movie = input("Enter a movie title: ")

# Find the closest matching movie title
closest_match = process.extractOne(user_movie, movies_df['processed_title'])

if closest_match[1] >= 80:
    closest_title = closest_match[0]
    print(f"\nClosest matching movie title: {closest_title}")
    
    # Get the index of the closest matching movie in the DataFrame
    movie_index = movies_df[movies_df['processed_title'] == closest_title].index[0]

    # Query the genre model to find similar movies based on genres
    _, similar_genre_indices = genre_model.kneighbors(genre_features[movie_index], n_neighbors=6)

    # Get the top 5 similar movie titles
    top_recommendations = movies_df.loc[similar_genre_indices[0][1:], 'title'].values

    # Print the recommendations
    print("\nTop 5 movie recommendations similar to", closest_title + ":")
    for movie_title in top_recommendations:
        print(movie_title)
else:
    print("Movie not found in the dataset.")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
import streamlit as st

# Load the movies dataset
movies_df = pd.read_csv('movies.csv')

# Preprocess the movie titles for matching
movies_df['processed_title'] = movies_df['title'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))

# Create a TF-IDF vectorizer to convert movie genres into feature vectors
vectorizer = TfidfVectorizer(token_pattern=r'[^\s|,]+')

# Fit the vectorizer on the movie genres
genre_features = vectorizer.fit_transform(movies_df['genres'])

# Build the Nearest Neighbors model based on genre features
genre_model = NearestNeighbors(metric='cosine', algorithm='brute')
genre_model.fit(genre_features)

# Streamlit app
def movie_recommendation():
    st.title("Movie Recommendation")

    # Get input from the user for a movie title
    user_movie = st.text_input("Enter a movie title:")

    if st.button("Get Recommendations"):
        # Find the closest matching movie title
        closest_match = process.extractOne(user_movie, movies_df['processed_title'])

        if closest_match[1] >= 80:
            closest_title = closest_match[0]
            st.write(f"\nClosest matching movie title: {closest_title}")

            # Get the index of the closest matching movie in the DataFrame
            movie_index = movies_df[movies_df['processed_title'] == closest_title].index[0]

            # Query the genre model to find similar movies based on genres
            _, similar_genre_indices = genre_model.kneighbors(genre_features[movie_index], n_neighbors=5)

            # Get the top 5 similar movie titles
            top_recommendations = movies_df.loc[similar_genre_indices[0], 'title'].values

            # Display the recommendations
            st.subheader("Top 5 movie recommendations similar to " + closest_title + ":")
            for i, movie_title in enumerate(top_recommendations, start=1):
                st.write(f"{i}. {movie_title}")
        else:
            st.write("Movie not found in the dataset.")
            
# Run the Streamlit app
if __name__ == '__main__':
    movie_recommendation()

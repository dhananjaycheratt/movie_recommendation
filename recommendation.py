import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load the movies dataset
movies_df = pd.read_csv('movies.csv')

# Create a TF-IDF vectorizer to convert movie genres into feature vectors
vectorizer = TfidfVectorizer(token_pattern=r'[^\s|,]+')

# Fit the vectorizer on the movie genres
genre_features = vectorizer.fit_transform(movies_df['genres'])

# Build the Nearest Neighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(genre_features)

# Streamlit app
def movie_recommendation():
    st.title("Movie Recommendation")

    # Get input from the user for a movie title
    user_movie = st.text_input("Enter a movie title:")

    # Check if the movie title is in the dataset
    if user_movie in movies_df['title'].values:
        # Get the index of the movie in the DataFrame
        movie_index = movies_df[movies_df['title'] == user_movie].index[0]

        # Query the model to find similar movies
        _, similar_movie_indices = model.kneighbors(genre_features[movie_index], n_neighbors=5)

        # Get the top 5 similar movie titles
        top_recommendations = movies_df.loc[similar_movie_indices[0][1:], 'title'].values

        # Display the recommendations
        st.subheader("Top 5 movie recommendations similar to " + user_movie + ":")
        for i, movie_title in enumerate(top_recommendations, start=1):
            st.write(f"{i}. {movie_title}")
    else:
        st.write("Movie not found in the dataset.")

# Run the Streamlit app
if __name__ == '__main__':
    movie_recommendation()


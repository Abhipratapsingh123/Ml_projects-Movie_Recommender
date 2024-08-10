import streamlit as st
import pickle
import pandas as pd

# loading new_df dataset and similarity function

movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# vectorizing the text to create bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')
vectors = cv.fit_transform(movies['tags']).toarray()
from sklearn.metrics.pairwise import cosine_similarity 
similarity = cosine_similarity(vectors)

# ttle of the web site
st.title("Movie Recommender System")

selected_movie_name = st.selectbox('Which movie you want to see', movies['title'].values)


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies


if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)

st.button("Reset", type="primary")


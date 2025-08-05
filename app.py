import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    responce= requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=330f75c546b68b44299896d1f44f86b3')
    data= responce.json()
    return "https://image.tmdb.org/t/p/w500/"+data["poster_path"]


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
st.title("Next Watch")

selected_movie_name = st.selectbox('Which movie you want to see', movies['title'].values)


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_poster=[]
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
          # fetching poster from API
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies,recommended_movies_poster


if st.button("Recommend"):
    names,posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5= st.columns(5)

    # Add content to each column
    with col1:
        st.header(names[0])
        st.image(posters[0])

    with col2:
        st.header(names[1])
        st.image(posters[1])

    with col3:
        st.header(names[2])
        st.image(posters[2])
    
    with col4:
        st.header(names[3])
        st.image(posters[3])
    
    with col5:
        st.header(names[4])
        st.image(posters[4])

st.button("Reset", type="primary")


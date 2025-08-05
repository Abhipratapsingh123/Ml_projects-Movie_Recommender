import streamlit as st
import pickle
import pandas as pd
import requests

st.set_page_config(page_title="Beta Columns UI", layout="wide")

st.markdown(
    """
    <style>
        .main-title {
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #1F4E79;
        }
        .beta-title {
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .beta-desc {
            font-size: 16px;
            color: #444444;
        }
    </style>
    """,
    unsafe_allow_html=True
)





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
st.markdown('<div class="main-title">Next Watch</div>', unsafe_allow_html=True)

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
    names, posters = recommend(selected_movie_name)

    # Use spacing between elements and maintain uniform image size
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

    with col1:
        st.image(posters[0], use_column_width=True, caption=names[0])

    with col2:
        st.image(posters[1], use_column_width=True, caption=names[1])

    with col3:
        st.image(posters[2], use_column_width=True, caption=names[2])

    with col4:
        st.image(posters[3], use_column_width=True, caption=names[3])

    with col5:
        st.image(posters[4], use_column_width=True, caption=names[4])

# Add vertical space before Reset button
st.markdown("###")
st.button("Reset", type="primary")


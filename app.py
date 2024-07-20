import streamlit  as st
import pickle
import pandas as pd

# loading new_df dataset and similarity function

movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity =  pickle.load(open('similarity.pkl','rb'))

# ttle of the web site
st.title("Movie Recommender System")


selected_movie_name = st.selectbox('Which movie you want to see', movies['title'].values)


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies =[]
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies
    

if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)


st.button("Reset", type="primary")


import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import json
import warnings
import sys
import random
# Set page configuration
st.set_page_config(page_title='FLIXANALYTICS', page_icon='🎬', layout='wide')

# Load CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# User authentication functions
user_data_file = 'users_data.json'

def load_user_data():
    if os.path.exists(user_data_file):
        with open(user_data_file, 'r') as file:
            try:
                data = json.load(file)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    return {}

def save_user_data(user_data):
    with open(user_data_file, 'w') as file:
        json.dump(user_data, file, indent=4)

def register_user(username, pin):
    user_data = load_user_data()
    if username in user_data:
        st.warning("Username already exists. Please choose a different one.")
    else:
        user_data[username] = {'pin': pin, 'searches': []}
        save_user_data(user_data)
        st.success("Registration successful!")

def login_user(username, pin):
    user_data = load_user_data()
    if username in user_data and user_data[username]['pin'] == pin:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.success("Login successful!")
    else:
        st.error("Invalid username or PIN.")

def update_user_searches(username, movie_name):
    user_data = load_user_data()
    if username in user_data:
        user_data[username]['searches'].append(movie_name)
        user_data[username]['searches'] = user_data[username]['searches'][-5:]  # Keep only last 5 searches
        save_user_data(user_data)

# Load movie data
@st.cache_data
def load_data_model():
    movies = pd.read_csv('movies.csv')
    return movies

movies = load_data_model()

# Prepare recommendation models
@st.cache_resource
def prepare_recommendation_models():
    # Model 1: Using Actors, Tags, and Summary
    movies['tags_1'] = movies['Actors'] + movies['Tags'] + movies['Summary']
    movies_1 = movies[['Title', 'tags_1']].dropna()
    tfidf_1 = TfidfVectorizer()
    tfidf_matrix_1 = tfidf_1.fit_transform(movies_1['tags_1'])
    model_1 = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine')
    model_1.fit(tfidf_matrix_1)

    # Model 2: Using only Summary
    movies['tags_2'] = movies['Summary']
    movies_2 = movies[['Title', 'tags_2']].dropna()
    tfidf_2 = TfidfVectorizer()
    tfidf_matrix_2 = tfidf_2.fit_transform(movies_2['tags_2'])
    model_2 = NearestNeighbors(n_neighbors=11, algorithm='brute', metric='cosine')
    model_2.fit(tfidf_matrix_2)

    return model_1, tfidf_matrix_1, movies_1, model_2, tfidf_matrix_2, movies_2

model_1, tfidf_matrix_1, movies_1, model_2, tfidf_matrix_2, movies_2 = prepare_recommendation_models()

# Recommendation function
def get_movie_recommendations(title, model, tfidf_matrix, movies_df, n_recommendations=10):
    movie_index = movies_df[movies_df['Title'] == title].index[0]
    distances, indices = model.kneighbors(tfidf_matrix[movie_index], n_neighbors=n_recommendations + 1)
    recommended_movies = [movies_df.iloc[i]['Title'] for i in indices.flatten()[1:]]
    return recommended_movies

# YouTube functions
def extract_youtube_id(url):
    if 'youtube.com/watch?v=' in url:
        return url.split('watch?v=')[-1]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[-1]
    return None

def generate_youtube_iframe(video_url):
    video_id = extract_youtube_id(video_url)
    if video_id:
        return f'<iframe width="300" height="169" src="https://www.youtube.com/embed/{video_id}" title="Video" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>'
    return None

# Function to display movie information with trailer
def display_movie_info(movie_title):
    movie_info = movies.loc[movies['Title'] == movie_title].iloc[0]
    st.write(f"**Title:** {movie_info['Title']}\n**Rating:** {movie_info['IMDb Score']}\n**Genre:** {movie_info['Genre']}")
    st.image(movie_info['Image'], width=150)
    
    video_url = movie_info["TMDb Trailer"]
    iframe_html = generate_youtube_iframe(video_url)
    if iframe_html:
        with st.expander(f"Watch Trailer for {movie_info['Title']}", expanded=False):
            st.markdown(iframe_html, unsafe_allow_html=True)
    else:
        st.write("Trailer URL is not available.")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Sidebar for user login, registration, and logout
st.sidebar.header("User Login / Registration")
if not st.session_state['logged_in']:
    username = st.sidebar.text_input("Username")
    pin = st.sidebar.text_input("4-digit PIN", type="password")
    
    # Check if user is already registered
    user_data = load_user_data()
    if username and username in user_data:
        st.sidebar.warning("User is already registered. Please login.")
    else:
        if st.sidebar.button("Register"):
            register_user(username, pin)
    
    if st.sidebar.button("Login"):
        login_user(username, pin)
else:
    with st.sidebar.expander("Account", expanded=True):
        st.sidebar.success(f"Welcome, {st.session_state['username']}!")
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.sidebar.success("Logged out successfully!")

# Filter options
st.sidebar.header("Series or Movie")
genres = st.sidebar.multiselect("Select Series or Movie", movies['Series or Movie'].unique())#filter by shows
min_rating = st.sidebar.slider("Minimum IMDb Rating", 0.0, 10.0, 0.0, 0.1)
max_rating = st.sidebar.slider("Maximum IMDb Rating", 0.0, 10.0, 10.0, 0.1)
msg=st.sidebar.info("A product of Flixanalytics")
# Apply filters
filtered_movies = movies
if genres:
    filtered_movies = filtered_movies[filtered_movies['Series or Movie'].apply(lambda x: any(genre in x for genre in genres))]
filtered_movies = filtered_movies[(filtered_movies['IMDb Score'] >= min_rating) & (filtered_movies['IMDb Score'] <= max_rating)]

# Main app content
if st.session_state['logged_in']:
    st.title('NETFLIX MOVIE RECOMMENDER APP')

    # Display top picks for the user
    user_data = load_user_data()
    username = st.session_state['username']
    user_searches = user_data.get(username, {}).get('searches', [])

    if user_searches:
        st.header(f"Top Picks for {username}")
        last_searched_movie = user_searches[-1]
        model_pick=random.choice([model_1,model_2])
        movies_pick= movies_1
        if model_pick==model_1:
            movies_pick=movies_1
        else:
            movies_pick=movies_2

        mat= tfidf_matrix_1
        if model_pick==model_1:
            mat=tfidf_matrix_1
        else:
            mat=tfidf_matrix_2
    
        recommendations = get_movie_recommendations(last_searched_movie, model_pick,mat, movies_pick, 5)
        for movie in recommendations:
            display_movie_info(movie)#model

    # Movie selection box
    st.write("---")
    st.title("Search for Movies and Series")
    selected_movie_name = st.selectbox("Select A Movie", filtered_movies["Title"].values)# to be changed

    # Slider for number of recommendations
    n_recommendations = st.slider("Select number of recommendations", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        st.title(f"{selected_movie_name} Movie Info")
        
        movie_info = movies.loc[movies['Title'] == selected_movie_name].iloc[0]
        title = movie_info["Title"]
        ratings = movie_info["IMDb Score"]
        image_url = movie_info["Image"]
        genre = movie_info["Genre"]
        video_url = movie_info["TMDb Trailer"]
        
        st.write(f"**Title:** {title}\n**Rating:** {ratings}\n**Genre:** {genre}")
        st.image(image_url, width=300)
        
        iframe_html = generate_youtube_iframe(video_url)
        if iframe_html:
            with st.expander("Watch Trailer", expanded=False):
                st.markdown(iframe_html, unsafe_allow_html=True)
        else:
            st.write("Trailer URL is not available.")
        
       
        st.write("---")
        st.title("Other People Watch This")
        recommendations_2 = get_movie_recommendations(selected_movie_name, model_2, tfidf_matrix_2, movies_2, n_recommendations)
        cols = st.columns(n_recommendations)
        for i, movie in enumerate(recommendations_2):
            with cols[i]:
                display_movie_info(movie)

        st.write("---")        

        st.title("Other Users also Watch...")
        recommendations_1 = get_movie_recommendations(selected_movie_name, model_1, tfidf_matrix_1, movies_1, n_recommendations)
        cols = st.columns(n_recommendations)
        for i, movie in enumerate(recommendations_1):
            with cols[i]:
                display_movie_info(movie)
        

        update_user_searches(st.session_state['username'], selected_movie_name)

    # Display top-rated movies
    top_rated_movies = filtered_movies.sort_values(by="IMDb Score", ascending=False).head(20)
    st.title("Top Rated Movies & Series")

    for index, row in top_rated_movies.iterrows():
        st.markdown(
            f"""
            <div>
                <img src="{row['Image']}" alt="{row['Title']}" style="float: left; margin-right: 10px; width: 150px;">
                <p title="{row['Summary']}">
                    <strong>{row['Title']}</strong><br>
                    Rating: {row['IMDb Score']}<br>
                    {row['Summary']}<br>
                    Genre: {row['Genre']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # About Section
    st.write("---")
    st.title("About FlixAnalytics")

    st.write("""
    **FlixAnalytics** is a leading data science company specializing in providing advanced analytics solutions for the entertainment industry. Our team of data scientists and analysts is dedicated to helping you discover and enjoy the best movies and series through cutting-edge recommendation systems and insightful data analysis.

    **Contact Us:**
    - Email: [flixanalytics@yahoo.com](mailto:flixanalytics@yahoo.com)

    **Our Team:**
    - The app is created by the FlixAnalytics Data Science Team, 2024.
    """)

else:
    st.write("Please log in to search for movies and see recommendations.")
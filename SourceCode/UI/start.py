import streamlit as st
from streamlit_option_menu import option_menu
from profile import main as profile_main
from pages.search import main as search_main
from streamlit import session_state as ss
import pickle 
import pandas as pd
st.set_page_config(layout="wide")
@st.cache_data
def reduced_game_load():
    with open("SourceCode/UI/data/reduced_games_with_clusters.pkl","rb") as f:
        return pickle.load(f)
@st.cache_data
def load_game_data():
    with open("SourceCode/UI/data/50000_games.pkl","rb") as game_pkl:
        data=pickle.load(game_pkl)
    return data
@st.cache_data
def load_ratings_data():
    with open("SourceCode/CB/data/processed_rec.pkl","rb") as f:
        data=pickle.load(f)
    return data
ss.game_data=load_game_data()
ss.rating_data=load_ratings_data()
@st.cache_data
def load_users_list():
    user_list=ss.rating_data["user_id"].unique().tolist()
    return user_list
@st.cache_data
def load_pca_matrix():
    with open("SourceCode/UI/data/pca_matrix.pkl","rb") as f:
        data=pickle.load(f)
    return data
@st.cache_data
def tag_sim_load():
    with open("SourceCode/UI/data/sim_matrix.pkl","rb") as tag_sim_pkl:
        return pickle.load(tag_sim_pkl)
ss.tag_sim=tag_sim_load()
ss.reduced_games=reduced_game_load()
ss.user_list=load_users_list()
ss.pca_matrix=load_pca_matrix()
# ss.rated={id:None for id in ss.game_data["app_id"]}
# ss.liked=set()
# ss.disliked=set()
st.markdown("<h1 style='text-align: center; color: black;'>COOL Game Recommender</h1>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2.5, 1, 2])  # Adjust column widths for better centering

# Add empty content to columns 1 and 3 for spacing
col1.write("")
col3.write("")

# Create a large button with centered text in the center column
with col2:
    submit_button=st.page_link("pages/main.py",label="LET'S GO")
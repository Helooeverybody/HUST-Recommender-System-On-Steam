import streamlit as st
from streamlit import session_state as ss
import pickle 
import pandas as pd
from pages.modules.display_poster import display_poster
from pages.modules.reg_rec import *
import streamlit as st
def main():
    users=ss.user_list
    ratings=ss.rating_data
    chosen_user=int(st.selectbox("WHO ARE YOU",users))
    def top_favorite_games(chosen_user):
        top_favorite=ratings[ratings["user_id"]==chosen_user].groupby("app_id")["rating"].mean().sort_values(ascending=False).head(12).to_dict().keys()
        return top_favorite
    def show(game_list):
        col1, col2,col3 = st.columns(3,gap="large")
        i=0
        for game in game_list:
            if i%3==0:
                display_poster(game,col1,display_like=False)
            elif i%3==1:
                display_poster(game,col2,display_like=False)
            else:
                display_poster(game,col3,display_like=False)
            i+=1
    def top_anti_games(chosen_user):
        top_anti=ratings[ratings["user_id"]==chosen_user].groupby("app_id")["rating"].mean().sort_values(ascending=True).head(6).to_dict().keys()
        return top_anti
    #if st.button("Show profile"):
    st.markdown("<h1 style='text-align: center; color: black;'>Your Top Favorite Games</h1>", unsafe_allow_html=True)
    show(top_favorite_games(chosen_user))
    st.markdown("<h1 style='text-align: center; color: black;'>Top Games You Don't Like</h1>", unsafe_allow_html=True)
    show(top_anti_games(chosen_user))
    st.markdown("\n") 
    st.markdown("\n")
    col1,col2 = st.columns([7,3])
    with col1:
        chosen_algorithm=st.selectbox("Choose the algorithm",["Ridge","Lasso","Regression Random Forest"])
    rec=st.button("Recommend")
    if rec:
        if chosen_algorithm=="Ridge":
            rec_games=rec_by_regression(chosen_user,0,games=ss.reduced_games,rec=ratings,pca_matrix=ss.pca_matrix)
        elif chosen_algorithm=="Lasso":
            rec_games=rec_by_regression(chosen_user,1,games=ss.reduced_games,rec=ratings,pca_matrix=ss.pca_matrix)
        else:
            rec_games=rec_by_regression(chosen_user,2,games=ss.reduced_games,rec=ratings,pca_matrix=ss.pca_matrix)
        show(rec_games)
if __name__=="__main__":
    main()
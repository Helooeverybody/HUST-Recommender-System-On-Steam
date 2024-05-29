import streamlit as st
from streamlit import session_state as ss
import pickle 
import random
import pandas as pd
from pages.modules.display_poster import display_poster
import streamlit as st
def main():
    games=ss.reduced_games
    col1,col2=st.columns([3,7])
    chosen_engine=col1.selectbox("Select the algortihm",["Cosine similarity KNN","KMeans"])
    st.markdown("\n\n")
    chosen_game=st.selectbox("Select the game you interested in",games["title"])
    def recommend_by_KNN(game):
        index=games[games["title"]==game].index[0]
        top_rec=sorted(list(enumerate(ss.tag_sim[index])),key=lambda x:x[1],reverse=True)
        top_rec=[a[0] for a in top_rec[0:15]]
        return games.loc[top_rec]["app_id"].tolist()
    def recommend_by_cluster(game):
        row=games.loc[games["title"]==game]
        this_game_id=row.app_id.item()
        cluster=row.cluster.item()
        top_terms=row.top_terms.item()
        games_in_cluster=games[games["cluster"]==cluster]["app_id"].tolist()
        n=15 if len(games_in_cluster)>15 else len(games_in_cluster)
        random.seed(42)
        top_rec=random.sample(games_in_cluster,n)
        if this_game_id not in top_rec:
            top_rec.insert(0,this_game_id)
            top_rec.pop()
        else:
            top_rec.remove(this_game_id)
            top_rec.insert(0,this_game_id)
        return (top_terms,top_rec)
    if "rec_clicked" not in ss:
        ss.rec_clicked=False
    def show_recommend(rec_list):
        col1, col2,col3 = st.columns(3,gap="large")
        i=0
        for candidate in rec_list:
            if i%3==0:
                display_poster(candidate,col1) #games.iloc[candidate[0]]["app_id"]
            elif i%3==1:
                display_poster(candidate,col2)
            else:
                display_poster(candidate,col3)
            i+=1
    def rec_callback():
        ss.rec_clicked=True
    if st.button("Search"):
        ss.rec_clicked = True
    if ss.rec_clicked:
        if chosen_engine=="Cosine similarity KNN":
            rec_list=recommend_by_KNN(chosen_game)
        else:
            top_terms,rec_list=recommend_by_cluster(chosen_game)
            st.subheader("Keywords:")
            st.markdown(f"\t {top_terms}")
        show_recommend(rec_list)
if __name__=="__main__":
    main()
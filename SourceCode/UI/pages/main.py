import streamlit as st
from streamlit_option_menu import option_menu
#from profile import main as profile_main
from pages.search import main as search_main
from pages.recommendation import main as rec_main
from streamlit import session_state as ss
import pickle 
import pandas as pd
def main():
    st.markdown("<h1 style='text-align: center; color: black;'>Game Recommender</h1>", unsafe_allow_html=True)
    selected=option_menu(
        menu_title=None,
        options=["Search","Recommendation"],
        icons=["search","controller"],
        orientation="horizontal"
    )
    if selected=="Search":
        search_main()
    #elif selected=="Profile":
       # profile_main()
    elif selected=="Recommendation":
        rec_main()
if __name__=="__main__":
    main()


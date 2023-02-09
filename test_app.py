import streamlit as st
import pandas as pd
import seaborn as sns

st.title("Revathi-Movie Recommender System")
data=sns.load_dataset("movie_item_based")

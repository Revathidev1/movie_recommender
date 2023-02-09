import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity

# To prepare requirements.txt file
# print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


# rating_final.csv
url = 'https://drive.google.com/file/d/1ptu4AlEXO4qQ8GytxKHoeuS1y4l_zWkC/view?usp=sharing' 
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
frame = pd.read_csv(path)

# 'geoplaces2.csv'
url = 'https://drive.google.com/file/d/1ee3ib7LqGsMUksY68SD9yBItRvTFELxo/view?usp=sharing' 
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
geodata = pd.read_csv(path, encoding = 'CP1252') # change encoding to 'mbcs' in Windows

places = geodata[['placeID', 'name']]

users_items = pd.pivot_table(data=frame, 
                                 values='rating', 
                                 index='userID', 
                                 columns='placeID')

users_items.fillna(0, inplace=True)

user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)


def weighted_user_rec(user_id, n):
  weights = (user_similarities.query("userID!=@user_id")[user_id] / sum(user_similarities.query("userID!=@user_id")[user_id]))
  not_visited_restaurants = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
  weighted_averages = pd.DataFrame(not_visited_restaurants.T.dot(weights), columns=["predicted_rating"])
  recommendations = weighted_averages.merge(places, left_index=True, right_on="placeID")
  top_recommendations = recommendations.sort_values("predicted_rating", ascending=False).head(n)
  return top_recommendations


def chat_bot():
    print("Hi! I'm your personal recommender. Tell me your userID.")
    user_id = input()
    recom = weighted_user_rec(user_id, 1)
    print(f"You will probably like the restaurant: {list(recom['name'])[0]}")
    
chat_bot()

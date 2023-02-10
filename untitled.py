import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity


# Create the similarity matrix
# In 3 simple steps:

# Create the big users-items table

# Replace NaNs with zeros

# Compute pairwise cosine similarities

# 1. Create the big users-items table.
# We are just reshaping (pivoting) the data, so that we have users as rows and restaurants as columns. We need the data to be in this shape to compute similarities between users in the next step.


df_links = pd.read_csv(r'links.csv')
df_movies = pd.read_csv(r'movies.csv')
df_ratings = pd.read_csv(r'ratings.csv')
df_tags = pd.read_csv(r'tags.csv')

df_ratings

# To prepare requirements.txt file
print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))


df_ratings.info()

users_items = pd.pivot_table(data=df_ratings, 
                                 values='rating', 
                                 index='userId', 
                                 columns='movieId')


users_items.head()

# ### 2. Replace NaNs with zeros
# The cosine similarity can't be computed with NaN's

users_items.fillna(0, inplace=True)
users_items.head()

### 3. Compute cosine similarities

from sklearn.metrics.pairwise import cosine_similarity

user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)
user_similarities.head()

### Building the recommender step by step:
# Let's focus on one random user (user 1) and compute the recommendations only for this user, as an example. Then, we will build a function that can compute recommendations for any users. We will follow these steps:

# Compute the weights.

# Find movies user 1 has not rated.

# Compute the ratings user 1 would give to those unrated movies.

# Find the top 5 movies from the rating predictions.

### 1. Compute the weights
# Here we will exclude user 1 using .query().

# compute the weights for one user
userId = 1

weights = (
    user_similarities.query("userId!=@userid")[userId] / sum(user_similarities.query("userId!=@userId")[userId])
          )
weights.head(6)

weights.sum()

### 2. Find restaurants user 1 has not rated.
# We will exclude our user, since we don't want to include them on the weights.

users_items.loc[userId,:]==0

# select restaurants that the inputed user has not visited
not_visited_movies = users_items.loc[users_items.index!=userId, users_items.loc[userId,:]==0]
not_visited_movies.T

### 3. Compute the ratings user 1 would give to those unrated restaurants.

not_visited_movies.T.dot(weights)

# dot product between the not-visited-restaurants and the weights
weighted_averages = pd.DataFrame(not_visited_movies.T.dot(weights), columns=["predicted_rating"])
weighted_averages

### 4. Find the top 5 movies from the rating predictions

recommendations = weighted_averages.merge(df_movies, left_index=True, right_on="movieId")
recommendations.sort_values("predicted_rating", ascending=False).head()

### Function:
# Make a function that recommends the top n movies to an inputted userId

def user_movie_similarity(userId=1,n=10,user_movie=users_items,movie_names=df_movies):
  userId=int(input("What is your userId "))
  n=int(input("How many movies do you want to get "))
  user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)
  weights = (
    user_similarities.query("userId!=@userId")[userId] / sum(user_similarities.query("userId!=@userId")[userId])
          )
  not_visited_movies = users_items.loc[users_items.index!=userId, users_items.loc[userId,:]==0]
  weighted_averages = pd.DataFrame(not_visited_movies.T.dot(weights), columns=["predicted_rating"])
  recommendations = weighted_averages.merge(df_movies, left_index=True, right_on="movieId")
  return recommendations.sort_values("predicted_rating", ascending=False).head(n)
  

user_movie_similarity()

st.title("Revathi-Movie Recommender System")


movie_list = df_movies['title'].values
movie_list # list of all movies in our dataset

selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list )
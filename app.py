import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
import random
from recsys_utils import load_precalc_params_small, load_ratings_small, load_Movie_List_pd, normalizeRatings

# Load data
X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

# Set Streamlit page configuration
st.set_page_config(layout="wide")
st.markdown("""<h1 style='color:red;text-align:center;'>Movie Recommender System</h1>""", unsafe_allow_html=True)
st.markdown("""<h3 style='color:white;text-align:center;'>Uses Collaborative Filtering</h3>""", unsafe_allow_html=True)
st.markdown("""<h4 style='color:white;text-align:center;'>Made by Damanjit</h4>""", unsafe_allow_html=True)

# Load movie list
movieList, movieList_df = load_Movie_List_pd()

# Session state to store options
st.session_state.options = []

# Form to select movies
with st.form("my_form"):
    options = []
    for i in range(10):
        option = st.selectbox(
            f"Enter your Movie {i + 1}",
            movieList
        )
        options.append(option)
    submitted = st.form_submit_button('Submit my picks')
    if submitted:
        st.session_state.options = options

# Display selected movies
st.write("Selected movies:")
st.write(st.session_state.options)

# Process ratings
my_ratings = np.zeros(num_movies)
for i, option in enumerate(st.session_state.options):
    movie_id = movieList_df.index[movieList_df['title'] == option].tolist()[0]
    if i < 4:
        my_ratings[movie_id] = 5  # Most favorite
    elif 4 <= i < 7:
        my_ratings[movie_id] = random.randint(3, 4)  # Favorite
    else:
        my_ratings[movie_id] = random.randint(1, 2)  # Least favorite

# Normalize ratings
Y, R = load_ratings_small()
Y = np.c_[my_ratings, Y]
R = np.c_[(my_ratings != 0).astype(int), R]
Ynorm, Ymean = normalizeRatings(Y, R)

# Model setup and training
tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
lambda_ = 1
for iter in range(200):
    with tf.GradientTape() as tape:
        cost_value = (tf.linalg.matmul(X, tf.transpose(W)) + b - Ynorm) * R
        cost_value = 0.5 * tf.reduce_sum(cost_value ** 2) + (lambda_ / 2) * (tf.reduce_sum(X ** 2) + tf.reduce_sum(W ** 2))
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))

# Make predictions
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
pm = p + Ymean
my_predictions = pm[:, 0]
ix = tf.argsort(my_predictions, direction='DESCENDING')

# Display predictions
st.markdown("""<h1 style='color:red;text-align:center;'>Predictions</h1>""", unsafe_allow_html=True)
st.dataframe(movieList_df.loc[ix[:300]].loc[movieList_df["number of ratings"] > 20].sort_values("mean rating", ascending=False))

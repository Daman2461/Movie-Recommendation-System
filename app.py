import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import *
import pandas as pd

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

def cofi_cost_func(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_ / 2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

movieList, movieList_df = load_Movie_List_pd()

if 'selected_movies' not in st.session_state:
    st.session_state.selected_movies = np.random.choice(range(num_movies), 10, replace=False)
    st.session_state.my_ratings = np.zeros(num_movies)

st.title("Movie Recommendation System")
st.write("Made by- Damanjit Singh")


if st.button("Select 10 Random Movies for Rating"):
    st.session_state.selected_movies = np.random.choice(range(num_movies), 10, replace=False)
    st.session_state.my_ratings = np.zeros(num_movies)

st.write("Please rate the following movies (0-5):")
for movie_id in st.session_state.selected_movies:
    st.session_state.my_ratings[movie_id] = st.slider(movieList_df.loc[movie_id, "title"], 0, 5)

st.write('\nNew user ratings:\n')
for i in range(len(st.session_state.my_ratings)):
    if st.session_state.my_ratings[i] > 0:
        st.write(f'Rated {st.session_state.my_ratings[i]} for {movieList_df.loc[i,"title"]}')

if np.count_nonzero(st.session_state.my_ratings) == 10:
    Y, R = load_ratings_small()
    Y = np.c_[st.session_state.my_ratings, Y]
    R = np.c_[(st.session_state.my_ratings != 0).astype(int), R]
    Ynorm, Ymean = normalizeRatings(Y, R)
    num_movies, num_users = Y.shape
    num_features = 100

    tf.random.set_seed(1234)
    W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
    b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')

    optimizer = keras.optimizers.Adam(learning_rate=1e-1)

    iterations = 200
    lambda_ = 1
    for iter in range(iterations):
        with tf.GradientTape() as tape:
            cost_value = cofi_cost_func(X, W, b, Ynorm, R, lambda_)

        grads = tape.gradient(cost_value, [X, W, b])
        optimizer.apply_gradients(zip(grads, [X, W, b]))

        if iter % 20 == 0:
            st.write(f"Training loss at iteration {iter}: {cost_value:0.1f}")

    p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()
    pm = p + Ymean
    my_predictions = pm[:, 0]
    ix = np.argsort(my_predictions)[::-1]  # Sort predictions in descending order

    recommended_movies = [(movieList[j], my_predictions[j]) for j in ix[:10]]
    recommended_df = pd.DataFrame(recommended_movies, columns=['Movie', 'Predicted Rating'])

    st.write('\nRecommended Movies as per Predictions:\n')
    st.dataframe(recommended_df)

    st.write('\n\nOriginal vs Predicted ratings:\n')
    for i in range(len(st.session_state.my_ratings)):
        if st.session_state.my_ratings[i] > 0:
            st.write(f'Original {st.session_state.my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')

    filter = (movieList_df["number of ratings"] > 20)
    movieList_df["pred"] = my_predictions
    movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])
    recommended_movies = movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False)

    st.write("\n\nRecommended Movies (Top 300 by Prediction and at least 20 ratings):")
    st.dataframe(recommended_movies)

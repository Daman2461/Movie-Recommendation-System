import numpy as np
import tensorflow as tf

from recsys_utils import *
import pandas as pd
import streamlit as st
import random
#Load data
st. set_page_config(layout="wide")
st.write(""" <h1> <b style="color:red"> Movie Reccomender System</b> </h1>""",unsafe_allow_html=True)
st.write(""" <h6> <b style="color:white"> Uses Collaborative Filtering</b> </h6>""",unsafe_allow_html=True)
st.write(""" <h7> <b style="color: white"> Made by Damanjit</b> </h7>""",unsafe_allow_html=True)

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()



def cofi_cost_func(X, W, b, Y, R, lambda_):
    
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J


movieList, movieList_df = load_Movie_List_pd()

my_ratings = np.zeros(num_movies) 
movieList_df['date'] = movieList_df['title'].str.extract(r'\((\d{4})\)', expand=False).astype(int)
movieList_df['ID'] = movieList_df.index




st.session_state.options = []
st.session_state.id = []
with st.form("my_form"):
    options = []
    for i in range(4):
        option = st.selectbox(
            "Enter your Most Favorite Movies",
            movieList,
            key=f'selectbox_{i}',placeholder="",index=None
        )
        options.append(option)
    for i in range(4,7):
        option = st.selectbox(
            "Enter your Favorite Movies",
            movieList,
            key=f'selectbox_{i}',placeholder="",index=None
        )
        options.append(option)
    for i in range(7,10):
        option = st.selectbox(
            "Enter your Least Favorite Movies",
            movieList,
            key=f'selectbox_{i}',placeholder="",index=None
        )
        options.append(option)
    
    submitted = st.form_submit_button('Submit my picks')
    if submitted:
        st.session_state.options = options

st.write("Selected movies:")
for option in st.session_state.options:
    st.session_state.id.append(movieList_df['ID'][movieList_df['title']==option])
    
   



for i in st.session_state.id[0:4]:
    my_ratings[i] = 5
for i in st.session_state.id[4:7]:
    my_ratings[i] = random.randint(3, 4)
for i in st.session_state.id[7:10]:
    my_ratings[i] = random.randint(1,2)


my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

print('\nNew user ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0 :
        print(f'Rated {my_ratings[i]} for  {movieList_df.loc[i,"title"]}');


        
# Reload ratings
Y, R = load_ratings_small()

# Add new user ratings to Y 
Y = np.c_[my_ratings, Y]

# Add new user indicator matrix to R
R = np.c_[(my_ratings != 0).astype(int), R]

# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)

#  Useful Values
num_movies, num_users = Y.shape
num_features = 100

# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
lambda_ = 1
for iter in range(iterations):
    # Use TensorFlow’s GradientTape
    # to record the operations used to compute the cost 
    with tf.GradientTape() as tape:

        # Compute the cost (forward pass included in cost)
        cost_value = cofi_cost_func(X, W, b, Ynorm, R, lambda_)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss
    grads = tape.gradient( cost_value, [X,W,b] )

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients( zip(grads, [X,W,b]) )

    # Log periodically.
    if iter % 20 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

#restore the mean
pm = p + Ymean

my_predictions = pm[:,0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

for i in range(17):
    j = ix[i]
    if j not in my_rated:
        print(f'Predicting rating {my_predictions[j]:0.2f} for movie {movieList[j]}')

print('\n\nOriginal vs Predicted ratings:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Original {my_ratings[i]}, Predicted {my_predictions[i]:0.2f} for {movieList[i]}')
        

filter=(movieList_df["number of ratings"] > 20)
movieList_df["pred"] = my_predictions
movieList_df = movieList_df.reindex(columns=["pred", "mean rating", "number of ratings", "title"])

st.write(""" <h1> <b style="color:red"> Predictions</b> </h1>""",unsafe_allow_html=True)
st.dataframe(movieList_df.loc[ix[:300]].loc[filter].sort_values("mean rating", ascending=False) )



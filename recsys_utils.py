import numpy as np
import pandas as pd
from numpy import loadtxt

def normalizeRatings(Y, R):
     
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def load_precalc_params_small():

    file = open('./small_movies_X.csv', 'rb')
    X = loadtxt(file, delimiter = ",")

    file = open('./small_movies_W.csv', 'rb')
    W = loadtxt(file,delimiter = ",")

    file = open('./small_movies_b.csv', 'rb')
    b = loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    file = open('./small_movies_Y.csv', 'rb')
    Y = loadtxt(file,delimiter = ",")

    file = open('./small_movies_R.csv', 'rb')
    R = loadtxt(file,delimiter = ",")
    return(Y,R)

def load_Movie_List_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    df = pd.read_csv('./small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return mlist, df






# Movie-Recommendation-System


## Overview
The Movie Recommendation System leverages collaborative filtering to suggest movies based on user preferences and viewing history. The system is deployed using Streamlit to provide an interactive and user-friendly experience.

## Features
- **Collaborative Filtering:** Utilizes collaborative filtering techniques to generate movie recommendations.
- **User Input:** Allows users to rate their favorite, neutral, and least favorite movies.
- **Interactive UI:** Deployed on Streamlit for a seamless and interactive user experience.

## Libraries Used
- **NumPy:** For numerical operations and data manipulation.
- **TensorFlow:** For building and training the recommendation model.
- **Pandas:** For data handling and manipulation.
- **Streamlit:** For creating an interactive web application.

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/Daman2461/Movie-Recommendation-System.git
    cd Movie-Recommendation-System
    ```

2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**
    ```bash
    pip install numpy tensorflow pandas streamlit
    ```

4. **Run the Streamlit app**
    ```bash
    streamlit run app.py
    ```

## How It Works

1. **Load Data:** Loads movie data and user ratings.
2. **User Interaction:** Allows users to select their favorite, neutral, and least favorite movies.
3. **Model Training:** Uses TensorFlow to train a collaborative filtering model based on user ratings.
4. **Recommendations:** Provides movie recommendations based on the trained model and displays them using Streamlit.

## Usage
- **Select Movies:** Use the Streamlit interface to enter your movie preferences.
- **Submit Ratings:** Submit your movie choices to update ratings and generate recommendations.
- **View Predictions:** Check the predicted ratings for movies and see the top recommendations.

## Example Output
After selecting and submitting movie preferences, the system will display personalized movie recommendations based on the trained collaborative filtering model.

**GitHub:** [Movie Recommendation System](https://github.com/Daman2461/Movie-Recommendation-System)
**Online:** [Movie Recommendation System]((https://movie-recon.streamlit.app))
## Author
- **Damanjit:** Developer and creator of the Movie Recommendation System.

---
 

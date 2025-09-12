import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load and Train Model
# -----------------------------
@st.cache_data
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("netflix_titles.csv")  # <-- Make sure this CSV is in the same folder

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    df = df.drop(columns=['director', 'show_id', 'title', 'cast', 'date_added', 'description'])
    df = df.fillna("Unknown")

    # Encode categorical features
    le_rating = LabelEncoder()
    df['rating'] = le_rating.fit_transform(df['rating'])

    le_country = LabelEncoder()
    df['country'] = le_country.fit_transform(df['country'])

    # Extract numeric duration (e.g., "90 min" -> 90, "2 Seasons" -> 2)
    df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
    df['duration_num'] = df['duration_num'].fillna(0)

    # TF-IDF vectorization for genres
    tfidf = TfidfVectorizer(stop_words="english", max_features=500)
    genre_features = tfidf.fit_transform(df['listed_in'])

    # Target variable: Movie=0, TV Show=1
    y = df['type'].map({'Movie': 0, 'TV Show': 1})

    # Combine numeric and TF-IDF features
    X = np.hstack([
        df[['release_year', 'rating', 'country', 'duration_num']].values,
        genre_features.toarray()
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train logistic regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    return model, le_rating, le_country, tfidf

# Load model and encoders
model, le_rating, le_country, tfidf = load_and_train_model()

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("🎬 Netflix Movie vs TV Show Predictor")
st.write("Enter the details below to predict whether a title is a Movie or a TV Show.")

# User Inputs
release_year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2020)
rating = st.text_input("Rating (e.g., PG, TV-MA, R)", "PG")
country = st.text_input("Country", "United States")
duration = st.text_input("Duration (e.g., 90 min or 2 Seasons)", "90 min")
genres = st.text_area("Genres (comma separated)", "Dramas, International Movies")

# -----------------------------
# Prediction Logic
# -----------------------------
if st.button("Predict"):
    # Encode rating safely
    rating_val = le_rating.transform([rating])[0] if rating in le_rating.classes_ else 0

    # Encode country safely
    country_val = le_country.transform([country])[0] if country in le_country.classes_ else 0

    # Extract numeric duration safely
    try:
        duration_num = int(''.join([c for c in duration if c.isdigit()]))
    except:
        duration_num = 0

    # Preprocess genres for TF-IDF
    genres_processed = genres.replace(",", " ")
    genre_vec = tfidf.transform([genres_processed]).toarray()

    # Combine all features into a single array
    features = np.hstack([[release_year, rating_val, country_val, duration_num], genre_vec[0]])

    # Make prediction
    pred = model.predict([features])[0]
    result = "🎥 Movie" if pred == 0 else "📺 TV Show"

    st.success(f"Prediction: {result}")

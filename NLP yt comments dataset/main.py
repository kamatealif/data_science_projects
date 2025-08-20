import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import random
import os




def generate_data():
    toxic_comments = [
        "I hope you die",
        "I hate you",
        "Fuck you",
        "You are a piece of s***",
        "I hope you get cancer and die",
        "You are a b****",
        "I hate your guts",
        "I hate your video",
        "I hate you so much",
        "I hate everything about you",
    ]

    supportive_comments = [
        "I love you",
        "You are the best",
        "I love your videos",
        "You are so talented",
        "I love your style",
        "You are so smart",
        "I love your sense of humor",
        "You are so beautiful",
        "I love your music",
        "You are so kind",
    ]
    data = []
    for i in range(100):
        data.append({'comment':random.choice(toxic_comments), 'label': 'toxic'})

        data.append({'comment':random.choice(supportive_comments), 'label': 'supportive'})
    
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)
    print("✅ Data Generated and saved to data.csv")

def main():
    print("Hello from nlp-yt-comments-dataset!")

@st.cache_resource
def load_model():
    # create dataset if missing
    if not os.path.exists("data.csv"):
        generate_data()

    try:
        df = pd.read_csv("data.csv")
    except Exception as e:
        # fallback: regenerate and retry
        generate_data()
        df = pd.read_csv("data.csv")

    # use a solver and larger max_iter to avoid convergence issues on small data
    model = Pipeline([
        ("tfid", TfidfVectorizer()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=200))
    ])
    model.fit(df['comment'], df['label'])

    return model

if __name__ == "__main__":
    main()

    # generate_data() #use this function to generate the data 
    model = load_model();
  
    st.title("""
    # NLP Youtube Comments classifier""")

    st.write("""Classify your comment as toxic or supportive""")
    comment = st.text_area("Enter a Youtube Comment")

    if comment:
        prediction = model.predict([comment])[0]
        if prediction =='toxic':
            st.error("This comment is likely Toxic")
        else:
            st.success("This comment is **Supportive**")


    
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import random




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


if __name__ == "__main__":
    main()

    # generate_data() #use this function to generate the data 

    df = pd.read_csv("data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['label'], test_size=0.2, random_state= 45)

    model = Pipeline(
        [('tfid', TfidfVectorizer()),
        ('clf', LogisticRegression())]
    )

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Model Accuracy: {round(acc * 100 , 2)}%")
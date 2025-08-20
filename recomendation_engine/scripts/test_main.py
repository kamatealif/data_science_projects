import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from main import get_recommendation


def smoke_test():
    repo_root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(repo_root / 'books.csv')
    df['title'] = df['title'].str.strip()
    df['description'] = df['description'].fillna('')

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_metrix = vectorizer.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_metrix, tfidf_metrix)
    indicies = pd.Series(df.index, index=df['title'].str.lower())

    # pick a title that exists
    title = df['title'].iloc[0]
    res = get_recommendation(title, df, cosine_sim, indicies)
    print('Result type:', type(res))
    print(res.head())

if __name__ == '__main__':
    smoke_test()

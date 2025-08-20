import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


def get_recommendation(title: str, df: pd.DataFrame, cosine_sim, indices: pd.Series):
    """Return top 5 recommendations (title+author) for a given title.

    - title: user input title (case-insensitive)
    - df: dataframe with columns ['title','author','description']
    - cosine_sim: precomputed cosine similarity matrix
    - indices: Series mapping lowercase title -> df index (may contain duplicates)
    """
    title_key = title.strip().lower()

    if title_key not in indices.index:
        return f"{title.strip()} is not in our database"

    # locate index (if multiple entries for the same title, take the first)
    idx = indices.loc[title_key]
    if isinstance(idx, pd.Series) or isinstance(idx, pd.Index):
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    book_indices = [i[0] for i in sim_scores]

    return df.loc[book_indices, ['title', 'author']]


@st.cache_data
def load_data(path: str = 'books.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df['title'] = df['title'].astype(str).str.strip()
    df['description'] = df.get('description', '').fillna('')
    return df


@st.cache_resource
def build_similarity(df: pd.DataFrame):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['title'].str.lower())
    return cosine_sim, indices


def main():
    df = load_data('books.csv')
    cosine_sim, indices = build_similarity(df)

    st.title('Book Recommender System')
    st.write('Enter a book title to get recommendations')
    selected_book = st.text_input('Title')

    if selected_book:
        results = get_recommendation(selected_book, df, cosine_sim, indices)
        if isinstance(results, pd.DataFrame):
            st.write("**Recommendations:**")
            st.dataframe(results)
        else:
            st.warning(results)


if __name__ == '__main__':
    main()
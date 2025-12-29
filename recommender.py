import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("data/movies.csv", usecols=["title","genre"])

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df['genre'])


similarity = cosine_similarity(genre_matrix)

def recommend(movie_title, top_k=3):
    if movie_title not in df['title'].values:
        return "Movie not found"

    idx = df[df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:top_k+1]:
        recommendations.append(df.iloc[i[0]]['title'])

    return recommendations

# Test
print(recommend("Inception"))

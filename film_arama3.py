import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# İzlenmiş filmler listesi
watched_titles = [
    "Inception", "Interstellar", "The Dark Knight", "Avatar", "The Avengers",
    "Deadpool", "Avengers: Infinity War", "Fight Club", "Guardians of the Galaxy",
    "Pulp Fiction", "John Wick", "World War Z", "X-Men: Days of Future Past"
]

# Veriyi yükle
df = pd.read_csv('C:/Users/aygun/Desktop/proje/kaggle/input/movie_dataset.csv')

# Verinin %20'sini al
df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

# Eksik değerleri doldur ve özellikleri birleştir
df['combined_features'] = df['genres'].fillna('') + ' ' + df['keywords'].fillna('') + ' ' + \
                          df['production_companies'].fillna('') + ' ' + df['overview'].fillna('') + ' ' + \
                          df['cast'].fillna('') + ' ' + df['director'].fillna('')

# İzlenmiş filmleri ve izlenmemiş filmleri ayır
watched_movies = df[df['title'].isin(watched_titles)]
unwatched_movies = df[~df['title'].isin(watched_titles)]

# TF-IDF vektörizasyonu
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Graf oluşturma
G = nx.Graph()

# İzlenmiş filmlerden düğümler oluştur
for index, row in watched_movies.iterrows():
    G.add_node(index, title=row['title'], watched=True)

# İzlenmemiş filmleri düğümlere ekle
for index, row in unwatched_movies.iterrows():
    G.add_node(index, title=row['title'], watched=False)

# Tüm filmler için TF-IDF benzerlik matrisini kullanarak kenar oluştur (eşik değeri: 0.1)
threshold = 0.1
cosine_sim = (tfidf_matrix * tfidf_matrix.T).toarray()

for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > threshold:
            G.add_edge(i, j, weight=cosine_sim[i, j])

# Merkeziyet puanlarını hesapla
centrality = nx.degree_centrality(G)

# İzlenmemiş filmler için en yüksek merkezilik puanına sahip 5 filmi al
unwatched_indices = unwatched_movies.index
recommendations = sorted(
    [(idx, centrality[idx]) for idx in unwatched_indices],
    key=lambda x: x[1],
    reverse=True
)[:5]

# En yüksek 5 film önerisini listele
print("En Yüksek Merkezilik Değerlerine Sahip 5 İzlenmemiş Film:")
for idx, score in recommendations:
    print(f"{df.loc[idx, 'title']}: Merkezilik Skoru {score:.4f}")

# Grafiği görselleştir
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
colors = ['green' if G.nodes[node]['watched'] else 'blue' for node in G.nodes]
sizes = [300 if G.nodes[node]['watched'] else 100 for node in G.nodes]

# Düğümleri ve kenarları çiz
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes)
nx.draw_networkx_edges(G, pos, alpha=0.3)

# İzlenmiş filmler için etiket ekle
labels = {idx: G.nodes[idx]['title'] for idx in watched_movies.index}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="red")

plt.title("Film Grafı (Yeşil: İzlenmiş, Mavi: İzlenmemiş)")
plt.show()

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# izlenmiş filmleri kendimiz belirlioruz oluşturuyoruz
# İzlenmiş filmler listesi
watched_titles = [
    "Inception", "Interstellar", "The Dark Knight", "Avatar", "The Avengers",
    "Avengers: Age of Ultron", "Guardians of the Galaxy", "World War Z",
    "X-Men: Days of Future Past"
]

# Veriyi yükle ve işleme
df = pd.read_csv('C:/Users/aygun/Desktop/proje/kaggle/input/movie_dataset.csv')
df['combined_features'] = (
        df['genres'].fillna('') + ' ' +
        df['keywords'].fillna('') + ' ' +
        df['production_companies'].fillna('') + ' ' +
        df['overview'].fillna('') + ' ' +
        df['cast'].fillna('') + ' ' +
        df['director'].fillna('')
)

# İzlenmiş ve izlenmemiş filmleri ayır
watched_movies = df[df['title'].isin(watched_titles)]
unwatched_movies = df[~df['title'].isin(watched_titles)]


# Özelleştirilmiş benzerlik hesaplama (her eşleşme 1 puan)
def calculate_custom_similarity(df):
    n = len(df)
    similarity_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                genres_i = df.loc[i, 'genres'] if isinstance(df.loc[i, 'genres'], str) else ''
                genres_j = df.loc[j, 'genres'] if isinstance(df.loc[j, 'genres'], str) else ''
                cast_i = df.loc[i, 'cast'] if isinstance(df.loc[i, 'cast'], str) else ''
                cast_j = df.loc[j, 'cast'] if isinstance(df.loc[j, 'cast'], str) else ''

                genre_similarity = len(set(genres_i.split()) & set(genres_j.split()))
                cast_similarity = len(set(cast_i.split()) & set(cast_j.split()))

                similarity_matrix[i][j] = genre_similarity + cast_similarity

    return similarity_matrix


# Yeni benzerlik matrisini hesapla
custom_similarity = calculate_custom_similarity(df)


# Grafik oluşturma
def create_base_graph_with_custom_similarity(watched_movies, watched_indices, custom_similarity, threshold=1):
    graph = nx.Graph()
    for idx in watched_indices:
        graph.add_node(idx, title=watched_movies.loc[idx, 'title'], watched=True)
    for i in watched_indices:
        for j in watched_indices:
            if i != j and custom_similarity[i][j] >= threshold:
                graph.add_edge(i, j, weight=custom_similarity[i][j])
    return graph


# Baz grafiği oluştur
base_graph_custom = create_base_graph_with_custom_similarity(watched_movies, watched_movies.index, custom_similarity)


# Ağırlıklı merkeziyet hesaplama fonksiyonu
def weighted_centrality(graph):
    centrality = {}
    for node in graph.nodes:
        centrality[node] = 0
        for neighbor in graph.neighbors(node):
            weight = graph.edges[node, neighbor]['weight']  # Kenar ağırlığı
            neighbor_degree = graph.degree[neighbor]  # Komşunun derece bilgisi
            centrality[node] += weight / neighbor_degree  # Formüle göre katkı
    return centrality


# İzlenmemiş filmler için ağırlıklı merkeziyet hesaplama
def calculate_unwatched_scores(unwatched_movies, base_graph, watched_indices, custom_similarity):
    scores = {}
    for idx, row in unwatched_movies.iterrows():
        subgraph = base_graph.copy()
        subgraph.add_node(idx, title=row['title'], watched=False)
        for watched_idx in watched_indices:
            if custom_similarity[idx][watched_idx] >= 1:
                subgraph.add_edge(idx, watched_idx, weight=custom_similarity[idx][watched_idx])
        # Merkeziyetleri hesapla
        centrality = weighted_centrality(subgraph)
        # Merkeziyet toplamını kullan
        scores[idx] = centrality[idx] if idx in centrality else 0
    return scores


unwatched_scores_custom = calculate_unwatched_scores(unwatched_movies, base_graph_custom, watched_movies.index,
                                                     custom_similarity)

# Merkeziyet değerlerine göre sıralama
sorted_scores_custom = sorted(unwatched_scores_custom.items(), key=lambda x: x[1], reverse=True)
top_5_custom = sorted_scores_custom[:5]
middle_scores_custom = sorted_scores_custom[len(sorted_scores_custom) // 3:len(sorted_scores_custom) // 3 + 2]
low_2_custom = sorted_scores_custom[-2:]

# Seçilen tüm düğümleri birleştir
selected_nodes = top_5_custom + middle_scores_custom + low_2_custom


# Görselleştirme işlevi (daha geniş yerleşim ve sabit kenar kalınlıkları için düzenlendi)
def draw_graph(graph, selected_node, watched_indices, custom_similarity, idx, unwatched_movies):
    subgraph = graph.copy()
    subgraph.add_node(selected_node, title=unwatched_movies.loc[selected_node, 'title'], watched=False)
    for watched_idx in watched_indices:
        if custom_similarity[selected_node][watched_idx] >= 1:
            subgraph.add_edge(selected_node, watched_idx, weight=custom_similarity[selected_node][watched_idx])

    # Düğüm renkleri ve boyutları
    colors = ['blue' if not subgraph.nodes[n].get('watched', True) else 'green' for n in subgraph.nodes]
    sizes = [300 if not subgraph.nodes[n].get('watched', True) else 500 for n in subgraph.nodes]

    # Yerleşim düzeni
    pos = nx.spring_layout(subgraph, seed=42, k=3.0, iterations=100)

    # Kenar kalınlıkları sabit
    edge_weights = [1.5 for _ in subgraph.edges]

    # Grafik çizimi
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(subgraph, pos, node_color=colors, node_size=sizes)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=edge_weights)
    nx.draw_networkx_labels(subgraph, pos, labels={n: subgraph.nodes[n]['title'] for n in subgraph.nodes}, font_size=8)

    # Kenar etiketleri
    edge_labels = {(u, v): f"{subgraph.edges[u, v]['weight']:.2f}" for u, v in subgraph.edges}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_color='red', font_size=8)

    plt.title(f"Graf {idx + 1}: {subgraph.nodes[selected_node]['title']}")
    plt.show()


# Seçilen düğümleri görselleştir
for idx, (movie_idx, _) in enumerate(selected_nodes):
    print(
        f"\nGraf {idx + 1}: {unwatched_movies.loc[movie_idx, 'title']} - Merkeziyet: {unwatched_scores_custom[movie_idx]:.4f}")
    draw_graph(base_graph_custom, movie_idx, watched_movies.index, custom_similarity, idx, unwatched_movies)

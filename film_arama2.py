import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import messagebox

# Veriyi yükle
df = pd.read_csv('C:/Users/aygun/Desktop/proje/kaggle/input/movie_dataset.csv')

# Eksik veriyi kontrol et ve NaN değerleri boş string ile doldur
df['combined_features'] = df['genres'].fillna('') + ' ' + df['keywords'].fillna('') + ' ' + \
                          df['production_companies'].fillna('') + ' ' + df['overview'].fillna('') + ' ' + \
                          df['cast'].fillna('') + ' ' + df['director'].fillna('')

# TF-IDF vektörizasyonu
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Graf oluşturma
G = nx.Graph()

# Düğümleri ekle
for index, row in df.iterrows():
    G.add_node(index, title=row['title'])

# Kenarları ekle (eşik değeri: 0.2)
threshold = 0.2
cosine_sim = (tfidf_matrix * tfidf_matrix.T).toarray()
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > threshold:
            G.add_edge(i, j, weight=cosine_sim[i, j])


# Benzer film önerisi fonksiyonu
def get_recommendations(title):
    try:
        # Film başlığını küçük harfe çevir ve veri çerçevesinde arama yap
        title = title.lower()
        idx = df[df['title'].str.lower() == title].index[0]

        # Komşu düğümleri al (benzer filmler)
        neighbors = list(G.neighbors(idx))

        # Komşuların merkezilik değerlerini hesapla
        centrality = nx.degree_centrality(G)
        neighbors_sorted = sorted(neighbors, key=lambda x: centrality[x], reverse=True)[:5]

        # Önerilen filmleri döndür
        recommended_titles = [df['title'].iloc[i] for i in neighbors_sorted]
        centrality_scores = [centrality[i] for i in neighbors_sorted]

        return recommended_titles, centrality_scores

    except IndexError:
        return None, None


# Tkinter GUI kodu
class MovieRecommendationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Film Öneri Sistemi")

        # Başlık etiketini oluştur
        self.label = tk.Label(root, text="Film Adı Girin:", font=("Arial", 14))
        self.label.pack(pady=10)

        # Film adı giriş kutusunu oluştur
        self.entry = tk.Entry(root, font=("Arial", 14), width=30)
        self.entry.pack(pady=10)

        # Önerileri göster butonu
        self.button = tk.Button(root, text="Önerileri Göster", font=("Arial", 14), command=self.show_recommendations)
        self.button.pack(pady=10)

        # Sonuçları görüntülemek için liste kutusu
        self.results_listbox = tk.Listbox(root, font=("Arial", 12), width=70, height=10)
        self.results_listbox.pack(pady=10)

    def show_recommendations(self):
        # Giriş film adını al
        movie_title = self.entry.get().strip()

        if not movie_title:
            messagebox.showwarning("Uyarı", "Lütfen bir film adı girin!")
            return

        # Benzer filmleri al
        recommendations, centrality_scores = get_recommendations(movie_title)

        # Listeyi temizle
        self.results_listbox.delete(0, tk.END)

        if recommendations is None:
            messagebox.showwarning("Film Bulunamadı", f"{movie_title} adıyla bir film bulunamadı.")
        else:
            # Benzer filmleri ve puanlarını listele
            for movie, score in zip(recommendations, centrality_scores):
                result_text = f"{movie} "
                self.results_listbox.insert(tk.END, result_text)


# Tkinter uygulamasını başlat
if __name__ == '__main__':
    root = tk.Tk()
    app = MovieRecommendationApp(root)
    root.mainloop()

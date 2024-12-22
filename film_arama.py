import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import tkinter as tk
from tkinter import messagebox, Scrollbar, Listbox

# Veriyi Yükleme
movies_file = 'C:/Users/aygun/Desktop/proje/kaggle/input/movielens-1m-dataset/movies.dat'
ratings_file = 'C:/Users/aygun/Desktop/proje/kaggle/input/movielens-1m-dataset/ratings.dat'

# Movies ve Ratings dosyalarını yükle
movies_df = pd.read_csv(movies_file, sep="::", header=None, names=["film_id", "baslik", "turler"], engine='python',
                        encoding='ISO-8859-1')
ratings_df = pd.read_csv(ratings_file, sep="::", header=None,
                         names=["kullanici_id", "film_id", "puan", "zaman_damgasi"], engine='python',
                         encoding='ISO-8859-1')

# Kullanıcı-Film Matrisi Oluşturma
kullanici_film_matrisi = ratings_df.pivot(index='kullanici_id', columns='film_id', values='puan').fillna(0)


def recommend_movies(film_adi):
    # Film ID'sini Bulma
    film_kayitlari = movies_df[movies_df['baslik'].str.contains(film_adi, case=False, na=False, regex=False)]

    if film_kayitlari.empty:
        messagebox.showerror("Film Bulunamadı", "Film bulunamadı. Lütfen geçerli bir film adı girin.")
        return []

    selected_film_id = film_kayitlari.iloc[0]['film_id']
    secilen_turler = film_kayitlari.iloc[0]['turler']

    # 1. Cosine Similarity (Kosinüs Benzerliği) ile Film Önerisi
    film_kosinus_benzerlik = cosine_similarity(kullanici_film_matrisi.T)
    benzerlik_skorlari = list(enumerate(film_kosinus_benzerlik[selected_film_id - 1]))
    benzer_filmler_cosine = sorted(benzerlik_skorlari, key=lambda x: x[1], reverse=True)[1:6]

    # 2. Collaborative Filtering (User-based) ile Film Önerisi
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
    model.fit(kullanici_film_matrisi.values.T)
    distances, indices = model.kneighbors(kullanici_film_matrisi.values.T[selected_film_id - 1].reshape(1, -1),
                                          n_neighbors=6)
    benzer_filmler_collab = [(i + 1, distances[0][idx]) for idx, i in enumerate(indices[0]) if i != selected_film_id]

    # 3. Content-based Filtering (Tür Benzerliği) ile Film Önerisi
    tur_benzerlikleri = []
    for idx, row in movies_df.iterrows():
        ortak_tur = set(secilen_turler.split('|')).intersection(set(row['turler'].split('|')))
        tur_benzerlikleri.append((row['film_id'], len(ortak_tur), row['baslik'], row['turler']))
    benzer_filmler_content = sorted(tur_benzerlikleri, key=lambda x: x[1], reverse=True)[:6]

    # Öneriler
    oneriler = []
    seen_movies = set()

    # Kosinüs Benzerliği Sonuçları
    for film in benzer_filmler_cosine:
        recommended_film_id = film[0] + 1
        if recommended_film_id != selected_film_id and recommended_film_id not in seen_movies:
            film_baslik = movies_df[movies_df['film_id'] == recommended_film_id].iloc[0]['baslik']
            aciklama = "Kosinüs benzerliğine göre seçildi."
            oneriler.append((film_baslik, aciklama))
            seen_movies.add(recommended_film_id)

    # Kullanıcı Benzerliğine Göre Sonuçlar
    for film in benzer_filmler_collab:
        recommended_film_id = film[0]
        if recommended_film_id != selected_film_id and recommended_film_id not in seen_movies:
            film_baslik = movies_df[movies_df['film_id'] == recommended_film_id].iloc[0]['baslik']
            aciklama = "Kullanıcı benzerliğine göre seçildi. "
            oneriler.append((film_baslik, aciklama))
            seen_movies.add(recommended_film_id)

    # Tür Benzerliği Sonuçları
    for film in benzer_filmler_content:
        recommended_film_id = film[0]
        if recommended_film_id != selected_film_id and recommended_film_id not in seen_movies:
            film_baslik = film[2]
            aciklama = "Tür benzerliğine göre seçildi. "
            oneriler.append((film_baslik, aciklama))
            seen_movies.add(recommended_film_id)

    return oneriler


def display_recommendations():
    film_adi = film_entry.get()
    recommendations = recommend_movies(film_adi)

    if recommendations:
        recommendation_listbox.delete(0, tk.END)
        for oneri in recommendations:
            recommendation_listbox.insert(tk.END, f"{oneri[0]}: {oneri[1]}")


# GUI Başlatma
root = tk.Tk()
root.title("Film Öneri Sistemi")

# Film adı girişi
film_label = tk.Label(root, text="Film Adı:")
film_label.pack(pady=5)
film_entry = tk.Entry(root, width=50)
film_entry.pack(pady=5)

# Öneri Butonu
recommend_button = tk.Button(root, text="Film Önerileri Al", command=display_recommendations)
recommend_button.pack(pady=10)

# Öneri Listesi
recommendation_listbox = Listbox(root, width=100, height=10)
recommendation_listbox.pack(pady=10)

# Scrollbar
scrollbar = Scrollbar(root, orient="vertical", command=recommendation_listbox.yview)
scrollbar.pack(side="right", fill="y")
recommendation_listbox.config(yscrollcommand=scrollbar.set)

# GUI'yı çalıştır
root.mainloop()
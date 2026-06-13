
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# =====================================
# LOAD DATA
# =====================================
df = pd.read_csv("spotify.csv")

# Remove duplicate songs
df = df.drop_duplicates(
    subset=["track_name", "track_artist"]
).reset_index(drop=True)

# =====================================
# FEATURES
# =====================================
features = [
    "danceability",
    "energy",
    "loudness",
    "tempo",
    "valence",
    "acousticness",
    "instrumentalness",
    "track_popularity"
]

df[features] = df[features].fillna(
    df[features].median()
)

# =====================================
# CORRELATION MATRIX
# =====================================
plt.figure(figsize=(10,8))
sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm")
plt.title("Spotify Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

# =====================================
# FEATURE SCALING
# =====================================
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# =====================================
# K-MEANS CLUSTERING
# =====================================
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X)

score = silhouette_score(X, df["Cluster"])

print(f"Dataset Shape: {df.shape}")
print(f"Silhouette Score: {score:.4f}")

# =====================================
# PCA VISUALIZATION
# =====================================
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)

plt.figure(figsize=(10,6))
plt.scatter(
    pca_data[:,0],
    pca_data[:,1],
    c=df["Cluster"]
)
plt.title("Spotify Song Clusters")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.savefig("spotify_clusters.png")
plt.close()

# =====================================
# CLUSTER SUMMARY
# =====================================
df.groupby("Cluster")[features].mean().to_csv(
    "cluster_summary.csv"
)

# =====================================
# GENRE + CLUSTER + SIMILARITY
# =====================================
print("\n===== GENRE-AWARE SPOTIFY RECOMMENDER =====")

song_name = input("Enter a song name: ").strip()

matched = df[
    df["track_name"].str.lower() == song_name.lower()
]

if len(matched) == 0:

    suggestions = df[
        df["track_name"].str.lower().str.contains(
            song_name.lower(),
            na=False
        )
    ][["track_name","track_artist"]].drop_duplicates()

    if len(suggestions) > 0:
        print("\nSong not found exactly.")
        print("\nDid you mean:\n")
        print(suggestions.head(10).to_string(index=False))
    else:
        print("\nSong not found in dataset.")

else:

    song_row = matched.iloc[0]

    genre = song_row["playlist_genre"]
    cluster = song_row["Cluster"]

    print(f"\nDetected Genre : {genre}")
    print(f"Detected Cluster : {cluster}")

    candidate_df = df[
        (df["playlist_genre"] == genre) &
        (df["Cluster"] == cluster)
    ].copy()

    candidate_indices = candidate_df.index.tolist()

    candidate_X = scaler.transform(
        candidate_df[features]
    )

    nn = NearestNeighbors(
        n_neighbors=min(100, len(candidate_df)),
        metric="cosine"
    )

    nn.fit(candidate_X)

    local_idx = candidate_df.index.get_loc(song_row.name)

    distances, indices = nn.kneighbors(
        [candidate_X[local_idx]]
    )

    print("\nTop 10 Recommended Songs:\n")

    shown = set()
    count = 0

    for pos in indices[0]:

        real_idx = candidate_indices[pos]

        song = df.loc[real_idx, "track_name"]
        artist = df.loc[real_idx, "track_artist"]

        key = (song, artist)

        if key == (
            song_row["track_name"],
            song_row["track_artist"]
        ):
            continue

        if key in shown:
            continue

        shown.add(key)

        count += 1

        print(
            f"{count}. {song} - {artist}"
        )

        if count >= 10:
            break

df.to_csv(
    "spotify_clustered_output.csv",
    index=False
)

print("\nProject Completed Successfully")

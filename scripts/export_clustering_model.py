#!/usr/bin/env python
"""
Model Export Script — Spotify Audio Profile Clustering (Clustering Track)
=========================================================================
Replicates the preprocessing, K-Means clustering, Hierarchical evaluation,
and PCA projection from notebooks/py_files/clustering.py.

Saves:
    models/best_clustering_pipeline.joblib  -- scaler, kmeans, pca, personas, centroids, exemplars
    models/clustering_metadata.json         -- metrics, schema, persona profiles
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# Prevent Windows joblib core detection warning
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "spotify-tracks-dataset.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_clustering_pipeline.joblib"
META_PATH = MODEL_DIR / "clustering_metadata.json"

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

PERSONA_NAMES = {
    0: "Calm & Meditative (Ambient / Sleep / New-Age)",
    1: "Acoustic, Romance & Ballads (Soft Vocals & Strings)",
    2: "High-Octane Rock & Heavy Beats (Metal / Industrial)",
    3: "Live Performance, Spoken Word & Comedy (Crowd Atmosphere)",
    4: "Dance, Pop & Latin Fiesta (High Energy & Valence)",
}

PERSONA_DESCRIPTIONS = {
    0: "Low energy, high acousticness, quiet dynamics. Ideal for relaxation, meditation, focus, or deep sleep.",
    1: "Moderate tempo, prominent acoustic instruments, intimate vocal presence. Classic ballads, singer-songwriter, and romance.",
    2: "Loud, high energy, aggressive distortion and heavy percussion. Hard rock, metal, punk, and industrial.",
    3: "High speechiness and liveness, crowd ambiance, spoken-word or live stage delivery. Stand-up comedy, poetry, and live concerts.",
    4: "High danceability, high energy, uplifting valence, punchy tempo. Mainstream pop, EDM, reggaeton, and club anthems.",
}


def main():
    print("=" * 70)
    print("MODEL EXPORT: Spotify Audio Profile Clustering -- K-Means (k=5)")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    raw_df = pd.read_csv(DATA_PATH)
    drop_unnamed = [c for c in raw_df.columns if "Unnamed" in c]
    if drop_unnamed:
        raw_df = raw_df.drop(columns=drop_unnamed)

    clean_df = raw_df.dropna(subset=["track_name", "artists", "album_name"]).copy()
    clean_df = clean_df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    print(f"Clean unique tracks: {len(clean_df):,}")

    SAMPLE_SIZE = 5000
    sample_df = clean_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Sample drawn: {len(sample_df):,} tracks across {sample_df['track_genre'].nunique()} genres.")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_df[AUDIO_FEATURES])

    # K-Means (k=5)
    OPTIMAL_K = 5
    kmeans = KMeans(
        n_clusters=OPTIMAL_K,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    kmeans_labels = kmeans.fit_predict(X_scaled)
    sample_df["kmeans_cluster"] = kmeans_labels

    km_silhouette = float(silhouette_score(X_scaled, kmeans_labels))
    km_db = float(davies_bouldin_score(X_scaled, kmeans_labels))
    km_ch = float(calinski_harabasz_score(X_scaled, kmeans_labels))

    print(f"K-Means (k=5): Silhouette={km_silhouette:.4f}, DB={km_db:.4f}, CH={km_ch:.1f}")

    # Hierarchical Clustering (Ward)
    hierarchical = AgglomerativeClustering(
        n_clusters=OPTIMAL_K,
        metric="euclidean",
        linkage="ward",
    )
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    h_silhouette = float(silhouette_score(X_scaled, hierarchical_labels))
    h_db = float(davies_bouldin_score(X_scaled, hierarchical_labels))
    h_ch = float(calinski_harabasz_score(X_scaled, hierarchical_labels))

    ari = float(adjusted_rand_score(kmeans_labels, hierarchical_labels))
    nmi = float(normalized_mutual_info_score(kmeans_labels, hierarchical_labels))
    print(f"Hierarchical:  Silhouette={h_silhouette:.4f}, DB={h_db:.4f}, CH={h_ch:.1f}")
    print(f"Agreement:     ARI={ari:.4f}, NMI={nmi:.4f}")

    # PCA 2D
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_scaled)
    var_pc1 = float(pca.explained_variance_ratio_[0] * 100)
    var_pc2 = float(pca.explained_variance_ratio_[1] * 100)
    print(f"PCA Fitted: PC1={var_pc1:.2f}%, PC2={var_pc2:.2f}% (Total={var_pc1 + var_pc2:.2f}%)")

    # Cluster Centroids in Original Units
    centroids_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    centroids_dict = {
        str(i): {feat: round(float(centroids_unscaled[i][j]), 3) for j, feat in enumerate(AUDIO_FEATURES)}
        for i in range(OPTIMAL_K)
    }

    # Extract sample song exemplars per cluster
    exemplars = {}
    for c in range(OPTIMAL_K):
        sub = sample_df[sample_df["kmeans_cluster"] == c]
        top_songs = []
        for _, r in sub.head(4).iterrows():
            top_songs.append({
                "track_name": str(r["track_name"]),
                "artists": str(r["artists"]),
                "album_name": str(r["album_name"]),
                "genre": str(r["track_genre"]),
            })
        exemplars[str(c)] = top_songs

    # Save artifact bundle
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "features": AUDIO_FEATURES,
        "persona_names": PERSONA_NAMES,
        "persona_descriptions": PERSONA_DESCRIPTIONS,
        "centroids_unscaled": centroids_dict,
        "exemplars": exemplars,
        "optimal_k": OPTIMAL_K,
    }

    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"\nClustering pipeline bundle saved --> {MODEL_PATH}")
    print(f"  File size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")

    import sklearn

    metadata = {
        "track": "Clustering",
        "dataset_name": "Spotify Tracks Audio Features (spotify-tracks-dataset.csv)",
        "optimal_k": OPTIMAL_K,
        "features": AUDIO_FEATURES,
        "persona_names": PERSONA_NAMES,
        "persona_descriptions": PERSONA_DESCRIPTIONS,
        "evaluation_metrics": {
            "kmeans": {
                "silhouette_score": round(km_silhouette, 4),
                "davies_bouldin_index": round(km_db, 4),
                "calinski_harabasz_index": round(km_ch, 1),
            },
            "hierarchical": {
                "silhouette_score": round(h_silhouette, 4),
                "davies_bouldin_index": round(h_db, 4),
                "calinski_harabasz_index": round(h_ch, 1),
            },
            "agreement": {
                "adjusted_rand_score": round(ari, 4),
                "normalized_mutual_info": round(nmi, 4),
            },
            "pca_variance_explained": {
                "pc1_pct": round(var_pc1, 2),
                "pc2_pct": round(var_pc2, 2),
                "total_pct": round(var_pc1 + var_pc2, 2),
            },
        },
        "centroids": centroids_dict,
        "exemplars": exemplars,
        "versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved --> {META_PATH}")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE -- Clustering model ready for Streamlit deployment")
    print("=" * 70)


if __name__ == "__main__":
    main()

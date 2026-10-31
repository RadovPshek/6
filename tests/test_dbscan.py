import numpy as np
import pytest
from sklearn.datasets import make_circles, make_moons, load_iris
from sklearn.metrics import (
    silhouette_score as sk_silhouette_score,
    davies_bouldin_score as sk_davies_bouldin_score,
    adjusted_rand_score,
    rand_score,
)

from src.dbscan import DBSCANClustering


def test_dbscan_not_constant():
    """DBSCAN не должен возвращать одинаковую метку для всех объектов."""
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    model = DBSCANClustering(eps=0.3, min_samples=7)
    labels = model.predict(X)

    assert labels.shape[0] == X.shape[0]
    assert len(set(labels)) > 1


def test_dbscan_more_than_one_cluster():
    """DBSCAN должен обнаружить более одного кластера."""
    X, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

    model = DBSCANClustering(eps=0.2, min_samples=5)
    labels = model.predict(X)

    non_noise = labels[labels >= 0]
    assert len(set(non_noise)) > 1


def test_dbscan_cluster_balance():
    """Кластеры не должны быть слишком несбалансированными."""
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    model = DBSCANClustering(eps=0.3, min_samples=7)
    labels = model.predict(X)

    non_noise = labels[labels >= 0]
    assert non_noise.size != 0, "DBSCAN вернул только шумовые метки"
    counts = np.bincount(non_noise)
    assert counts.min() > 0.05 * counts.max()


def test_dbscan_quality_synthetic():
    """Качество DBSCAN: silhouette > 0.1, DBI < 2.5 на синтетических данных."""
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=22)

    model = DBSCANClustering(eps=0.3, min_samples=7)
    labels = model.predict(X)

    non_noise = labels[labels >= 0]
    assert len(set(non_noise)) > 1, "DBSCAN нашёл < 2 кластеров"

    sil = sk_silhouette_score(X[labels >= 0], non_noise)
    db = sk_davies_bouldin_score(X[labels >= 0], non_noise)

    assert sil > 0.2
    assert db < 2.5


def test_dbscan_quality_iris():
    """Качество DBSCAN на Iris: Rand index ≥ 0.3, ARI ≥ 0.1."""
    data = load_iris()
    X = data.data
    y_true = data.target

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    X, y_true = X[idx], y_true[idx]

    model = DBSCANClustering(eps=0.5, min_samples=5)
    labels = model.predict(X)

    non_noise_idx = labels >= 0
    assert non_noise_idx.sum() > 1, "Слишком мало неshумовых точек"

    ri = rand_score(y_true[non_noise_idx], labels[non_noise_idx])
    ari = adjusted_rand_score(y_true[non_noise_idx], labels[non_noise_idx])

    assert ri >= 0.3
    assert ari >= 0.1


def test_dbscan_predict_shape():
    """predict возвращает массив правильной длины."""
    X, _ = make_moons(n_samples=150, noise=0.05, random_state=42)

    model = DBSCANClustering(eps=0.3, min_samples=5)
    labels = model.predict(X)

    assert labels.shape == (150,)


def test_dbscan_noise_label():
    """Шумовые точки должны быть помечены меткой -1."""
    rng = np.random.RandomState(42)
    cluster = rng.randn(50, 2) * 0.1
    outliers = rng.randn(10, 2) * 10 + 50
    X = np.vstack([cluster, outliers])

    model = DBSCANClustering(eps=0.5, min_samples=5)
    labels = model.predict(X)

    assert -1 in labels

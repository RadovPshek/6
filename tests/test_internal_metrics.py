import numpy as np
import pytest
from sklearn.metrics import (
    silhouette_score as sk_silhouette_score,
    davies_bouldin_score as sk_davies_bouldin_score,
)
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from permetrics import ClusteringMetric

from src.internal_metrics import (
    r_squared_score,
    silhouette_score,
    dunn_index,
    davies_bouldin_index,
    dbcv_score,
)


def _make_well_separated():
    """Три хорошо разделённых кластера."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 2) * 0.5 + np.array([0, 0])
    c2 = rng.randn(30, 2) * 0.5 + np.array([10, 10])
    c3 = rng.randn(30, 2) * 0.5 + np.array([10, 0])
    X = np.vstack([c1, c2, c3])
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
    return X, labels


def _make_overlapping():
    """Два перекрывающихся кластера."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 2) * 2.0 + np.array([0, 0])
    c2 = rng.randn(30, 2) * 2.0 + np.array([1, 1])
    X = np.vstack([c1, c2])
    labels = np.array([0] * 30 + [1] * 30)
    return X, labels


# ========== R-squared ==========

def test_r_squared_well_separated():
    """R² для хорошо разделённых кластеров должен быть близок к 1."""
    X, labels = _make_well_separated()

    r2 = r_squared_score(X, labels)

    assert r2 > 0.85


def test_r_squared_single_cluster():
    """R² для одного кластера = 0 (SS_within == SS_total)."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 2)
    labels = np.zeros(50, dtype=int)

    r2 = r_squared_score(X, labels)

    assert abs(r2) < 1e-7


def test_r_squared_perfect():
    """R² = 1, когда каждая точка — отдельный кластер."""
    X = np.array([[0, 0], [5, 5], [10, 0]], dtype=float)
    labels = np.array([0, 1, 2])

    r2 = r_squared_score(X, labels)

    assert abs(r2 - 1.0) < 1e-7


def test_r_squared_range():
    """R² для случайной кластеризации в допустимом диапазоне [0, 1]."""
    X, labels = _make_well_separated()

    r2 = r_squared_score(X, labels)

    assert 0.0 <= r2 <= 1.0


# ========== Silhouette ==========

def test_silhouette_well_separated():
    """Силуэт для хорошо разделённых кластеров > 0.7."""
    X, labels = _make_well_separated()

    s = silhouette_score(X, labels)

    assert s > 0.7


def test_silhouette_matches_sklearn():
    """Силуэт должен совпадать с sklearn."""
    X, labels = _make_well_separated()

    our = silhouette_score(X, labels)
    expected = sk_silhouette_score(X, labels)

    assert abs(our - expected) < 1e-5


def test_silhouette_overlapping_lower():
    """Силуэт для перекрывающихся кластеров ниже, чем для разделённых."""
    X_good, labels_good = _make_well_separated()
    X_bad, labels_bad = _make_overlapping()

    s_good = silhouette_score(X_good, labels_good)
    s_bad = silhouette_score(X_bad, labels_bad)

    assert s_good > s_bad


def test_silhouette_range():
    """Силуэт в диапазоне [-1, 1]."""
    X, labels = _make_overlapping()

    s = silhouette_score(X, labels)

    assert -1.0 <= s <= 1.0


def test_silhouette_matches_sklearn_overlapping():
    """Силуэт для перекрывающихся кластеров совпадает с sklearn."""
    X, labels = _make_overlapping()

    our = silhouette_score(X, labels)
    expected = sk_silhouette_score(X, labels)

    assert abs(our - expected) < 1e-5


# ========== Dunn Index ==========

def test_dunn_well_separated():
    """Индекс Данна для хорошо разделённых кластеров > 1."""
    X, labels = _make_well_separated()

    d = dunn_index(X, labels)

    assert d > 1.0


def test_dunn_positive():
    """Индекс Данна всегда положителен."""
    X, labels = _make_overlapping()

    d = dunn_index(X, labels)

    assert d > 0


def test_dunn_overlapping_lower():
    """Индекс Данна для перекрывающихся кластеров меньше, чем для разделённых."""
    X_good, labels_good = _make_well_separated()
    X_bad, labels_bad = _make_overlapping()

    d_good = dunn_index(X_good, labels_good)
    d_bad = dunn_index(X_bad, labels_bad)

    assert d_good > d_bad


def test_dunn_known_value():
    """Индекс Данна на простом примере."""
    X = np.array([
        [0.0, 0.0], [1.0, 0.0],
        [10.0, 0.0], [11.0, 0.0],
    ])
    labels = np.array([0, 0, 1, 1])

    d = dunn_index(X, labels)

    # min inter-cluster dist = |10-1| = 9, max intra-cluster diam = 1
    assert abs(d - 9.0) < 1e-7


# ========== Davies-Bouldin Index ==========

def test_dbi_well_separated():
    """DBI для хорошо разделённых кластеров < 0.5."""
    X, labels = _make_well_separated()

    dbi = davies_bouldin_index(X, labels)

    assert dbi < 0.5


def test_dbi_matches_sklearn():
    """DBI должен совпадать с sklearn."""
    X, labels = _make_well_separated()

    our = davies_bouldin_index(X, labels)
    expected = sk_davies_bouldin_score(X, labels)

    assert abs(our - expected) < 1e-5


def test_dbi_overlapping_higher():
    """DBI для перекрывающихся кластеров выше, чем для разделённых."""
    X_good, labels_good = _make_well_separated()
    X_bad, labels_bad = _make_overlapping()

    dbi_good = davies_bouldin_index(X_good, labels_good)
    dbi_bad = davies_bouldin_index(X_bad, labels_bad)

    assert dbi_bad > dbi_good


def test_dbi_non_negative():
    """DBI всегда ≥ 0."""
    X, labels = _make_overlapping()

    dbi = davies_bouldin_index(X, labels)

    assert dbi >= 0.0


def test_dbi_matches_sklearn_overlapping():
    """DBI для перекрывающихся кластеров совпадает с sklearn."""
    X, labels = _make_overlapping()

    our = davies_bouldin_index(X, labels)
    expected = sk_davies_bouldin_score(X, labels)

    assert abs(our - expected) < 1e-5


# ========== DBCV ==========

def test_dbcv_well_separated():
    """DBCV для хорошо разделённых кластеров > 0."""
    X, labels = _make_well_separated()

    score = dbcv_score(X, labels)

    assert score > 0.0


def test_dbcv_range():
    """DBCV в диапазоне [-1, 1]."""
    X, labels = _make_well_separated()

    score = dbcv_score(X, labels)

    assert -1.0 <= score <= 1.0


def test_dbcv_overlapping_lower():
    """DBCV для перекрывающихся кластеров ниже, чем для разделённых."""
    X_good, labels_good = _make_well_separated()
    X_bad, labels_bad = _make_overlapping()

    score_good = dbcv_score(X_good, labels_good)
    score_bad = dbcv_score(X_bad, labels_bad)

    assert score_good > score_bad


def test_dbcv_range_overlapping():
    """DBCV для перекрывающихся кластеров тоже в допустимом диапазоне."""
    X, labels = _make_overlapping()

    score = dbcv_score(X, labels)

    assert -1.0 <= score <= 1.0


# ========== Сравнение с permetrics ==========

def test_r_squared_matches_permetrics():
    """R² должен совпадать с permetrics RSI."""
    X, labels = _make_well_separated()

    our = r_squared_score(X, labels)
    pm = ClusteringMetric(X=X, y_pred=labels)
    expected = pm.RSI()

    assert abs(our - expected) < 1e-5


def test_r_squared_matches_permetrics_overlapping():
    """R² для перекрывающихся кластеров совпадает с permetrics RSI."""
    X, labels = _make_overlapping()

    our = r_squared_score(X, labels)
    pm = ClusteringMetric(X=X, y_pred=labels)
    expected = pm.RSI()

    assert abs(our - expected) < 1e-5


def test_dunn_matches_permetrics():
    """Индекс Данна должен совпадать с permetrics DI (классическая версия)."""
    X, labels = _make_well_separated()

    our = dunn_index(X, labels)
    pm = ClusteringMetric(X=X, y_pred=labels)
    expected = pm.dunn_index(use_modified=False)

    assert abs(our - expected) < 1e-5


def test_dunn_matches_permetrics_overlapping():
    """Индекс Данна для перекрывающихся кластеров совпадает с permetrics DI."""
    X, labels = _make_overlapping()

    our = dunn_index(X, labels)
    pm = ClusteringMetric(X=X, y_pred=labels)
    expected = pm.dunn_index(use_modified=False)

    assert abs(our - expected) < 1e-5


def test_dbcv_ordering_matches_permetrics():
    """DBCV и permetrics DBCVI согласованы: хорошая кластеризация лучше плохой."""
    X_good, labels_good = _make_well_separated()
    X_bad, labels_bad = _make_overlapping()

    our_good = dbcv_score(X_good, labels_good)
    our_bad = dbcv_score(X_bad, labels_bad)

    pm_good = ClusteringMetric(X=X_good, y_pred=labels_good).DBCVI()
    pm_bad = ClusteringMetric(X=X_bad, y_pred=labels_bad).DBCVI()

    assert our_good > our_bad
    assert pm_good < pm_bad


# ========== Интеграционный тест: k-means на Iris ==========

def test_all_internal_metrics_kmeans_iris():
    """Проверка всех внутренних метрик на результатах k-means для Iris."""
    data = load_iris()
    X = data.data

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    r2 = r_squared_score(X, labels)
    assert 0.0 < r2 < 1.0

    sil = silhouette_score(X, labels)
    sil_expected = sk_silhouette_score(X, labels)
    assert abs(sil - sil_expected) < 1e-5

    di = dunn_index(X, labels)
    assert di > 0

    dbi = davies_bouldin_index(X, labels)
    dbi_expected = sk_davies_bouldin_score(X, labels)
    assert abs(dbi - dbi_expected) < 1e-5

    dbcv = dbcv_score(X, labels)
    assert -1.0 <= dbcv <= 1.0

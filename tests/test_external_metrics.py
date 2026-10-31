import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    precision_score as sk_precision_score,
    recall_score as sk_recall_score,
    f1_score as sk_f1_score,
    roc_curve as sk_roc_curve,
    mean_squared_error as sk_mean_squared_error,
)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.external_metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    mean_squared_error,
    cross_val_score,
)


# ========== Confusion Matrix ==========

def test_confusion_matrix_binary():
    """Матрица неточностей для бинарной классификации."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    cm = confusion_matrix(y_true, y_pred)
    expected = sk_confusion_matrix(y_true, y_pred)

    assert np.array_equal(cm, expected)


def test_confusion_matrix_multiclass():
    """Матрица неточностей для многоклассовой классификации."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 2, 1, 0, 1, 2])

    cm = confusion_matrix(y_true, y_pred)
    expected = sk_confusion_matrix(y_true, y_pred)

    assert np.array_equal(cm, expected)


def test_confusion_matrix_perfect():
    """Идеальная классификация — диагональная матрица."""
    y = np.array([0, 1, 2, 0, 1, 2])

    cm = confusion_matrix(y, y)

    assert np.array_equal(cm, np.diag([2, 2, 2]))


def test_confusion_matrix_shape():
    """Размер матрицы = (n_classes, n_classes)."""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])

    cm = confusion_matrix(y_true, y_pred)

    assert cm.shape == (4, 4)


# ========== Precision ==========

def test_precision_binary_macro():
    """Precision (macro) для бинарной классификации."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    result = precision_score(y_true, y_pred, average='macro')
    expected = sk_precision_score(y_true, y_pred, average='macro')

    assert abs(result - expected) < 1e-7


def test_precision_per_class():
    """Precision для каждого класса отдельно."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    result = precision_score(y_true, y_pred, average=None)
    expected = sk_precision_score(y_true, y_pred, average=None)

    assert np.allclose(result, expected)


def test_precision_multiclass_macro():
    """Precision (macro) для многоклассовой классификации."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])

    result = precision_score(y_true, y_pred, average='macro')
    expected = sk_precision_score(y_true, y_pred, average='macro')

    assert abs(result - expected) < 1e-7


def test_precision_perfect():
    """Precision = 1 при идеальной классификации."""
    y = np.array([0, 1, 2, 0, 1, 2])

    result = precision_score(y, y, average='macro')

    assert abs(result - 1.0) < 1e-7


# ========== Recall ==========

def test_recall_binary_macro():
    """Recall (macro) для бинарной классификации."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    result = recall_score(y_true, y_pred, average='macro')
    expected = sk_recall_score(y_true, y_pred, average='macro')

    assert abs(result - expected) < 1e-7


def test_recall_per_class():
    """Recall для каждого класса отдельно."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    result = recall_score(y_true, y_pred, average=None)
    expected = sk_recall_score(y_true, y_pred, average=None)

    assert np.allclose(result, expected)


def test_recall_multiclass_macro():
    """Recall (macro) для многоклассовой классификации."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])

    result = recall_score(y_true, y_pred, average='macro')
    expected = sk_recall_score(y_true, y_pred, average='macro')

    assert abs(result - expected) < 1e-7


def test_recall_perfect():
    """Recall = 1 при идеальной классификации."""
    y = np.array([0, 1, 2, 0, 1, 2])

    result = recall_score(y, y, average='macro')

    assert abs(result - 1.0) < 1e-7


# ========== F1 Score ==========

def test_f1_binary_macro():
    """F1 (macro) для бинарной классификации."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    result = f1_score(y_true, y_pred, average='macro')
    expected = sk_f1_score(y_true, y_pred, average='macro')

    assert abs(result - expected) < 1e-7


def test_f1_per_class():
    """F1 для каждого класса отдельно."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    result = f1_score(y_true, y_pred, average=None)
    expected = sk_f1_score(y_true, y_pred, average=None)

    assert np.allclose(result, expected)


def test_f1_multiclass_macro():
    """F1 (macro) для многоклассовой классификации."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])

    result = f1_score(y_true, y_pred, average='macro')
    expected = sk_f1_score(y_true, y_pred, average='macro')

    assert abs(result - expected) < 1e-7


def test_f1_perfect():
    """F1 = 1 при идеальной классификации."""
    y = np.array([0, 1, 2, 0, 1, 2])

    result = f1_score(y, y, average='macro')

    assert abs(result - 1.0) < 1e-7


def test_f1_consistency():
    """F1 согласована с precision и recall: F1 = 2*P*R / (P+R)."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    y_pred = np.array([0, 2, 1, 0, 2, 1, 0, 1])

    p = precision_score(y_true, y_pred, average=None)
    r = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    denom = np.asarray(p) + np.asarray(r)
    safe_denom = np.where(denom > 0, denom, 1.0)
    expected = np.where(denom > 0, 2 * np.asarray(p) * np.asarray(r) / safe_denom, 0.0)

    assert np.allclose(f1, expected)


# ========== ROC Curve ==========

def test_roc_curve_endpoints():
    """ROC-кривая начинается в (0,0) и заканчивается в (1,1)."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    assert fpr[0] == 0.0
    assert tpr[0] == 0.0
    assert fpr[-1] == 1.0
    assert tpr[-1] == 1.0


def test_roc_curve_perfect_classifier():
    """ROC-кривая идеального классификатора проходит через (0, 1)."""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    found = False
    for f, t in zip(fpr, tpr):
        if t == 1.0 and f == 0.0:
            found = True
            break
    assert found


def test_roc_curve_monotonic():
    """FPR и TPR должны быть неубывающими."""
    np.random.seed(42)
    y_true = np.array([0] * 50 + [1] * 50)
    y_scores = np.random.rand(100)

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    assert np.all(np.diff(fpr) >= -1e-10)
    assert np.all(np.diff(tpr) >= -1e-10)


def test_roc_curve_auc_matches_sklearn():
    """Площадь под ROC-кривой приблизительно совпадает с sklearn."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_scores = np.array([0.1, 0.3, 0.4, 0.7, 0.8, 0.95, 0.5, 0.6])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    sk_fpr, sk_tpr, _ = sk_roc_curve(y_true, y_scores)

    auc_ours = float(np.trapezoid(tpr, fpr))
    auc_sk = float(np.trapezoid(sk_tpr, sk_fpr))

    assert abs(auc_ours - auc_sk) < 0.05


def test_roc_curve_random_classifier():
    """AUC случайного классификатора ≈ 0.5."""
    np.random.seed(123)
    y_true = np.array([0] * 500 + [1] * 500)
    y_scores = np.random.rand(1000)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = float(np.trapezoid(tpr, fpr))

    assert 0.35 < auc < 0.65


# ========== Mean Squared Error ==========

def test_mse_perfect():
    """MSE при идеальном предсказании = 0."""
    y = np.array([1.0, 2.0, 3.0])

    assert mean_squared_error(y, y) == 0.0


def test_mse_known_value():
    """MSE на известном примере."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    result = mean_squared_error(y_true, y_pred)
    expected = sk_mean_squared_error(y_true, y_pred)

    assert abs(result - expected) < 1e-7


def test_mse_large_array():
    """MSE на большом массиве совпадает с sklearn."""
    np.random.seed(42)
    y_true = np.random.randn(1000)
    y_pred = np.random.randn(1000)

    result = mean_squared_error(y_true, y_pred)
    expected = sk_mean_squared_error(y_true, y_pred)

    assert abs(result - expected) < 1e-7


def test_mse_symmetric():
    """MSE(a, b) == MSE(b, a)."""
    y_true = np.array([1.0, 3.0, 5.0])
    y_pred = np.array([2.0, 4.0, 6.0])

    assert abs(
        mean_squared_error(y_true, y_pred) -
        mean_squared_error(y_pred, y_true)
    ) < 1e-10


# ========== Cross-Validation ==========

class _MajorityClassifier:
    """Классификатор большинства для тестирования кросс-валидации."""

    def __init__(self):
        self._majority = None

    def fit(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self._majority = classes[np.argmax(counts)]

    def predict(self, X):
        return np.full(X.shape[0], self._majority)


class _NearestNeighborClassifier:
    """1-NN классификатор для тестирования кросс-валидации."""

    def __init__(self):
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = np.array(X, dtype=float)
        self._y = np.array(y)

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = np.zeros(len(X), dtype=self._y.dtype)
        for i, xi in enumerate(X):
            dists = np.sqrt(np.sum((self._X - xi) ** 2, axis=1))
            preds[i] = self._y[np.argmin(dists)]
        return preds


def test_cross_val_score_shape():
    """cross_val_score возвращает массив длины k_folds."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([0] * 50 + [1] * 50)

    model = _MajorityClassifier()
    scores = cross_val_score(model, X, y, k_folds=5)

    assert len(scores) == 5


def test_cross_val_score_values_in_range():
    """Значения кросс-валидации в диапазоне [0, 1] при accuracy."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([0] * 50 + [1] * 50)

    model = _MajorityClassifier()
    scores = cross_val_score(model, X, y, k_folds=5)

    assert all(0.0 <= s <= 1.0 for s in scores)


def test_cross_val_score_custom_metric():
    """Кросс-валидация с пользовательской метрикой."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.array([0] * 50 + [1] * 50)

    def my_accuracy(y_true, y_pred):
        return float(np.mean(y_true == y_pred))

    model = _MajorityClassifier()
    scores = cross_val_score(model, X, y, k_folds=5, metric_fn=my_accuracy)

    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_cross_val_score_perfect_classifier():
    """1-NN на хорошо разделимых данных: все фолды дают accuracy = 1.0."""
    X = np.vstack([
        np.array([[0, i] for i in range(20)], dtype=float),
        np.array([[100, i] for i in range(20)], dtype=float),
    ])
    y = np.array([0] * 20 + [1] * 20)

    model = _NearestNeighborClassifier()
    scores = cross_val_score(model, X, y, k_folds=4)

    assert all(abs(s - 1.0) < 1e-7 for s in scores)


def test_cross_val_score_3_folds():
    """Кросс-валидация с 3 фолдами."""
    np.random.seed(42)
    X = np.random.randn(90, 3)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)

    model = _MajorityClassifier()
    scores = cross_val_score(model, X, y, k_folds=3)

    assert len(scores) == 3


# ========== Интеграционный тест: kNN на Iris ==========

def test_all_external_metrics_knn_iris():
    """Проверка всех внешних метрик на результатах kNN для Iris."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=42
    )

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm_expected = sk_confusion_matrix(y_test, y_pred)
    assert np.array_equal(cm, cm_expected)

    p = precision_score(y_test, y_pred, average='macro')
    p_expected = sk_precision_score(y_test, y_pred, average='macro')
    assert abs(p - p_expected) < 1e-7

    r = recall_score(y_test, y_pred, average='macro')
    r_expected = sk_recall_score(y_test, y_pred, average='macro')
    assert abs(r - r_expected) < 1e-7

    f1 = f1_score(y_test, y_pred, average='macro')
    f1_expected = sk_f1_score(y_test, y_pred, average='macro')
    assert abs(f1 - f1_expected) < 1e-7

    mse = mean_squared_error(y_test.astype(float), y_pred.astype(float))
    mse_expected = sk_mean_squared_error(y_test.astype(float), y_pred.astype(float))
    assert abs(mse - mse_expected) < 1e-7

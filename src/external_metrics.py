import numpy as np
from typing import Union, Optional, Tuple


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Матрица ошибок Таблица, где строки - реальные классы, столбцы - предсказанные.
    По диагонали - правильные ответы, вне диагонали - ошибки. Используется как основа для всех остальных метрик."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[class_to_idx[t], class_to_idx[p]] += 1
    return matrix

"""Точность
Из всех, кого модель назвала положительными, сколько реально положительные. 
Формула: TP / (TP + FP). 
Используют, когда важно избегать ложных срабатываний (например, спам-фильтр)."""
def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Optional[str] = 'macro'
) -> Union[float, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    denom = tp + fp
    precision = np.where(denom > 0, tp / denom, 0.0)
    if average == 'macro':
        return float(precision.mean())
    return precision

"""Полнота
Из всех реально положительных, сколько модель нашла. 
Формула: TP / (TP + FN). Используют, когда важно не пропустить ни одного случая (например, диагностика болезней)."""
def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Optional[str] = 'macro'
) -> Union[float, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    denom = tp + fn
    recall = np.where(denom > 0, tp / denom, 0.0)
    if average == 'macro':
        return float(recall.mean())
    return recall

"""F1-мера
Баланс между precision и recall. 
Формула: 2 * P * R / (P + R). 
Используют, когда оба типа ошибок одинаково важны."""
def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Optional[str] = 'macro'
) -> Union[float, np.ndarray]:
    p = precision_score(y_true, y_pred, average=None)
    r = recall_score(y_true, y_pred, average=None)
    denom = p + r
    f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
    if average == 'macro':
        return float(f1.mean())
    return f1

"""ROC-кривая
График зависимости TPR от FPR при разных порогах классификации.
 Чем кривая выше и левее - тем лучше модель. Используют для сравнения бинарных классификаторов."""
def roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sorted_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[sorted_idx]
    thresholds = y_scores[sorted_idx]

    P = y_true.sum()
    N = len(y_true) - P

    tpr_list = [0.0]
    fpr_list = [0.0]
    thresh_list = []

    tp = 0
    fp = 0
    prev_thresh = None

    for i, (label, thresh) in enumerate(zip(y_true_sorted, thresholds)):
        if thresh != prev_thresh and i > 0:
            tpr_list.append(tp / P if P > 0 else 0.0)
            fpr_list.append(fp / N if N > 0 else 0.0)
            thresh_list.append(prev_thresh)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_thresh = thresh

    tpr_list.append(tp / P if P > 0 else 0.0)
    fpr_list.append(fp / N if N > 0 else 0.0)
    thresh_list.append(prev_thresh)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    thresholds_arr = np.array(thresh_list) if thresh_list else np.array([0.0])

    order = np.argsort(fpr)
    return fpr[order], tpr[order], thresholds_arr


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Среднеквадратичная ошибка
    Среднее квадратов разностей между реальными и предсказанными значениями.
    Формула: mean((y_true - y_pred)²). Используют в задачах регрессии (предсказание числа, а не класса)."""
    diff = y_true - y_pred
    return float((diff ** 2).mean())

"""Кросс-валидация
Делит данные на k частей, k раз обучает модель и проверяет на оставшейся части. Возвращает k оценок. Используют чтобы честно оценить модель и избежать переобучения."""
def cross_val_score(
    model,
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    metric_fn=None
) -> np.ndarray:
    n = len(X)
    indices = np.arange(n)
    fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
    fold_sizes[:n % k_folds] += 1

    scores = np.zeros(k_folds)
    current = 0

    for fold, size in enumerate(fold_sizes):
        test_idx = indices[current:current + size]
        train_idx = np.concatenate([indices[:current], indices[current + size:]])

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if metric_fn is None:
            scores[fold] = float((y_pred == y_test).mean())
        else:
            scores[fold] = float(metric_fn(y_test, y_pred))

        current += size

    return scores
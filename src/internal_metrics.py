import numpy as np

""" R² (коэффициент детерминации)
Насколько хорошо кластеры объясняют разброс данных. 
Сравнивает общую дисперсию с дисперсией внутри кластеров. Чем ближе к 1 - тем лучше разделение."""
def r_squared_score(X: np.ndarray, labels: np.ndarray) -> float:
    total_var = np.var(X, axis=0).sum()
    if total_var == 0:
        return 0.0
    unique = np.unique(labels)
    var_within = sum(
        np.var(X[labels == k], axis=0).sum()
        for k in unique if (labels == k).sum() > 1
    )
    return float((total_var - var_within) / total_var)

"""Силуэтный коэффициент
Насколько точка похожа на свой кластер и непохожа на соседние. 
Для каждой точки считает (b - a) / max(a, b), где a - среднее расстояние до своих, b - до ближайшего чужого кластера. 
Диапазон от -1 до 1, выше - лучше."""
def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    n = len(X)
    if len(unique) < 2:
        return 0.0

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    s_vals = np.zeros(n)
    for i in range(n):
        own = labels[i]
        own_mask = labels == own
        own_mask[i] = False
        own_count = own_mask.sum()

        if own_count == 0:
            a = 0.0
        else:
            a = dist[i, own_mask].mean()

        b = np.inf
        for k in unique:
            if k == own:
                continue
            other_mask = labels == k
            mean_dist = dist[i, other_mask].mean()
            if mean_dist < b:
                b = mean_dist

        max_ab = max(a, b)
        s_vals[i] = (b - a) / max_ab if max_ab > 0 else 0.0

    return float(s_vals.mean())


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Индекс Данна
Отношение минимального расстояния между кластерами к максимальному диаметру кластера.
Чем выше - тем кластеры компактнее и дальше друг от друга.
    """
    unique = np.unique(labels)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    min_inter = np.inf
    for i, ci in enumerate(unique):
        for cj in unique[i + 1:]:
            mask_i = labels == ci
            mask_j = labels == cj
            d = dist[np.ix_(mask_i, mask_j)].min()
            if d < min_inter:
                min_inter = d

    max_intra = 0.0
    for c in unique:
        mask = labels == c
        if mask.sum() < 2:
            continue
        d = dist[np.ix_(mask, mask)].max()
        if d > max_intra:
            max_intra = d

    if max_intra == 0:
        return 0.0
    return float(min_inter / max_intra)


def davies_bouldin_index(X: np.ndarray, labels: np.ndarray) -> float:
    """ Индекс Дэвиса-Болдина
Среднее отношение "разброс внутри кластера / расстояние между кластерами" для каждой пары. Чем ниже - тем лучше (в отличие от остальных)."""
    unique = np.unique(labels)
    k = len(unique)
    centroids = np.array([X[labels == c].mean(axis=0) for c in unique])

    s = np.array([
        np.sqrt(((X[labels == c] - centroids[i]) ** 2).sum(axis=1)).mean()
        for i, c in enumerate(unique)
    ])

    db = 0.0
    for i in range(k):
        max_r = 0.0
        for j in range(k):
            if i == j:
                continue
            d_ij = np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
            if d_ij == 0:
                continue
            r = (s[i] + s[j]) / d_ij
            if r > max_r:
                max_r = r
        db += max_r

    return float(db / k)


def dbcv_score(X: np.ndarray, labels: np.ndarray) -> float:
    """DBCV (плотностная валидация)
Продвинутая метрика для кластеров произвольной формы (например, от DBSCAN).
Строит минимальное остовное дерево через mutual reachability distance, оценивает плотность внутри и между кластерами.
 Диапазон от -1 до 1, выше - лучше."""
    unique = np.unique(labels[labels >= 0])
    n_features = X.shape[1]
    n = len(X)

    if len(unique) < 2:
        return 0.0


    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_all = np.sqrt((diff ** 2).sum(axis=2))


    core_dist = np.zeros(n)
    for c in unique:
        mask = np.where(labels == c)[0]
        for idx in mask:
            dists_in_cluster = np.sort(dist_all[idx][mask])
            d = min(n_features, len(dists_in_cluster) - 1)
            core_dist[idx] = dists_in_cluster[d]

    def mrd(a, b):
        """ Mutual Reachability Distance
Расстояние между точками с учётом плотности.
Берёт максимум из: core_dist точки a, core_dist точки b, реального расстояния между ними.
Это защищает от ложного "сближения" точек в разреженных зонах."""
        return max(core_dist[a], core_dist[b], dist_all[a, b])


    def cluster_mst_max_edge(indices):
        """Максимальное ребро MST кластера
Строит минимальное остовное дерево (алгоритм Прима) внутри кластера через mrd-расстояния.
Возвращает вес самого тяжёлого ребра - это и есть dspc, внутренняя "рыхлость" кластера."""
        m = len(indices)
        if m < 2:
            return 0.0
        in_mst = np.zeros(m, dtype=bool)
        in_mst[0] = True
        key = np.full(m, np.inf)
        for v in range(1, m):
            key[v] = mrd(indices[0], indices[v])

        max_edge = 0.0
        for _ in range(m - 1):
            candidates = np.where(~in_mst)[0]
            u = candidates[np.argmin(key[candidates])]
            max_edge = max(max_edge, key[u])
            in_mst[u] = True
            for v in range(m):
                if not in_mst[v]:
                    w = mrd(indices[u], indices[v])
                    if w < key[v]:
                        key[v] = w

        return max_edge


    dspc = {}
    "Плотность внутри кластера"
    cluster_indices = {}
    for c in unique:
        idx = np.where(labels == c)[0]
        cluster_indices[c] = idx
        dspc[c] = cluster_mst_max_edge(idx)

    dsep = {}
    "Расстояние между парами точек"
    for i, ci in enumerate(unique):
        for cj in unique[i + 1:]:
            idx_i = cluster_indices[ci]
            idx_j = cluster_indices[cj]
            min_sep = np.inf
            for a in idx_i:
                for b in idx_j:
                    val = mrd(a, b)
                    if val < min_sep:
                        min_sep = val
            dsep[(ci, cj)] = min_sep
            dsep[(cj, ci)] = min_sep


    vc = {}
    "На сколько отделен от других кластеров"
    for ci in unique:
        min_val = np.inf
        for cj in unique:
            if ci == cj:
                continue
            sep = dsep[(ci, cj)]
            spc = dspc[ci]
            denom = max(sep, spc)
            val = (sep - spc) / denom if denom > 0 else 0.0
            if val < min_val:
                min_val = val
        vc[ci] = min_val if min_val != np.inf else 0.0

    # Взвешенное среднее по размерам кластеров
    total = sum(len(cluster_indices[c]) for c in unique)
    score = sum(vc[c] * len(cluster_indices[c]) for c in unique) / total
    return float(np.clip(score, -1.0, 1.0))
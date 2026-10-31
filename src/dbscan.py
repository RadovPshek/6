import numpy as np


class DBSCANClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Пустой вход - пустой массив меток, чтобы не падать на граничных случаях
        if X is None or X.shape[0] == 0:
            return np.zeros(0, dtype=int)

        n = X.shape[0]

        # Изначально все точки помечены как «шум» (-1).
        # В процессе обхода часть из них получит номера кластеров.
        labels = np.full(n, -1, dtype=int)

        # Матрица попарных расстояний
        # diff[i, j] = X[i] - X[j], затем берём евклидову норму по последней оси
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))

        # Для каждой точки заранее находим соседей в радиусе eps.
        # Это избавляет от повторных пересчётов при обходе очереди.
        neighbors = [np.where(dist[i] <= self.eps)[0] for i in range(n)]

        # Ядровые точки - те, у которых соседей не меньше min_samples
        # (включая саму точку, поэтому сравнение >=, а не >)
        is_core = np.array([len(neighbors[i]) >= self.min_samples for i in range(n)])

        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        # Основной обход: запускаем рост кластера только из не посещённых
        # ядровых точек - остальные либо шум, либо граничные и достанутся
        # уже существующему кластеру
        for i in range(n):
            if visited[i] or not is_core[i]:
                continue

            visited[i] = True
            labels[i] = cluster_id

            # Расширяем кластер в ширину (очередь):
            # добавляем всех соседей ядровых точек, пока не кончатся
            queue = list(neighbors[i])
            while queue:
                j = queue.pop()

                if not visited[j]:
                    # Точка ещё не встречалась - помечаем кластером,
                    # а если она ядровая - её соседи тоже идут в очередь
                    visited[j] = True
                    labels[j] = cluster_id
                    if is_core[j]:
                        queue.extend(neighbors[j])
                elif labels[j] == -1:
                    # Точка была помечена как шум, но сейчас дотянулась
                    # до кластера - перекрашиваем в граничную
                    labels[j] = cluster_id

            cluster_id += 1

        return labels
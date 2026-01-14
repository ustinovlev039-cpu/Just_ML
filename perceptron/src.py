from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Perceptron:
    lr: float = 0.1          # скорость обучения
    n_epochs: int = 100      # число эпох
    add_bias: bool = True    # добавлять ли bias (смещение)
    random_state: int | None = 42

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        # ожидаем метки {0,1}; если у тебя {-1,1}, можно адаптировать
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("y должен содержать метки 0/1")

        if self.add_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]  # столбец bias=1

        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(0.0, 0.01, size=X.shape[1])

        for _ in range(self.n_epochs):
            ошибок = 0
            for xi, yi in zip(X, y):
                y_pred = self._predict_row(xi)
                update = self.lr * (yi - y_pred)
                if update != 0.0:
                    self.w_ += update * xi
                    ошибок += 1
            # Можно раскомментировать для отладки:
            # print("ошибок:", ошибок)
            if ошибок == 0:
                break

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.add_bias:
            X = np.c_[np.ones((X.shape[0], 1)), X]
        scores = X @ self.w_
        return (scores >= 0.0).astype(int)

    def _predict_row(self, xi):
        return 1 if (xi @ self.w_) >= 0.0 else 0


# Пример: логическое AND
if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    p = Perceptron(lr=0.1, n_epochs=50, add_bias=True).fit(X, y)
    print("weights:", p.w_)
    print("pred:", p.predict(X))

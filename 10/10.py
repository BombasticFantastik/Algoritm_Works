import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------
# Тестовые функции
# ------------------------------------------------------------

def skin_function(x):
    """Skin function: smooth with several local extrema"""
    return -(np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2))

def forest_function(x):
    """Forest function: mix of smooth and non-differentiable extrema"""
    return -(np.abs(np.sin(x[0]) * np.cos(x[1])) + 0.25 * np.sqrt(x[0]**2 + x[1]**2))

def megacity_function(x):
    """Megacity function: discrete-like, flat floor with sharp peaks"""
    return -(
        np.floor(np.abs(np.sin(2 * x[0]) + np.cos(2 * x[1]))) +
        0.1 * (x[0]**2 + x[1]**2)
    )

# Выбор функции
test_function = skin_function  # or forest_function / megacity_function

# ------------------------------------------------------------
# GWO Implementation
# ------------------------------------------------------------

class GreyWolfOptimizer:
    def __init__(self, func, n_wolves=20, n_leaders=3, n_dim=2,
                 bounds=[(-5, 5), (-5, 5)], n_iter=100):
        self.func = func
        self.n_wolves = n_wolves
        self.n_leaders = n_leaders
        self.n_dim = n_dim
        self.bounds = np.array(bounds)
        self.n_iter = n_iter

        # Инициализация
        self.wolves = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(n_wolves, n_dim)
        )
        self.fitness = np.array([self.func(w) for w in self.wolves])
        
        # Лучшее найденное решение
        self.best_pos = self.wolves[np.argmax(self.fitness)].copy()
        self.best_val = np.max(self.fitness)

        # Для визуализации
        self.history = []

    def optimize(self):
        for t in range(self.n_iter):
            a = 2.0 * (1.0 - t / self.n_iter)  # линейное уменьшение от 2 до 0

            # Сортировка по фитнесу (максимизация)
            idx = np.argsort(self.fitness)[::-1]
            self.wolves = self.wolves[idx]
            self.fitness = self.fitness[idx]

            # Обновление глобального лучшего
            if self.fitness[0] > self.best_val:
                self.best_val = self.fitness[0]
                self.best_pos = self.wolves[0].copy()

            # Сохранить для визуализации
            self.history.append(self.wolves.copy())

            # Обновление позиций
            wolves_new = np.copy(self.wolves)

            # Обновление лидеров (альфа, бета, ..., до n_leaders)
            for i in range(self.n_leaders):
                for d in range(self.n_dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    wolves_new[i, d] = (
                        self.best_pos[d] - A * (C * self.best_pos[d] - self.wolves[i, d])
                    )

            # Обновление омег (остальные волки)
            for i in range(self.n_leaders, self.n_wolves):
                for d in range(self.n_dim):
                    position_sum = 0.0
                    for j in range(self.n_leaders):
                        r1, r2 = np.random.rand(), np.random.rand()
                        A = 2 * a * r1 - a
                        C = 2 * r2
                        position_sum += (
                            self.wolves[j, d] - A * (C * self.wolves[j, d] - self.wolves[i, d])
                        )
                    wolves_new[i, d] = position_sum / self.n_leaders

            # Приведение к границам
            wolves_new = np.clip(wolves_new, self.bounds[:, 0], self.bounds[:, 1])
            self.wolves = wolves_new
            self.fitness = np.array([self.func(w) for w in self.wolves])

        return self.best_pos, self.best_val

# ------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------

# Настройки
n_wolves = 20
n_leaders = 3
n_iter = 50
bounds = [(-5, 5), (-5, 5)]

# Инициализация и запуск
gwo = GreyWolfOptimizer(
    func=test_function,
    n_wolves=n_wolves,
    n_leaders=n_leaders,
    n_dim=2,
    bounds=bounds,
    n_iter=n_iter
)

print("Запуск оптимизации...")
best_pos, best_val = gwo.optimize()
print(f"Лучшее решение: {best_pos}, значение функции: {-best_val:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_title("GWO Optimization")
scat = ax.scatter([], [], c='gray', s=30, alpha=0.7, label='Волки')
leader_scat = ax.scatter([], [], c='red', s=80, label='Лидеры')
best_scat = ax.scatter([], [], c='gold', s=100, marker='*', label='Лучшее')

#сетка
x = np.linspace(bounds[0][0], bounds[0][1], 100)
y = np.linspace(bounds[1][0], bounds[1][1], 100)
X, Y = np.meshgrid(x, y)
Z = np.array([[test_function([xi, yi]) for xi in x] for yi in y])
contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.5)

def animate(frame):
    wolves = gwo.history[frame]
    leaders = wolves[:n_leaders]
    best = gwo.best_pos if frame == len(gwo.history) - 1 else wolves[0]

    scat.set_offsets(wolves[n_leaders:])
    leader_scat.set_offsets(leaders)
    best_scat.set_offsets([best])
    return scat, leader_scat, best_scat

anim = FuncAnimation(fig, animate, frames=len(gwo.history), interval=200, blit=True)
ax.legend()
plt.show()
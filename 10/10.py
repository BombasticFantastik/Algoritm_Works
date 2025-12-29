import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def skin_function(x):
    return -(np.sin(x[0]) * np.cos(x[1]) + 0.1 * (x[0]**2 + x[1]**2))

def forest_function(x):
    return -(np.abs(np.sin(x[0]) * np.cos(x[1])) + 0.25 * np.sqrt(x[0]**2 + x[1]**2))

def megacity_function(x):
    return -(
        np.floor(np.abs(np.sin(2 * x[0]) + np.cos(2 * x[1]))) +
        0.1 * (x[0]**2 + x[1]**2)
    )

# Выберите одну из функций
test_function = skin_function  # или forest_function, megacity_function

class GreyWolfOptimizer:
    def __init__(self, func, n_wolves=20, n_dim=2, bounds=[(-5, 5), (-5, 5)], n_iter=50):
        self.func = func
        self.n_wolves = n_wolves
        self.n_dim = n_dim
        self.bounds = np.array(bounds)
        self.n_iter = n_iter

        self.wolves = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(n_wolves, n_dim)
        )
        self.fitness = np.array([self.func(wolf) for wolf in self.wolves])

        #лучшее решение
        best_idx = np.argmax(self.fitness)
        self.best_pos = self.wolves[best_idx].copy()
        self.best_val = self.fitness[best_idx]
        self.history = []

    def optimize(self):
        for t in range(self.n_iter):
            idx_sorted = np.argsort(self.fitness)[::-1]  
            alpha = self.wolves[idx_sorted[0]].copy()
            beta  = self.wolves[idx_sorted[1]].copy()
            delta = self.wolves[idx_sorted[2]].copy()

            if self.fitness[idx_sorted[0]] > self.best_val:
                self.best_val = self.fitness[idx_sorted[0]]
                self.best_pos = alpha.copy()

            self.history.append(self.wolves.copy())

            a = 2.0 * (1.0 - t / self.n_iter)  

            wolves_new = np.empty_like(self.wolves)

            for i in range(self.n_wolves):
                pos_new = np.zeros(self.n_dim)
                for d in range(self.n_dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[d] - self.wolves[i, d])
                    X1 = alpha[d] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[d] - self.wolves[i, d])
                    X2 = beta[d] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[d] - self.wolves[i, d])
                    X3 = delta[d] - A3 * D_delta

                    pos_new[d] = (X1 + X2 + X3) / 3.0

                wolves_new[i] = pos_new

            wolves_new = np.clip(wolves_new, self.bounds[:, 0], self.bounds[:, 1])
            self.wolves = wolves_new
            self.fitness = np.array([self.func(wolf) for wolf in self.wolves])

        return self.best_pos, self.best_val

if __name__ == "__main__":
    n_wolves = 20
    n_iter = 60
    bounds = [(-5, 5), (-5, 5)]

    gwo = GreyWolfOptimizer(
        func=test_function,
        n_wolves=n_wolves,
        n_dim=2,
        bounds=bounds,
        n_iter=n_iter
    )

    print("Оптимизация...")
    best_pos, best_val = gwo.optimize()
    print(f"Найдено решение: {best_pos}")
    print(f"Значение функции: {-best_val:.6f}  (оригинал максимизируется)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title("Grey Wolf Optimizer (GWO) — Каноническая версия")
    
    x = np.linspace(bounds[0][0], bounds[0][1], 100)
    y = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[test_function([xi, yi]) for xi in x] for yi in y])
    ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.5)

    scat_omega = ax.scatter([], [], c='gray', s=30, alpha=0.7, label='Омеги')
    scat_alpha = ax.scatter([], [], c='red', s=100, label='Альфа')
    scat_beta  = ax.scatter([], [], c='blue', s=80, label='Бета')
    scat_delta = ax.scatter([], [], c='green', s=70, label='Дельта')
    scat_best  = ax.scatter([], [], c='gold', s=120, marker='*', label='Лучшее')

    def animate(frame):
        wolves = gwo.history[frame]
        fitness_frame = np.array([test_function(w) for w in wolves])
        idx = np.argsort(fitness_frame)[::-1]
        sorted_wolves = wolves[idx]

        alpha = sorted_wolves[0]
        beta  = sorted_wolves[1]
        delta = sorted_wolves[2]
        omegas = sorted_wolves[3:]

        scat_omega.set_offsets(omegas)
        scat_alpha.set_offsets([alpha])
        scat_beta.set_offsets([beta])
        scat_delta.set_offsets([delta])
        scat_best.set_offsets([gwo.best_pos if frame == len(gwo.history)-1 else alpha])
        return scat_omega, scat_alpha, scat_beta, scat_delta, scat_best
    ani = FuncAnimation(fig, animate, frames=len(gwo.history), interval=300, blit=True)
    ax.legend()
    plt.show()
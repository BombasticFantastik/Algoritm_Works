import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def hilly(x):
    return np.sum(np.cos(x) ** 4)

class DOS:
    def __init__(self, func, bounds, pop_size=30, movement_factor=0.95, max_iter=200):
        self.func = func
        self.bounds = np.array(bounds)  
        self.dim = len(bounds)
        self.pop_size = pop_size
        self.movement_factor = movement_factor
        self.max_iter = max_iter

        #создание частиц
        self.positions = self._deterministic_init()
        self.velocities = np.zeros((pop_size, self.dim))
        self.slopes = np.zeros(pop_size, dtype=int)  # -1, 0, 1
        self.prev_fitness = np.full(pop_size, -np.inf)
        self.current_fitness = np.array([self.func(p) for p in self.positions])

        best_idx = np.argmax(self.current_fitness)
        self.global_best_pos = self.positions[best_idx].copy()
        self.global_best_fit = self.current_fitness[best_idx]

        self.history_positions = [self.positions.copy()]
        self.history_fitness = [self.current_fitness.copy()]

    def _deterministic_init(self):
        positions = np.zeros((self.pop_size, self.dim))

        if self.dim == 2:
            side = int(np.sqrt(self.pop_size))
            if side * side < self.pop_size:
                side += 1

            x_coords = np.linspace(self.bounds[0, 0], self.bounds[0, 1], side)
            y_coords = np.linspace(self.bounds[1, 0], self.bounds[1, 1], side)

            idx = 0
            for x in x_coords:
                for y in y_coords:
                    if idx < self.pop_size:
                        positions[idx] = [x, y]
                        idx += 1
                    else:
                        break
                if idx >= self.pop_size:
                    break

        else:
            for d in range(self.dim):
                low, high = self.bounds[d]
                positions[:, d] = np.linspace(low, high, self.pop_size)

        return positions

    def _clip_position(self, pos):
        return np.clip(pos, self.bounds[:, 0], self.bounds[:, 1])

    def run(self):
        for it in range(self.max_iter):
            #сохранение фитнеса
            self.prev_fitness = self.current_fitness.copy()

            self.positions += self.velocities
            self.positions = np.array([self._clip_position(p) for p in self.positions])
            self.current_fitness = np.array([self.func(p) for p in self.positions])

            best_idx = np.argmax(self.current_fitness)
            if self.current_fitness[best_idx] > self.global_best_fit:
                self.global_best_pos = self.positions[best_idx].copy()
                self.global_best_fit = self.current_fitness[best_idx]

            for i in range(self.pop_size):
                fit_diff = self.current_fitness[i] - self.prev_fitness[i]

                if self.slopes[i] == 0: 
                    if fit_diff > 1e-12:
                        self.slopes[i] = 1
                    elif fit_diff < -1e-12:
                        self.slopes[i] = -1

                elif self.slopes[i] == 1 and fit_diff < -1e-12:
                    #отскок
                    self.velocities[i] *= -0.5
                    self.slopes[i] = -1

                elif self.slopes[i] == -1 and fit_diff < -1e-12:
                    #роение
                    self.velocities[i] += (self.global_best_pos - self.positions[i]) * self.movement_factor
                    self.slopes[i] = 0

                #проверка на нулевую скорость
                if np.all(np.abs(self.velocities[i]) < 1e-10):
                    self.velocities[i] = (self.global_best_pos - self.positions[i]) * self.movement_factor
                    self.slopes[i] = 0

            self.history_positions.append(self.positions.copy())
            self.history_fitness.append(self.current_fitness.copy())

        return self.global_best_pos, self.global_best_fit

#визуализация
def animate_dos(dos_optimizer):
    history_pos = np.array(dos_optimizer.history_positions)  # (iter+1, pop, dim)

    x = np.linspace(dos_optimizer.bounds[0, 0], dos_optimizer.bounds[0, 1], 200)
    y = np.linspace(dos_optimizer.bounds[1, 0], dos_optimizer.bounds[1, 1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[hilly([xi, yi]) for xi in x] for yi in y])

    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, ax=ax, shrink=0.8)
    ax.set_title("DOS: Траектории частиц")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    scat = ax.scatter([], [], c='red', s=20, edgecolor='k')

    def animate(frame):
        pos = history_pos[frame]
        scat.set_offsets(pos)
        return scat,

    anim = FuncAnimation(
        fig, animate, frames=len(history_pos), interval=200, blit=True, repeat=False
    )
    plt.tight_layout()
    plt.show()

#main
if __name__ == "__main__":
    # Границы поиска для 2D
    bounds = [(-5, 5), (-5, 5)]

    dos = DOS(
        func=hilly,
        bounds=bounds,
        pop_size=30,
        movement_factor=0.95,
        max_iter=100
    )

    best_pos, best_val = dos.run()
    print(f"\nЛучшее решение: {best_pos}")
    print(f"Лучшее значение фитнеса: {best_val:.6f}")

    animate_dos(dos)
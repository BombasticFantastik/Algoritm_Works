import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# #def himmelblau(x):
#     return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def fitness(x):
    return -rastrigin(x)

def scale(val, in_min, in_max, out_min, out_max, reverse=False):
    if out_min == out_max:
        return out_min
    if in_min == in_max:
        return (out_min + out_max) / 2.0
    val = np.clip(val, in_min, in_max)
    res = ((val - in_min) * (out_max - out_min) / (in_max - in_min)) + out_min
    return out_max - (res - out_min) if reverse else res

#обрезка
def ensure_bounds(x, bounds):
    return np.clip(x, bounds[:, 0], bounds[:, 1])

class Ant:
    def __init__(self, dim, bounds):
        self.dim = dim
        self.bounds = bounds
        self.position = np.random.uniform(bounds[:, 0], bounds[:, 1])
        self.position_last = self.position.copy()
        self.best_position = self.position.copy()
        self.fitness = -np.inf
        self.best_fitness = -np.inf


class ContinuousACO:
    def __init__(self, n_ants=25, dim=2, bounds=None,
                 pheromone_effect=1.0,
                 path_length_effect=1.0,
                 pheromone_radius=0.8,   
                 path_deviation=0.5):    
        self.n_ants = n_ants
        self.dim = dim
        self.bounds = bounds if bounds is not None else np.array([[-6, 6], [-6, 6]])
        self.pheromone_effect = pheromone_effect
        self.path_length_effect = path_length_effect
        self.pheromone_radius = pheromone_radius
        self.path_deviation = path_deviation

        self.ants = [Ant(dim, self.bounds) for _ in range(n_ants)]
        self.global_best_position = None
        self.global_best_fitness = -np.inf

    def prepare(self, first_time=False):
        if first_time:
            self.global_best_fitness = -np.inf
            for ant in self.ants:
                ant.position = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
                ant.position_last = ant.position.copy()
                ant.best_position = ant.position.copy()
                ant.fitness = -np.inf
                ant.best_fitness = -np.inf
        else:
            self.move_ants()

    def dwell(self):
        for ant in self.ants:
            ant.position_last = ant.position.copy()
            if ant.fitness > ant.best_fitness:
                ant.best_fitness = ant.fitness
                ant.best_position = ant.position.copy()
            if ant.fitness > self.global_best_fitness:
                self.global_best_fitness = ant.fitness
                self.global_best_position = ant.position.copy()

    def move_ants(self):
        positions_last = np.array([ant.position_last for ant in self.ants])
        diff = positions_last[:, np.newaxis, :] - positions_last[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(distances, np.inf)

        min_dist = np.min(distances, axis=1, keepdims=True)
        max_dist = np.max(distances, axis=1, keepdims=True)
        range_dist = np.where(max_dist == min_dist, 1.0, max_dist - min_dist)
        norm_distances = (distances - min_dist) / range_dist

        fitnesses = np.array([ant.fitness for ant in self.ants])
        min_fit = np.min(fitnesses)
        max_fit = np.max(fitnesses)
        range_fit = max_fit - min_fit if max_fit != min_fit else 1.0
        norm_fitnesses = (fitnesses - min_fit) / range_fit

        for i in range(self.n_ants):
            scores = np.full(self.n_ants, -np.inf)
            for k in range(self.n_ants):
                if i == k:
                    continue
                rnd_ph = np.random.uniform(0.0, self.pheromone_effect)
                rnd_pt = np.random.uniform(0.0, self.path_length_effect)
                score = norm_fitnesses[k] * rnd_ph * (1.0 - norm_distances[i, k]) * rnd_pt
                scores[k] = score

            target_idx = np.argmax(scores)
            way_to_goal = distances[i, target_idx]
            radius_near_goal = way_to_goal * self.pheromone_radius
            end_way = way_to_goal + radius_near_goal

            x = np.random.uniform(-1.0, 1.0)
            y = x * x
            if x > 0:
                y = scale(y, 0.0, 1.0, way_to_goal, end_way, reverse=False)
            else:
                y = scale(y, 0.0, 1.0, 0.0, way_to_goal, reverse=True)

            if way_to_goal == 0:
                direction = np.zeros(self.dim)
            else:
                direction = (positions_last[target_idx] - positions_last[i]) / way_to_goal

            new_pos = positions_last[i] + direction * y
            deviation = direction * np.random.uniform(-1.0, 1.0, self.dim) * self.path_deviation
            new_pos += deviation
            new_pos = ensure_bounds(new_pos, self.bounds)

            self.ants[i].position = new_pos

def main():
    n_ants = 30
    bounds = np.array([[-6.0, 6.0], [-6.0, 6.0]])

    aco = ContinuousACO(
        n_ants=n_ants,
        dim=2,
        bounds=bounds,
        pheromone_effect=1.0,
        path_length_effect=1.0,
        pheromone_radius=0.8,    
        path_deviation=0.5       
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_title('Непрерывный ACO')
    scat = ax.scatter([], [], c='blue', s=40, label='Муравьи')
    best_scat = ax.scatter([], [], c='red', s=120, marker='*', label='Лучшая точка')

    
    x = np.linspace(bounds[0, 0], bounds[0, 1], 150)
    y = np.linspace(bounds[1, 0], bounds[1, 1], 150)
    X, Y = np.meshgrid(x, y)
    Z = np.array([rastrigin(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())])
    Z = Z.reshape(X.shape)
    ax.contour(X, Y, Z, levels=20, cmap='Greys', alpha=0.6)

    #инициализация
    aco.prepare(first_time=True)
    for ant in aco.ants:
        ant.fitness = fitness(ant.position)
    aco.dwell()

    def update(frame):
        aco.prepare(first_time=False)
        for ant in aco.ants:
            ant.fitness = fitness(ant.position)
        aco.dwell()

        positions = np.array([ant.position for ant in aco.ants])
        scat.set_offsets(positions)
        if aco.global_best_position is not None:
            best_scat.set_offsets([aco.global_best_position])
        current_val = rastrigin(aco.global_best_position) if aco.global_best_position is not None else np.inf
        ax.set_xlabel(f'Итерация: {frame + 1} | rastrigin(лучшее) = {current_val:.4f}')
        return scat, best_scat

    ani = FuncAnimation(fig, update, frames=120, interval=200, blit=False, repeat=False)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
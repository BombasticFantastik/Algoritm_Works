import numpy as np
from ABHA import ABHA
import math

def rastrigin_max(x):
    #A = 10
    return -(2 + np.sum(x**2 -np.cos(2 * np.pi * x)))

dim = 5
bounds = [(-5, 5) for _ in range(dim)]

bounds = [(-5, 5), (-5, 5)]

#new
abha_vis = ABHA(
    objective_func=rastrigin_max,
    bounds=bounds,
    pop_size=30,
    max_iter=120
)

best_pos, best_val = abha_vis.optimize(visualize=True)
print("Лучшая позиция:", best_pos)
print("Лучшее значение:", best_val)
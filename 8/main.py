import numpy as np
from ABHA import ABHA
import math

def sphere(x):
    return -np.sum(math.cos(x)/(x**2))
dim = 5
bounds = [(-5, 5) for _ in range(dim)]

abha = ABHA(
    objective_func=sphere,
    dim=dim,
    bounds=bounds,
    pop_size=20,
    max_iter=200
)

best_pos, best_val = abha.optimize()
print("Лучшая позиция:", best_pos)
print("Лучшее значение:", best_val)
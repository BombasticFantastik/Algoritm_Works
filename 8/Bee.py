import numpy as np
#пчела
class Bee:
    STATE_NOVICE = 0
    STATE_EXPERIENCED = 1
    STATE_SEARCH = 2
    STATE_SOURCE = 3

    def __init__(self, dim, bounds, init_step_size):
        self.dim = dim
        self.bounds = bounds
        self.position = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
        self.best_position = self.position.copy()
        self.direction = np.random.uniform(-1, 1, dim)
        self.cost = -np.inf
        self.prev_cost = -np.inf
        self.best_cost = -np.inf
        self.step_size = init_step_size
        self.state = self.STATE_NOVICE
        self.search_counter = 0

        self.p_srs = 0.0  # случайный поиск
        self.p_rul = 0.0  # следование танцу
        self.p_ab = 0.0   # отказ от источника

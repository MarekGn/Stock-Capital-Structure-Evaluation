import numpy as np


class Individual():
    def __init__(self, idv_size):
        self.fitness = 0
        self.probability = 0
        self.genome = np.array([np.random.dirichlet(np.ones(idv_size), 1)[0][:-1]])
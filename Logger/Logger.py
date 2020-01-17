import numpy as np


class Logger():
    def __init__(self):
        self.log_idv = []

    def save_best_genome(self, filename):
        np.savetxt("Results/{}".format(filename), self.log_idv)

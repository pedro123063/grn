# Modules/Datasets.py

import numpy as np

# chamar o modelo, treinar e testar
# time-series

class TemporalTrainTest:
    def __init__(self, t, y, train_percentage=1.0):
        assert 0 < train_percentage <= 1.0
        self.t = t
        self.y = y
        self.train_percentage = train_percentage
        self._split()

    def _split(self):
        n = len(self.t)
        n_train = int(np.ceil(self.train_percentage * n))

        self.t_train = self.t[:n_train]
        self.y_train = self.y[:, :n_train]

        self.t_test = self.t[n_train:]
        self.y_test = self.y[:, n_train:]

    @property
    def has_test(self):
        return len(self.t_test) > 0

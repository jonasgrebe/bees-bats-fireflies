from base import BaseSearchAlgorithm
import numpy as np

class BatAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bat", **kwargs)

        self.a = self.params['a']
        self.r = self.params['r']
        self.q_min = self.params['q_min']
        self.q_max = self.params['q_max']

    
    def initialize(self):
        self.solutions = np.random.uniform(self.range_min, self.range_max, (self.n, self.d))
        self.q = np.zeros(self.n)
        self.v = np.zeros((self.n, self.d))
        self.best = self.get_best_solution()
        self.b = np.zeros((self.n, self.d))


    def execute_search_step(self):
        self.q = np.random.uniform(self.q_min, self.q_max, self.n)
        for i in range(self.n):
            for j in range(self.d):
                self.v[i][j] = self.v[i][j] + (self.solutions[i][j] - self.best[j]) * self.q[i]
                self.b[i][j] = self.solutions[i][j] + self.v[i][j]
                self.b[i][j] = np.clip(self.b[i][j], self.range_min, self.range_max)

            if np.random.uniform(0, 1) > self.r:
                for j in range(self.d):
                    self.b[i][j] = self.best[j] + 10e-3 * np.random.normal(0, 1)
                    self.b[i][j] = np.clip(self.b[i][j], self.range_min, self.range_max)

            if self.compare_objective_value(self.b[i], self.solutions[i]) < 0:
                self.solutions[i] = self.b[i]

            if self.compare_objective_value(self.b[i], self.best) < 0:
                self.best = self.b[i]

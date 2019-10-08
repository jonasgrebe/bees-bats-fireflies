from base import BaseSearchAlgorithm
import numpy as np

class BatAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bat", **kwargs)

        self.a = self.params['a']
        self.r_max = self.params['r_max']
        self.alpha = self.params['alpha']
        self.gamma = self.params['gamma']
        self.f_min = self.params['f_min']
        self.f_max = self.params['f_max']
    
    def initialize(self):
        self.solutions = np.zeros(shape=(self.n, self.d))
        for i in range(self.n):
            self.solutions[i] = self.random_uniform_in_ranges()
            
        self.q = np.zeros(self.n)
        self.v = np.zeros((self.n, self.d))
        self.b = np.zeros((self.n, self.d))
            
        self.a = np.repeat(self.a, self.n)
        self.r = np.zeros(self.n)

    def execute_search_step(self, t):
        self.f = np.random.uniform(self.f_min, self.f_max, self.n)

        for i in range(self.n):
            self.v[i] = self.v[i] + (self.solutions[i] - self.best_solution) * self.f[i]
            self.b[i] = self.solutions[i] + self.v[i]
            
            if np.random.uniform(0, 1) < self.r[i]:
                self.b[i] = self.best_solution + np.mean(self.a) * np.random.normal(0, 1, self.d)
                
            self.b[i] = self.clip_to_ranges(self.b[i])
            
            if self.compare_objective_value(self.b[i], self.solutions[i]) < 0 and np.random.uniform(0, 1) < self.a[i]:
                self.solutions[i] = np.copy(self.b[i])
                self.a[i] *= self.alpha
                self.r[i] = self.r_max * (1-np.exp(-self.gamma * t))
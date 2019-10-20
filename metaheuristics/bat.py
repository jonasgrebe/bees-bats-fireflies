from .base import BaseSearchAlgorithm
import numpy as np

class BatAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bat", **kwargs)

        self.a_init = self.params['a'] # initial loudness of all bats
        self.r_max = self.params['r_max'] # maximum pulse rate of bats
        self.alpha = self.params['alpha'] # loudness decreasing factor
        self.gamma = self.params['gamma'] # pulse rate increasing factor
        self.f_min = self.params['f_min'] # minimum sampled frequency
        self.f_max = self.params['f_max'] # maximum sampled frequency
    
    
    def initialize(self):
        self.solutions = np.zeros(shape=(self.n, self.d))
        for i in range(self.n):
            self.solutions[i] = self.random_uniform_in_ranges()
            
        self.q = np.zeros(self.n)
        self.v = np.zeros((self.n, self.d))
        self.b = np.zeros((self.n, self.d))
            
        self.a = np.repeat(self.a_init, self.n)
        self.r = np.zeros(self.n)
        
        
    def constraints_satisfied(self):
        return self.f_min < self.f_max


    def execute_search_step(self, t):
        self.f = np.random.uniform(self.f_min, self.f_max, self.n)

        for i in range(self.n):
            
            if np.random.uniform(0, 1) < self.r[i]:
                self.b[i] = self.best_solution + np.mean(self.a) * np.random.uniform(-1, 1, self.d)
            else:
                self.v[i] = self.v[i] + (self.solutions[i] - self.best_solution) * self.f[i]
                self.b[i] = self.solutions[i] + self.v[i]
            
            self.b[i] = self.clip_to_ranges(self.b[i])
            
            if self.compare_objective_value(self.b[i], self.solutions[i]) < 0:
                if np.random.uniform(0, 1) < self.a[i]:
                    self.solutions[i] = self.b[i]
                    self.a[i] *= self.alpha
                    self.r[i] = self.r_max * (1-np.exp(-self.gamma * t))
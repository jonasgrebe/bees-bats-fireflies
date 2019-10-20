from .base import BaseSearchAlgorithm
import numpy as np

class FireflyAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("firefly", **kwargs)

        self.alpha = self.params['alpha'] # neighbor sphere radius
        self.beta_max = self.params['beta_max'] # maximum attractivneness
        self.gamma = self.params['gamma'] # attractiveness descreasing factor


    def initialize(self):
        self.solutions = np.zeros(shape=(self.n, self.d))
        for i in range(self.n):
            self.solutions[i] = self.random_uniform_in_ranges()
            
            
    def execute_search_step(self, t):
        for i in range(self.n):
            for j in range(self.n):

                if self.light_intensity(i) < self.light_intensity(j):
                    diff_ij = (self.solutions[j] - self.solutions[i])
                    r_ij = np.sqrt(np.sum(np.square(diff_ij)))
                    beta_ij = self.beta_max * np.exp(-self.gamma * r_ij**2)
                    self.solutions[i] += beta_ij * diff_ij + self.alpha * np.random.uniform(-0.5, 0.5, self.d)
                    self.clip_to_ranges(self.solutions[i])


    def light_intensity(self, i):
        if self.objective == 'min':
            return 1 / (1e-16+self.objective_fct(self.solutions[i]))
        elif self.objective == 'max':
            return self.objective_fct(self.solutions[i])



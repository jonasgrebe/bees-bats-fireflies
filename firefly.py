from base import BaseSearchAlgorithm
import numpy as np

class FireflyAlgorithm(BaseSearchAlgorithm):
    
    def __init__(self, **kwargs):
        super().__init__("firefly", **kwargs)
        
        self.alpha = self.params['alpha']
        self.beta0 = self.params['beta0']
        self.gamma = self.params['gamma']
        
    
    def initialize(self):
        self.solutions = np.random.uniform(self.range_min, self.range_max, (self.n, self.d))
            
    
    def execute_search_step(self):
        for i in range(self.n):
            for j in range(self.n):
                
                if self.light_intensity(i) < self.light_intensity(j):
                    diff_ij = (self.solutions[j] - self.solutions[i])
                    r_ij = np.sqrt(np.sum(np.square(diff_ij)))
                    beta_ij = self.beta0 * np.exp(-self.gamma * r_ij**2)
                    self.solutions[i] += beta_ij * diff_ij + self.alpha * np.random.uniform(-0.5, 0.5, self.d)
                
    
    def light_intensity(self, i):
        if self.objective == 'min':
            return 1 / self.objective_fct(self.solutions[i])
        elif self.objective == 'max':
            return self.objective_fct(self.solutions[i])


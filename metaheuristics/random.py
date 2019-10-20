from .base import BaseSearchAlgorithm
import numpy as np

class RandomSamplingAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("random", **kwargs)

    def initialize(self):
        self.solutions = np.zeros(shape=(self.n, self.d))
        
    def execute_search_step(self, t):
        
        for i in range(self.n):
            self.solutions[i] = self.random_uniform_in_ranges()
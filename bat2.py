from search import SwarmSearchAlgorithm

import numpy as np
import matplotlib.pyplot as plt


class BatAlgorithm(SwarmSearchAlgorithm):
    
    def __init__(self, d, np, range_min, range_max, a, r, qmin, qmax):
        super().__init__("bat", d, np, range_min, range_max)
        self
        self.a = a
        self.r = r
        self.qmax = qmax
        self.qmin = qmin
    
    def initialize(self):
        self.population = np.random.uniform(self.range_min, self.range_max, (self.np, self.d))
        self.q = np.zeros(self.np)
        self.v = np.zeros((self.np, self.d))
        self.best = self.population[np.argmin([self.cost_function(s) for s in self.population])]
        self.f_min = self.cost_function(self.best) 
        self.b = np.zeros((self.np, self.d))
        
        
    def single_search_step(self):
        for i in range(self.np):
            self.q[i] = np.random.uniform(self.qmin, self.qmax)
            for j in range(self.d):
                self.v[i][j] = self.v[i][j] + (self.population[i][j] - self.best[j]) * self.q[i]
                self.b[i][j] = self.population[i][j] + self.v[i][j]
                self.b[i][j] = np.clip(self.b[i][j], self.range_min, self.range_max)

            if np.random.uniform(0, 1) > self.r:
                for j in range(self.d):
                    self.b[i][j] = self.best[j] + 10e-3 * np.random.normal(0, 1)
                    self.b[i][j] = np.clip(self.b[i][j], self.range_min, self.range_max)
                    
            f_new = self.cost_function(self.b[i])

            if (f_new < self.cost_function(self.population[i])) and (np.random.uniform(0, 1) < self.a):
                self.population[i] = self.b[i]

            if f_new < self.f_min:
                self.best = self.b[i]
                self.f_min = f_new
        
        
    def highlight_population(self, ax):
        if self.d == 1:
            ax.scatter(self.best, self.cost_function(self.best))
        if self.d == 2:
            ax.scatter(self.best[0], self.best[1], c='red')
         
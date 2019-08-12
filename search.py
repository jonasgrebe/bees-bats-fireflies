from abc import ABC as abstract
from abc import abstractmethod

import numpy as np
import matplotlib.pyplot as plt

class SwarmSearchAlgorithm(abstract):
    
    def __init__(self, name, d, np, range_min, range_max):
        self.name = name
        
        self.d = d
        self.np = np
        
        self.range_min = range_min
        self.range_max = range_max
        
        self.population = []
    
    
    def search(self, score_function, T, k=1, minimize=True, visualize=False):
        self.score_function = score_function if not minimize else lambda x: -score_function(x)
        self.initialize()
        
        if visualize:
            self.visualize()
        
        for t in range(T):
            self.single_search_step()
            if visualize:
                self.visualize(t)
        
        return self.get_best_k_solutions(k, minimize)
    
    
    def visualize(self, t=0):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect='equal')
        
        self.plot_score_function(ax)
        self.plot_population(ax)
        self.highlight_population(ax)
        
        plt.savefig(f"images/{self.name}/{self.name}_t{t}.png")
        # plt.show()
        plt.close()
        
    
    def plot_score_function(self, ax):
        
        if self.d == 1:
            ax.set_xlim([self.range_min, self.range_max])
            
            x = np.linspace(self.range_min, self.range_max, 100)
            y = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                y[i] = self.score_function(x[i])
            ax.plot(x, y)
            
        elif self.d == 2:
            ax.set_xlim([self.range_min, self.range_max])
            ax.set_ylim([self.range_min, self.range_max]) 
            
            x = np.linspace(self.range_min, self.range_max, 100)
            y = np.linspace(self.range_min, self.range_max, 100)
            
            X, Y = np.meshgrid(x, y)
            
            XY = np.array((X, Y)).T
            Z = np.zeros(XY.shape[:-1])
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    Z[i,j] = self.score_function(XY[i,j])
                  
            ax.contour(X, Y, Z, colors='black');
        
        
    def plot_population(self, ax):
        if self.d == 1:
            ax.scatter(self.population, [self.score_function(i) for i in self.population])
        elif self.d == 2:
            ax.scatter(self.population.T[0], self.population.T[1])
    
        
    @abstractmethod
    def initialize(self):
        raise NotImplementedError

    
    @abstractmethod
    def highlight_population(self, ax):
        raise NotImplementedError
    
    
    @abstractmethod
    def single_search_step():
        raise NotImplementedError
    
    
    def get_best_k_solutions(self, k, minimize):
        idxs = np.argsort([self.score_function(i) for i in self.population])
        self.population = self.population[idxs]
        return self.population[:k], [(-1 if minimize else 1) * self.score_function(i) for i in self.population[:k]]
        
    
    
    
        
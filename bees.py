
from search import SwarmSearchAlgorithm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class BeesAlgorithm(SwarmSearchAlgorithm):
    
    def __init__(self, d, np, range_min, range_max, ne, nb, nre, nrb, shrink_factor, stgn_lim):
        super().__init__("bees", d, np, range_min, range_max)
        
        self.ne = ne
        self.nb = nb
        self.nre = nre
        self.nrb = nrb
        self.shrink_factor = shrink_factor
        self.stgn_lim = stgn_lim
        self.initial_nght = 1.0
        
    
    def initialize(self):
        self.population = np.zeros((self.np, self.d))
        self.flower_patch = [None] * self.np
        self.nght = [self.initial_nght] * self.np
        self.best_sites = []
        
        for i in range(self.np):
            self.initialize_flower_patch(i)

    
    def single_search_step(self):                
        self.waggle_dance()
        
        for i in range(self.nb):
            self.local_search(i)
            self.abandon_sites(i)
            self.shrink_neighborhood(i)
            
        for i in range(self.nb, self.np):
            self.global_search(i)
    
        
    def highlight_population(self, ax):
        return
        for i in range(self.np):
            nght = self.flower_patch[i]['nght']
            sx, sy = self.population[i]
            ax.add_patch(patches.Rectangle((sx-nght, sy-nght), 2*nght, 2*nght, fill=False)) 
        
        
    def initialize_flower_patch(self, i):
        self.population[i] = self.create_random_scout()
        self.flower_patch[i] = {'foragers': 0, 'nght': self.initial_nght, 'stagnation': False,'stagnation_cnt': 0}
        
        
    def create_random_scout(self):
        return np.random.uniform(self.range_min, self.range_max, self.d)
    
    
    def create_random_forager(self, i):
        forager = np.zeros_like(self.population[i])
        for j in range(self.d):
            nght = self.flower_patch[i]['nght']
            forager[j] = np.clip(np.random.uniform(-1, 1) * nght + self.population[i][j], self.range_min, self.range_max)
        return forager
    
        
    def waggle_dance(self):
        idxs = np.argsort([self.cost_function(s) for s in self.population])
        self.population = self.population[idxs]
        
        self.flower_patch = list(np.array(self.flower_patch)[idxs])
        
        for i in range(self.ne):
            self.flower_patch[i]['foragers'] = self.nre
        for i in range(self.ne, self.nb):
            self.flower_patch[i]['foragers'] = self.nrb
        
    
    def local_search(self, i):
        for j in range(self.flower_patch[i]['foragers']):
            forager = self.create_random_forager(i)
            forager_score = self.cost_function(forager)
            if forager_score < self.cost_function(self.population[i]):
                self.population[i] = forager
                self.flower_patch[i]['stagnation'] = True
            
    
    def global_search(self, i):
        self.initialize_flower_patch(i)
        
        
    def shrink_neighborhood(self, i):
        if self.flower_patch[i]['stagnation']:
            self.flower_patch[i]['nght'] *= 0.8
        
        
    def abandon_sites(self, i):
        if self.flower_patch[i]['stagnation']:
            if self.flower_patch[i]['stagnation_cnt'] < self.stgn_lim:
                self.flower_patch[i]['stagnation_cnt'] += 1
            else:
                self.best_sites.append(self.population[i])
                self.initialize_flower_patch(i)
                
                
    def get_best_k_solutions(self, k, minimize):
        solutions = np.array(self.best_sites + list(self.population))
        idxs = np.argsort([self.cost_function(i) for i in solutions])
        solutions = solutions[idxs]
        best_k_solutions = solutions[:k]
        return best_k_solutions, [(1 if minimize else -1) * self.cost_function(i) for i in best_k_solutions]
        
        
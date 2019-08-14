from base import BaseSearchAlgorithm
import numpy as np

class BeesAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bees", **kwargs)

        self.nb = self.params['nb']
        self.ne = self.params['ne']
        self.nrb = self.params['nrb']
        self.nre = self.params['nre']
        self.shrink_factor = self.params['shrink_factor']
        self.stgn_lim = self.params['stgn_lim']
        
        self.initial_nght = 1.0

    
    def initialize(self):
        self.solutions = np.zeros((self.n, self.d))
        self.flower_patch = [None] * self.n
        self.nght = [self.initial_nght] * self.n
        
        for i in range(self.n):
            self.initialize_flower_patch(i)


    def execute_search_step(self):
        self.waggle_dance()
        
        for i in range(self.nb):
            self.local_search(i)
            self.abandon_sites(i)
            self.shrink_neighborhood(i)
            
        for i in range(self.nb, self.n):
            self.global_search(i)


    def initialize_flower_patch(self, i):
        self.solutions[i] = self.create_random_scout()
        self.flower_patch[i] = {'foragers': 0, 'nght': self.initial_nght, 'stagnation': False,'stagnation_cnt': 0}
        
    
    def create_random_scout(self):
        return np.random.uniform(self.range_min, self.range_max, self.d)
    
    
    def create_random_forager(self, i):
        forager = np.zeros_like(self.solutions[i])
        for j in range(self.d):
            nght = self.flower_patch[i]['nght']
            forager[j] = np.clip(np.random.uniform(-1, 1) * nght + self.solutions[i][j], self.range_min, self.range_max)
        return forager
    
    
    def waggle_dance(self):
        idxs = self.argsort_objective()
        self.solutions = self.solutions[idxs]
        self.flower_patch = np.array(self.flower_patch)[idxs].ravel()
        
        for i in range(self.ne):
            self.flower_patch[i]['foragers'] = self.nre
        for i in range(self.ne, self.nb):
            self.flower_patch[i]['foragers'] = self.nrb
    
    
    def local_search(self, i):
        for j in range(self.flower_patch[i]['foragers']):
            forager = self.create_random_forager(i)
            if self.compare_objective_value(forager, self.solutions[i]) < 0:
                self.solutions[i] = forager
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
                self.get_best_solution()
                self.initialize_flower_patch(i)
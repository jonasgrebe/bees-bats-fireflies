from base import BaseSearchAlgorithm
import numpy as np


class BeesAlgorithm(BaseSearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bees", **kwargs)

        self.nb = self.params['nb'] # number of (best) scouts that recruit foragers
        self.ne = self.params['ne'] # number of elite scouts that recruit more
        self.nrb = self.params['nrb'] # number recruited foragers per best scout
        self.nre = self.params['nre'] # number of recruited foragers per elite scout

        self.initial_size = 1.0


    def initialize(self):
        self.solutions = np.zeros((self.n, self.d))
        self.flower_patch = [None] * self.n
        self.size = [self.initial_size] * self.n

        for i in range(self.n):
            self.initialize_flower_patch(i)


    def execute_search_step(self, t):
        self.waggle_dance()

        for i in range(self.nb):
            self.local_search(i)

        for i in range(self.nb, self.n):
            self.global_search(i)


    def initialize_flower_patch(self, i):
        self.solutions[i] = self.create_random_scout()
        self.flower_patch[i] = {'size': self.initial_size}


    def create_random_scout(self):
        return self.random_uniform_in_ranges()


    def create_random_forager(self, i):
        nght = self.flower_patch[i]['size']
        forager = np.random.uniform(-1, 1) * nght + self.solutions[i]
        for j in range(self.d):
            forager[j] = np.clip(forager[j], self.range_min[j], self.range_max[j])
        return forager


    def waggle_dance(self):
        idxs = self.argsort_objective()
        self.solutions = self.solutions[idxs]
        self.flower_patch = np.array(self.flower_patch)[idxs].ravel()
        # recruitment is done in header of local search loop


    def local_search(self, i):
        for j in range(self.nrb if i < self.nb else self.nre):
            forager = self.create_random_forager(i)
            if self.compare_objective_value(forager, self.solutions[i]) < 0:
                self.solutions[i] = forager
                self.initialize_flower_patch(i)


    def global_search(self, i):
        self.initialize_flower_patch(i)
            

class ImprovedBeesAlgorithm(BeesAlgorithm):

    def __init__(self, **kwargs):
        super().__init__("bees", **kwargs)

        self.nb = self.params['nb'] # number of (best) scouts that recruit foragers
        self.ne = self.params['ne'] # number of elite scouts that recruit more
        self.nrb = self.params['nrb'] # number recruited foragers per best scout
        self.nre = self.params['nre'] # number of recruited foragers per elite scout
        self.sf = self.params['sf'] # shrinking factor
        self.sl = self.params['sl'] # stagnation limit

        self.initial_size = 1.0
        

    def execute_search_step(self, t):
        self.waggle_dance()

        for i in range(self.nb):
            self.local_search(i)
            self.abandon_sites(i)
            self.shrink_neighborhood(i)

        for i in range(self.nb, self.n):
            self.global_search(i)


    def initialize_flower_patch(self, i):
        self.solutions[i] = self.create_random_scout()
        self.flower_patch[i] = {'size': self.initial_size, 'scnt': self.sl}


    def shrink_neighborhood(self, i):
        self.flower_patch[i]['size'] *= self.sf


    def abandon_sites(self, i):
        if self.flower_patch[i]['scnt'] >  0:
            self.flower_patch[i]['scnt'] -= 1
        else:
            self.initialize_flower_patch(i)

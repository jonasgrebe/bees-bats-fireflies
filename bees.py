import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class BeesAlgorithm():
    
    def __init__(self, n, ns, ne, nb, nre, nrb, range_min, range_max, shrink_factor, stgn_lim):
        self._validate(ns, ne, nb, nre, nrb)
        
        self.n = n
        self.ns = ns
        self.ne = ne
        self.nb = nb
        self.nre = nre
        self.nrb = nrb
        
        self.range_min = range_min
        self.range_max = range_max
        
        self.shrink_factor = shrink_factor
        self.stgn_lim = stgn_lim
        
        self.scout = None
        self.flower_patch = None
        self.initial_nght = 3.0
        
        self.best_sites = []
        
    def _validate(self, ns, ne, nb, nre, nrb):
        pass
    
    
    def visualize(self, t, res=100):
        if self.n != 2:
            return
        
        # visualisation
        x = np.linspace(self.range_min, self.range_max, res)
        y = np.linspace(self.range_min, self.range_max, res)
        
        X, Y = np.meshgrid(x, y)
        XY = np.array((X, Y)).T
        Z = np.zeros(XY.shape[:-1])
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = self.score_function(XY[i,j])
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([self.range_min, self.range_max])
        ax.set_ylim([self.range_min, self.range_max])       
        ax.contour(X, Y, Z, colors='black');
        
        s = np.array(self.scout)
        ax.scatter(s[:self.ne].T[0], s[:self.ne].T[1], c='yellow', edgecolors='black')
        ax.scatter(s[self.ne:].T[0], s[self.ne:].T[1], c='black')
        
        for i in range(self.ns):
            nght = self.flower_patch[i]['nght']
            sx, sy = self.scout[i]
            ax.add_patch(patches.Rectangle((sx-nght, sy-nght), 2*nght, 2*nght, fill=False)) 
        
        plt.show()
        plt.close()
        
    
    def search(self, score_function, T, k=1, minimizing=False):
        self.scout = [None] * self.ns
        self.flower_patch = [None] * self.ns
        self.nght = [self.initial_nght] * self.ns
        
        self.score_function = score_function if not minimizing else lambda x: -score_function(x)
        
        for i in range(self.ns):
            self.initialize_flower_patch(i)
        
        for t in range(T):
            self.visualize(t)
            self.waggle_dance()
            
            for i in range(self.nb):
                self.local_search(i)
                self.abandon_sites(i)
                self.shrink_neighborhood(i)
                
            for i in range(self.nb, self.ns):
                self.global_search(i)
                
        return self.get_best_solutions(k)
        
    
    def initialize_flower_patch(self, i):
        self.scout[i] = (self.create_random_scout())
        self.flower_patch[i] = {'foragers': 0, 'nght': self.initial_nght, 'stagnation': False,'stagnation_cnt': 0}
        
        
    def create_random_scout(self):
        return np.random.uniform(self.range_min, self.range_max, self.n)
    
    
    def create_random_forager(self, i):
        forager = np.zeros_like(self.scout[i])
        for j in range(self.n):
            nght = self.flower_patch[i]['nght']
            forager[j] = np.clip(np.random.uniform(-1, 1) * nght + self.scout[i][j], self.range_min, self.range_max)
        return forager
    
        
    def waggle_dance(self):
        sorted_idxs = np.argsort([self.score_function(s) for s in self.scout])
        sorted_idxs = list(reversed(sorted_idxs))
        self.scout = list(np.array(self.scout)[sorted_idxs])
        self.flower_patch = list(np.array(self.flower_patch)[sorted_idxs])
        
        for i in range(self.ne):
            self.flower_patch[i]['foragers'] = self.nre
        for i in range(self.ne, self.nb):
            self.flower_patch[i]['foragers'] = self.nrb
        
    
    def local_search(self, i):
        for j in range(self.flower_patch[i]['foragers']):
            forager = self.create_random_forager(i)
            forager_score = self.score_function(forager)
            if forager_score > self.score_function(self.scout[i]):
                self.scout[i] = forager
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
                self.best_sites.append(self.scout[i])
                self.initialize_flower_patch(i)
                
                
    def get_best_solutions(self, k=1):
        self.best_sites.sort(key=self.score_function, reverse=True)
        best_k_sites = self.best_sites[:k]
        return best_k_sites, list(map(self.score_function, best_k_sites))
    
    
        
if __name__ == '__main__':
    
    rosenbrock_fct = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 
    sphere_fct = lambda x: x[0]**2 + x[1]**2
    rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])
    himmelblau_fct = lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
    
    ba = BeesAlgorithm(n=2, ns=300, ne=20, nb=20, nre=30, nrb=5, range_min=-5, range_max=5, shrink_factor=0.8, stgn_lim=10)
    solutions = ba.search(rastrigin_fct, T=1000, minimizing=True)
    print(solutions)
    
    
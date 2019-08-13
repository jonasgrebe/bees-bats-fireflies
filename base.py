from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

class BaseSearchAlgorithm():
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.objective = None
        self.objective_fct = None

        self.solutions = []
        self.history = []
        self.memory = []
        self.params = kwargs

        self.n = self.params['n']
        self.d = self.params['d']
        self.range_min = self.params['range_min']
        self.range_max = self.params['range_max']


    def get_best_solution(self):
        candidates = list(self.solutions)# + list(self.memory)
        if self.objective == 'min':
            return min(candidates, key=self.objective_fct)
        elif self.objective == 'max':
            return max(candidates, key=self.objective_fct)


    def compare_objective_value(self, s0, s1):
        v0 = self.objective_fct(s0)
        v1 = self.objective_fct(s1)

        if self.objective == 'min':
            return v0 - v1
        elif self.objective == 'max':
            return v1 - v0
        
        
    def argsort_objective(self):
        if self.objective == 'min':
            return np.argsort([self.objective_fct(s) for s in self.solutions]).ravel()
        elif self.objective == 'max':
            return np.argsort([self.objective_fct(s)for s in self.solutions])[::-1].ravel()
        

    def search(self, objective, objective_fct, T):
        self.objective = objective
        self.objective_fct = objective_fct

        t_start = perf_counter()

        self.initialize()
        self.visualize_search_step()
        for t in range(1, T+1):
            self.execute_search_step()
            self.history.append(self.get_best_solution())
            self.visualize_search_step(t)

        t_end = perf_counter()

        return (self.history[-1], self.objective_fct(self.history[-1])), t_end-t_start


    def plot_history(self):
        plt.plot([self.objective_fct(s) for s in self.history])
        plt.show()


    def initialize(self):
        raise NotImplementedError


    def execute_search_step(self):
        raise NotImplementedError


    def visualize_search_step(self, t=0):
        if self.d != 2:
            return
        
        x = np.linspace(self.range_min, self.range_max, 100)
        y = np.linspace(self.range_min, self.range_max, 100)
        
        X, Y = np.meshgrid(x, y)
        
        XY = np.array((X, Y)).T
        Z = np.zeros(XY.shape[:-1])
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = self.objective_fct(XY[i,j])
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect='equal')
        ax.set_xlim([self.range_min, self.range_max])
        ax.set_ylim([self.range_min, self.range_max])       

        ax.contourf(X, Y, Z, 20, cmap='Greys');
        ax.contour(X, Y, Z, 20, colors='black', linestyles='dotted');


        ax.scatter(self.solutions.T[0], self.solutions.T[1], c='black')
        
        plt.savefig(f"images/{self.name}/{self.name}_{t}.png")
        plt.close()
        
        

from time import perf_counter
import matplotlib.pyplot as plt

class BaseSearchAlgorithm():
    
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.objective = None
        self.objective_fct = None
        
        self.solutions = []
        self.history = []
        self.params = kwargs
        
        self.n = self.params['n']
        self.d = self.params['d']
        self.range_min = self.params['range_min']
        self.range_max = self.params['range_max']
    
    
    def get_best_solution(self):
        if self.objective == 'min':
            return min(self.solutions, key=self.objective_fct)
        elif self.objective == 'max':
            return max(self.solutions, key=self.objective_fct)
            
    def compare_objective_value(self, s0, s1):
        v0 = self.objective_fct(s0)
        v1 = self.objective_fct(s1)
        
        if self.objective == 'min':
            return v0 - v1
        elif self.objective == 'max':
            return v1 - v0
    
    def search(self, objective, objective_fct, T):
        self.objective = objective
        self.objective_fct = objective_fct
        
        t_start = perf_counter()
        
        self.initialize()
        self.visualize_search_step()
        for t in range(T):
            self.execute_search_step()
            self.history.append(self.get_best_solution())
            self.visualize_search_step(t)
            
        t_end = perf_counter()   
        
        return self.history[-1], t_end-t_start
    
    def plot_history(self):
        plt.plot([self.objective_fct(s) for s in self.history])
        plt.show()
    
    
    def initialize(self):
        raise NotImplementedError
        
        
    def execute_search_step(self):
        raise NotImplementedError
        
        
    def visualize_search_step(self, t=0):
        raise NotImplementedError
            
import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter, process_time
import imageio
import os

class BaseSearchAlgorithm():
    
    def __init__(self, name, **kwargs):
        print(name, kwargs)
        self.name = name
        self.objective = None
        self.objective_fct = None

        self.solutions = []
        self.history = []
        self.best_solution = None
        self.params = kwargs

        self.n = self.params['n'] # size of population
        self.d = self.params['d'] # dimensionality of solution space
        
        self.range_min = self.params['range_min'] # lower bound in all dimensions
        self.range_max = self.params['range_max'] # upper bound in all dimensions
        
        if np.isscalar(self.range_min):
            self.range_min = np.repeat(self.range_min, self.d)
            
        if np.isscalar(self.range_max):
            self.range_max = np.repeat(self.range_max, self.d)
            
        self.evaluation_count = 0
            
        
    def constraints_satisfied(self):
        return True
    

    def get_best_solution(self, key=None):
        if not key:
            key = self.objective_fct
        
        if self.objective == 'min':
            candidate = min(self.solutions, key=key)
        elif self.objective == 'max':
            candidate = max(self.solutions, key=key)
            
        if self.best_solution is None or self.compare_objective_value(candidate, self.best_solution) < 0:
            self.best_solution = np.copy(candidate)
        
        return self.best_solution


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
    
    
    def evaluation_count_decorator(self, f, x):
        self.evaluation_count += 1
        return f(x)
    
    
    def random_uniform_in_ranges(self):
        rnd = np.zeros(self.d)
        for i in range(self.d):
            rnd[i] = np.random.uniform(self.range_min[i], self.range_max[i])
        return rnd
    
    def clip_to_ranges(self, x):
        for i in range(self.d):
            x[i] = np.clip(x[i], self.range_min[i], self.range_max[i])
        return x
    
    
    def search(self, objective, objective_fct, T, visualize=False):
        
        if not self.constraints_satisfied():
            return (np.nan, np.nan), np.nan
        
        self.objective = objective
        self.evaluation_count = 0
        self.objective_fct = lambda x: self.evaluation_count_decorator(objective_fct, x)
        self.history = np.zeros((T, self.d))
        self.best_solution = self.random_uniform_in_ranges()
        self.initialize()
        
        t_start = process_time()
        

        if visualize:
            self.visualize_search_step()
            
        for t in range(T):
            self.execute_search_step(t)
            self.history[t] = self.get_best_solution() 

            if visualize:
                self.visualize_search_step(t+1)

        t_end = process_time()

        return (self.best_solution, self.objective_fct(self.best_solution)), t_end-t_start


    def plot_history(self):
        plt.plot([self.objective_fct(s) for s in self.history])
        plt.xlabel('iteration')
        plt.ylabel('objective function value')
        plt.show()


    def initialize(self):
        raise NotImplementedError


    def execute_search_step(self, t):
        raise NotImplementedError


    def visualize_search_step(self, t=0):
        if self.d != 2:
            return
        
        range_min = np.max(self.range_min)
        range_max = np.max(self.range_max)
        
        x = np.linspace(range_min, range_max, 100)
        y = np.linspace(range_min, range_max, 100)
        
        X, Y = np.meshgrid(x, y)
        
        XY = np.array((X, Y)).T
        Z = np.zeros(XY.shape[:-1])
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i,j] = self.objective_fct(XY[i,j])
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, aspect='equal')
        
        
        ax.set_xlim([range_min, range_max])
        ax.set_ylim([range_min, range_max])       

        ax.contourf(X, Y, Z, 20, cmap='Greys');
        ax.contour(X, Y, Z, 20, colors='black', linestyles='dotted');
        
        ax.scatter(self.solutions.T[0], self.solutions.T[1], marker='.', c='black')
        ax.scatter(self.best_solution[0], self.best_solution[1], marker='X', s=100, c='red')
        
        savepath = f"images/{self.name}/"
        
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        
        plt.savefig(os.path.join(savepath, f"{self.name}_{t}.png"))
        plt.close()
        
        
    def generate_gif(self, filename=None):
        if not filename:
            filename = self.name
        
        plot_paths = [f"images/{self.name}/"+file for file in os.listdir(f"images/{self.name}")]
        
        with imageio.get_writer(f"{filename}.gif", mode='I') as writer:
            for filepath in plot_paths:
                image = imageio.imread(filepath)
                writer.append_data(image)
                
        
        
        

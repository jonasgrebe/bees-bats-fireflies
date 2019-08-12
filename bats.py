import numpy as np

class BatAlgorithm():
    def __init__(self, d, np, a, r, qmin, qmax, range_min, range_max):
        self.d = d # dimensionality of solution-space
        self.np = np # population size
        self.a = a  # loudness
        self.r = r  # pulse rate
        self.qmin = qmin  # frequency min
        self.qmax = qmax  # frequency max
        self.range_min = range_min
        self.range_max = range_max
        

    def init_bats(self):
        self.bats = np.random.uniform(self.range_min, self.range_max, (self.np, self.d))
        self.q = np.zeros(self.np)
        self.v = np.zeros((self.np, self.d))
        self.best = self.bats[np.argmax([self.score_function(s) for s in self.bats])]
        self.f_max = self.score_function(self.best) 
        self.b = np.zeros((self.np, self.d))
        

    def search(self, G, score_function, minimizing=True):
        self.score_function = score_function if not minimizing else lambda x: -score_function(x)
        self.init_bats()

        for t in range(G):
            for i in range(self.np):
                self.q[i] = np.random.uniform(self.qmin, self.qmax)
                for j in range(self.d):
                    self.v[i][j] = self.v[i][j] + (self.bats[i][j] - self.best[j]) * self.q[i]
                    self.b[i][j] = self.bats[i][j] + self.v[i][j]
                    self.b[i][j] = np.clip(self.b[i][j], self.range_min, self.range_max)

                if np.random.uniform(0, 1) > self.r:
                    for j in range(self.d):
                        self.b[i][j] = self.best[j] + 10e-3 * np.random.normal(0, 1)
                        self.b[i][j] = np.clip(self.b[i][j], self.range_min, self.range_max)
                        
                f_new = self.score_function(self.b[i])

                if (f_new >= self.score_function(self.bats[i])) and (np.random.uniform(0, 1) < self.a):
                    self.bats[i] = self.b[i]

                if f_new >= self.f_max:
                    self.best = self.b[i]
                    self.f_max = f_new

        print(self.best)
        print(self.f_max)
        
        
if __name__ == '__main__':
    
    rosenbrock_fct = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 
    sphere_fct = lambda x: x[0]**2 + x[1]**2
    rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])
    himmelblau_fct = lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
    
    bat = BatAlgorithm(2, 200, 0.5, 0.5, 0.0, 3.0, -5.0, 5.0)
    bat.search(G=2000, score_function=rosenbrock_fct)
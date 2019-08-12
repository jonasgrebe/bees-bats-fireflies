from bats2 import BatAlgorithm
from bees2 import BeesAlgorithm
import numpy as np

rosenbrock_fct = lambda x: 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 
sphere_fct = lambda x: x[0]**2 + x[1]**2
rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])
himmelblau_fct = lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

    
bat = BatAlgorithm(d=2, np=50, range_min=-5.0, range_max=5.0,
                   a=0.5, r=0.5, qmin=0.0, qmax=3.0)
solutions = bat.search(sphere_fct, 50, k=1, visualize=True, minimize=True)
print(solutions)


bees = BeesAlgorithm(d=2, np=50, range_min=-5, range_max=5, ne=20,
                     nb=10, nre=20, nrb=2, shrink_factor=0.8, stgn_lim=3)
solutions = bees.search(sphere_fct, 50, k=1, visualize=True, minimize=True)
print(solutions)
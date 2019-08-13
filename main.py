from bat import BatAlgorithm
from bees import BeesAlgorithm
from firefly import FireflyAlgorithm
import numpy as np

rosenbrock_fct = lambda x: sum([100*(x[i+1]-x[i])**2+(1-x[i])**2 for i in range(len(x)-1)])
sphere_fct = lambda x: x[0]**2 + x[1]**2
rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])
himmelblau_fct = lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
beale_fct = lambda x: (1.3-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]+x[1]**3)**2

objective_fct = himmelblau_fct
d = 2


bat = BatAlgorithm(d=d, n=100, range_min=-5.0, range_max=5.0,
                   a=0.5, r=0.5, q_min=0.0, q_max=3.0)

bees = BeesAlgorithm(d=d, n=100, range_min=-5.0, range_max=5.0,
                     ne=40, nb=10, nre=10, nrb=5, shrink_factor=0.8, stgn_lim=50)

firefly = FireflyAlgorithm(d=d, n=100, range_min=-5.0, range_max=5.0,
                           alpha=1.0, beta0=1.0, gamma=0.5)


solution, _ = bat.search('min', objective_fct, 50)
bat.plot_history()
print(solution)


solution, _ = bees.search('min', objective_fct, 50)
bees.plot_history()
print(solution)


solution, _ = firefly.search('min', objective_fct, 50)
firefly.plot_history()
print(solution)



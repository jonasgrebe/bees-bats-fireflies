from bat import BatAlgorithm
from bees import BeesAlgorithm
from firefly import FireflyAlgorithm
import numpy as np

rosenbrock_fct = lambda x: sum([100*(x[i+1]-x[i])**2+(1-x[i])**2 for i in range(len(x)-1)])
sphere_fct = lambda x: x[0]**2 + x[1]**2
rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])
himmelblau_fct = lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
beale_fct = lambda x: (1.3-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]+x[1]**3)**2

objective_fct = rastrigin_fct
d = 2
n = 100
range_min, range_max = -10.0, 10.0
T = 100


bat = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                   a=0.5, r=0.5, q_min=0.0, q_max=3.0)

bees = BeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     nb=50, ne=20, nrb=5, nre=10, shrink_factor=0.8, stgn_lim=5)

firefly = FireflyAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                           alpha=1.0, beta0=1.0, gamma=0.5)


solution, latency = bat.search('min', objective_fct, T, visualize=True)
bat.plot_history()
bat.generate_gif()
print(solution, latency)


solution, latency = bees.search('min', objective_fct, T, visualize=True)
bees.plot_history()
bees.generate_gif()
print(solution, latency)


solution, latency = firefly.search('min', objective_fct, T, visualize=True)
firefly.plot_history()
firefly.generate_gif()
print(solution, latency)



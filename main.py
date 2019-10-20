from metaheuristics.bat import BatAlgorithm
from metaheuristics.bees import BeesAlgorithm
from metaheuristics.firefly import FireflyAlgorithm
import numpy as np

rosenbrock_fct = lambda x: sum([100*(x[i+1]-x[i])**2+(1-x[i])**2 for i in range(len(x)-1)])
sphere_fct = lambda x: x[0]**2 + x[1]**2
rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])
himmelblau_fct = lambda x: (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2
beale_fct = lambda x: (1.3-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]+x[1]**3)**2

test_fct = lambda x: np.prod([np.sin(x[i]) for i in range(len(x))])
alpine_fct = lambda x: np.prod([np.sqrt(x[i]) * np.sin(x[i]) for i in range(len(x))]) - 2.808**len(x)
brown_fct = lambda x: sum([(x[i]**2)**(x[i+1]**2+1) + (x[i+1]**2)**(x[i]**2+1) for i in range(len(x)-1)])    
csendes_fct = lambda x: sum([x[i]**6 * (2 + np.sin(1/x[i]))for i in range(len(x))])
griewank_fct = lambda x: sum([x[i]*2/4000 for i in range(len(x))]) - np.prod([np.cos(x[i]/np.sqrt(i)) for i in range(len(x))]) + 1
step_fct = lambda x: sum([np.trunc(x[i]**2) for i in range(len(x))])
zakhariv_fct = lambda x: sum([x[i]**2 for i in range(len(x))]) + 0.5*(sum([x[i]*i for i in range(len(x))]))**2+  0.5*(sum([x[i]*i for i in range(len(x))]))**4
qing_fct = lambda x: sum([(x[i]**2-i-1)**2 for i in range(len(x))])


def n_queens_cost(x):
    x = np.round(x)
    loss = sum([(x==i).sum()-1 for i in set(x)])
    loss += sum(np.clip([(x-np.arange(len(x))==i).sum()-1 for i in set(x-np.arange(len(x)))], 0, np.inf))
    loss += sum(np.clip([(x+np.arange(len(x))==i).sum()-1 for i in set(x+np.arange(len(x)))], 0, np.inf))
    return loss


objective = 'min'
objective_fct = sphere_fct
d = 4
n = 250
range_min, range_max = [-5]*d, [5]*d
T = 100

bees = BeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     nb=int(0.3*n), ne=int(0.1*n), nrb=int(0.1*n), nre=int(0.2*n))

firefly = FireflyAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                           alpha=0.5, beta_max=1.0, gamma=0.5)

bat = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
       a=1.0, r_min=0.5, r_max=1.0, alpha=0.99, gamma=0.9, f_min=1.0, f_max=5.0)

#%%
solution, latency = bees.search(objective, objective_fct, T, visualize=True)
bees.plot_history()
bees.generate_gif()
print(solution, latency, bees.evaluation_count)

# %%

solution, latency = bat.search(objective, objective_fct, T, visualize=True)
bat.plot_history()
bat.generate_gif()
print(solution, latency, bat.evaluation_count)
    
# %%

solution, latency = firefly.search(objective, objective_fct, T, visualize=True)
firefly.plot_history()
firefly.generate_gif()
print(solution, latency, firefly.evaluation_count)



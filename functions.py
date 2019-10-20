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
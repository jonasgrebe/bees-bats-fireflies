import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# ================== PROBLEM SPECIFICATION ====================================
# =============================================================================

# -------------------- unconstrained objective function -----------------------

def weight(x):
    return (x[2]+2)*x[1]*x[0]**2

# ----------------------------- constraints  ----------------------------------

def constraint_1(x):
    return 71785*x[0]**4 <= x[1]**3*x[2]

def constraint_2(x):
    a = 4*x[1]**2 - x[0]*x[1]
    a /= 12566 * (x[1]*x[0]**3 - x[0]**4)
    
    b = 1
    b /= 5108 * x[0]**2
    
    return a + b <= 1

def constraint_3(x):    
    return x[1]**2*x[2] <= 140.45*x[0]

def constraint_4(x):
    return x[1]+x[0] <= 1.5


constraints = [constraint_1, constraint_2, constraint_3, constraint_4]

# --------------------- constrained objective function ------------------------
    
def barrier_fct(x):
    for constraint in constraints:
        if not constraint(x):
            return np.infty
    return 0


global_minimum = np.array([0.05044713178541634, 0.32746441361099429, 13.23998350856038107])

# =============================================================================
# ================== PROBLEM SPECIFICATION ====================================
# =============================================================================

objective = 'min'
objective_fct = lambda x: weight(x) + barrier_fct(x)

from metaheuristics.random import RandomSamplingAlgorithm
from metaheuristics.bees import BeesAlgorithm, ImprovedBeesAlgorithm
from metaheuristics.bat import BatAlgorithm
from metaheuristics.firefly import FireflyAlgorithm

# ------------------- algorithm-independent parameters ------------------------

d = 3
n = 50
range_min, range_max = (0.05, 0.25, 2.0), (2.0, 1.3, 15.0)
T = 100
R = 20

rand = RandomSamplingAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max)

bees = BeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     nb=20, ne=10, nrb=10, nre=30)#, sf=0.9, sl=5)

bat = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                   a=10.0, r_min=0.1, r_max=1.0, alpha=0.9, gamma=0.9, f_min=0.0, f_max=1.0)

firefly = FireflyAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                           alpha=0.5, beta_max=1.0, gamma=1.0)



for i, algo in enumerate([bat]):
    
    x_list = []
    y_list = []
    latency_list = []
    history_list = []
    for _ in range(R):

        solution, latency = algo.search(objective, objective_fct, T, visualize=False)
        x_list.append(solution[0])
        y_list.append(solution[1])
        history_list.append(algo.history)
        latency_list.append(latency)
    
    print("====== ALGORITHM ======================", algo.name, i)
    
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    
    non_inf = np.where(y_list != np.inf)
    
    x_list = x_list[non_inf]
    y_list = y_list[non_inf]
    
    best = np.nanargmin(y_list, axis=0)
    worst = np.nanargmax(y_list, axis=0)
    
    print("best", x_list[best], y_list[best])
    print("worst", x_list[worst], y_list[worst])
    
    print("mean", np.nanmean(y_list, axis=0))
    print("latency", np.nanmean(latency_list, axis=0))
    print("std", np.nanstd(y_list, axis=0))
    
    print("Global optimum", objective_fct(global_minimum))
    
    history_list = np.array(history_list)
    
    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('spring weight')
    for h in history_list[non_inf]:
        h = [objective_fct(x) for x in h]
        plt.plot(h)
    plt.hlines(objective_fct(global_minimum), 0, T)
    plt.ylim((0, 0.15))
    plt.show()
    
    print("======================================")
    





    
    
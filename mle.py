import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy

x0 = 42
gamma = 7

global_maximum = np.array([x0, gamma])


samples = cauchy.rvs(loc=x0, scale=gamma, size=10000)


def log_likelihood(x):
    x0, gamma = x[0], x[1]
    return -len(samples) * np.log(gamma*np.pi)- np.sum(np.log(1+np.square((samples-x0)/gamma)))


objective = 'max'
objective_fct = log_likelihood

d = 2
n = 50
range_min = (0, 0)
range_max = (100, 100)

T = 100
R = 10

#%%

from metaheuristics.random import RandomSamplingAlgorithm
from metaheuristics.bees import ImprovedBeesAlgorithm
from metaheuristics.bat import BatAlgorithm
from metaheuristics.firefly import FireflyAlgorithm

rand = RandomSamplingAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max)

# normal
bees_1 = ImprovedBeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                       nb=10, ne=5, nrb=5, nre=10, sf=0.99, sl=5)

# heavy exploration
bees_2 = ImprovedBeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                       nb=5, ne=0, nrb=5, nre=0, sf=1.0, sl=T)

# heavy exploitation
bees_3 = ImprovedBeesAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                       nb=n, ne=int(0.5*n), nrb=10, nre=30, sf=0.99, sl=5)


bat_1 = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     a=1.0, r_min=0.1, r_max=0.2, alpha=0.9, gamma=0.9, f_min=1.0, f_max=3.0)

bat_2 = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     a=1.0, r_min=0.5, r_max=1.0, alpha=0.99, gamma=0.9, f_min=1.0, f_max=5.0)

bat_3 = BatAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                     a=5.0, r_min=0.2, r_max=0.7, alpha=0.9, gamma=0.9, f_min=3.0, f_max=5.0)


firefly_1 = FireflyAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                           alpha=0.5, beta_max=1.0, gamma=0.25)

firefly_2 = FireflyAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                           alpha=0.9, beta_max=0.5, gamma=0.5)

firefly_3 = FireflyAlgorithm(d=d, n=n, range_min=range_min, range_max=range_max,
                           alpha=2.0, beta_max=2.0, gamma=0.75)


for i, algo in enumerate([rand, bees_1, bees_2, bees_3, bat_1, bat_2, bat_3, firefly_1, firefly_2, firefly_3]):
    
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
    
    best = np.nanargmax(y_list, axis=0)
    worst = np.nanargmin(y_list, axis=0)
    
    print("best", x_list[best], y_list[best])
    print("worst", x_list[worst], y_list[worst])
    
    print("mean", np.nanmean(x_list, axis=0))
    print("latency", np.nanmean(latency_list, axis=0))
    print("std", np.nanstd(x_list, axis=0))
    
    print("Global optimum", objective_fct(global_maximum))
    
    history_list = np.array(history_list)
    
    plt.figure()
    plt.xlabel('iteration')
    plt.ylabel('log likelihood')
    for h in history_list[non_inf]:
        h = [objective_fct(x) for x in h]
        plt.plot(h)
    plt.hlines(objective_fct(global_maximum), 0, T)
    plt.show()
    
    print("======================================")
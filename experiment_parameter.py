import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle

from time import perf_counter

matplotlib.rcParams.update({'font.size': 12})

def grid_search(algo, objective, objective_fct, T, repetitions, axis1_values, axis2_values, relative_to=False, except_for=[], store_dir='', **kwargs):
    orig_kwargs = kwargs.copy()
    axis_kwargs = []

    for kwarg in kwargs:
        if kwargs[kwarg] is None:
            axis_kwargs.append(kwarg)

    if len(axis_kwargs) != 2:
        raise Exception

    if relative_to:
        if relative_to in axis_kwargs and axis_kwargs[1] == relative_to:
            axis_kwargs[0], axis_kwargs[1] = axis_kwargs[1], axis_kwargs[0]
            axis1_values, axis2_values = axis2_values, axis1_values

    solution_y_grid = np.zeros((len(axis1_values), len(axis2_values)))
    solution_std_grid = np.zeros((len(axis1_values), len(axis2_values)))
    latency_grid = np.zeros((len(axis1_values), len(axis2_values)))

    print(axis1_values)
    print(axis2_values)

    for i, value1 in enumerate(axis1_values):
        for j, value2 in enumerate(axis2_values):
            print(f"{i}/{len(axis1_values)-1} - {j}/{len(axis2_values)-1}")

            kwargs = orig_kwargs.copy()
            kwargs[axis_kwargs[0]] = value1
            kwargs[axis_kwargs[1]] = value2

            if relative_to:
                for kwarg in kwargs:
                    if kwarg not in except_for and kwarg != relative_to:
                        kwargs[kwarg] = int(kwargs[relative_to] * kwargs[kwarg])

            a = algo(**kwargs)

            r_solution_y_list = []
            r_latency_list = []

            for r in range(repetitions):
                solution, latency = a.search(objective=objective, objective_fct=objective_fct, T=T)
    
                r_solution_y_list.append(solution[1])
                r_latency_list.append(latency)

            solution_y_grid[i, j] = np.mean(r_solution_y_list)
            solution_std_grid[i, j] = np.std(r_solution_y_list)
            latency_grid[i, j] = np.mean(r_latency_list)
            
                

    def plot_grid_figure(matrix, metric_name):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix, interpolation='nearest')
        fig.colorbar(cax)

        x_ticklabels = ['']+axis2_values
        x_ticklabels[1] = axis_kwargs[1] + '=' + str(x_ticklabels[1])

        y_ticklabels = ['']+axis1_values
        y_ticklabels[1] = axis_kwargs[0] + '=' + str(y_ticklabels[1])

        ax.set_xticklabels(x_ticklabels)
        ax.set_yticklabels(y_ticklabels)

        store_path = os.path.join(store_dir, str(axis_kwargs[0])+'_'+str(axis_kwargs[1])+'_grid_'+fct_name+'_'+algo.__name__+'_'+metric_name)

        plt.savefig(store_path+'.png', dpi=300)
        np.savetxt(store_path+'.txt', matrix, delimiter=',')
        print(f"{metric_name} extracted")

    with open(os.path.join(store_dir, 'axes_info.pkl'), 'wb') as f:
        pickle.dump({'x_kwarg': axis_kwargs[1], 'x_values': axis2_values,
                     'y_kwarg': axis_kwargs[0], 'y_values': axis1_values}, f)

    plot_grid_figure(solution_y_grid, 'solution_mean')
    plot_grid_figure(solution_std_grid, 'solution_std')
    plot_grid_figure(latency_grid, 'latency')

from metaheuristics.bees import BeesAlgorithm, ImprovedBeesAlgorithm
from metaheuristics.bat import BatAlgorithm
from metaheuristics.firefly import FireflyAlgorithm

sphere_fct = lambda x: sum([x[i]**2 for i in range(len(x))])
rosenbrock_fct = lambda x: sum([100*(x[i+1]-x[i])**2+(1-x[i])**2 for i in range(len(x)-1)]) if len(x)>1 else np.nan
rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])
qing_fct = lambda x: sum([(x[i]**2-i-1)**2 for i in range(len(x))])

alpine_fct = lambda x: sum(abs(x * np.sin(x) + 0.1 * x));

def get_ranges(fct_name):
    if fct_name == 'sphere':
        return -10.0, 10.0
    if fct_name == 'rosenbrock':
        return -5.0, 5.0
    if fct_name == 'rastrigin':
        return -5.12, 5.12
    if fct_name == 'alpine':
        return -10, 10
    if fct_name == 'qing':
        return -10, 10


# experiment hyperparameters
T = 150
repetitions = 50

#%%
for fct, fct_name in zip([sphere_fct, rosenbrock_fct, rastrigin_fct], ['sphere', 'rosenbrock', 'rastrigin']):

    range_min, range_max = get_ranges(fct_name)

#    grid_search(BeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
#                axis1_values=[1, 2, 3, 4, 5, 6, 7, 8],
#                axis2_values=[10, 20, 30, 40, 50, 75, 100, 250],
#                d=None, n=None, nb=0.3, ne=0.1, nrb=0.1, nre=0.2, range_min=range_min, range_max=range_max,
#                store_dir=f'images/experiments/exp_bees/exp1/', relative_to='n', except_for=['d', 'range_min','range_max'])

    # experiment 2:
    grid_search(BeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                axis2_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                d=2, n=100, nb=None, ne=None, nrb=0.1, nre=0.2, range_min=range_min, range_max=range_max,
                store_dir=f'images/experiments/exp_bees/exp2/', relative_to='n', except_for=['d', 'range_min','range_max'])

#    # experiment 3:
#    grid_search(BeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
#                axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
#                axis2_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
#                d=2, n=100, nb=0.3, ne=0.1, nrb=None, nre=None, range_min=range_min, range_max=range_max,
#                store_dir=f'images/experiments/exp_bees/exp3/', relative_to='n', except_for=['d', 'range_min','range_max'])

    # experiment 4:
    grid_search(ImprovedBeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                axis2_values=[1, 2, 3, 4, 5, 10, 15, 20],
                d=2, n=100, nb=0.3, ne=0.1, nrb=0.1, nre=0.2, sf=None, sl=None, range_min=range_min, range_max=range_max,
                store_dir=f'images/experiments/exp_bees/exp4/', relative_to='n', except_for=['d', 'sl', 'range_min','range_max'])

#%%
for fct, fct_name in zip([sphere_fct, rosenbrock_fct, rastrigin_fct], ['sphere', 'rosenbrock', 'rastrigin']):

    range_min, range_max = get_ranges(fct_name)
    
    #BATS ALGORITHM EXPERIMENTS
    # experiment 1:
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                 axis2_values=[1, 2, 3, 4, 5, 6, 7, 8],
                 axis1_values=[10, 20, 30, 40, 50, 75, 100, 250],
                 n=None, d=None, a=1.0, r_max=0.1, alpha=0.9, gamma=0.9, f_min=0.0, f_max=5.0, range_min=range_min, range_max=range_max,
                 store_dir=f'images/experiments/exp_bats/exp1/', relative_to=None)

    # experiment 2:f
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                 axis1_values=[0, 0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0],
                 axis2_values=[0.0, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0],
                 d=2, n=100, a=None, r_max=0.1, alpha=None, gamma=0.9, f_min=0.0, f_max=5.0, range_min=range_min, range_max=range_max,
                 store_dir=f'images/experiments/exp_bats/exp2/', relative_to=None)

    # experiment 3:
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                 axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                 axis2_values=[0.0, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0],
                 d=2, n=100, a=1.0, r_max=None, alpha=0.9, gamma=None, f_min=0.0, f_max=5.0, range_min=range_min, range_max=range_max,
                 store_dir=f'images/experiments/exp_bats/exp3/', relative_to=None)

    # experiment 4:
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                 axis1_values=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
                 axis2_values=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0],
                 d=2, n=100, a=1.0, r_max=0.1, alpha=0.9, gamma=0.9, f_min=None, f_max=None, range_min=range_min, range_max=range_max,
                 store_dir=f'images/experiments/exp_bats/exp4/', relative_to=None)

#%%
    
R = 10
T = 20

for fct, fct_name in zip([sphere_fct, rosenbrock_fct, rastrigin_fct], ['sphere', 'rosenbrock', 'rastrigin']):

    range_min, range_max = get_ranges(fct_name)
    
    # FIREFLY ALGORITHM EXPERIMENTS
    # experiment 1:
    grid_search(FireflyAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                 axis2_values=[1, 2, 3, 4, 5, 6, 7, 8],
                 axis1_values=[10, 20, 30, 40, 50, 75, 100, 250],
                 n=None, d=None, alpha=0.9, beta_max=0.9, gamma=10, range_min=range_min, range_max=range_max,
                 store_dir=f'images/experiments/exp_fireflies/exp1/', relative_to=None)
    
    # experiment 2:
    grid_search(FireflyAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                 axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                 axis2_values=[0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                 d=2, n=100, alpha=None, beta_max=None, gamma=10, range_min=range_min, range_max=range_max,
                 store_dir=f'images/experiments/exp_fireflies/exp2/', relative_to=None)
    
    # experiment 3:
    grid_search(FireflyAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                 axis1_values=[1, 2, 3, 4, 5, 6, 7, 8],
                 axis2_values=[0.0, 0.25, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0],
                 d=None, n=100, alpha=0.9, beta_max=0.9, gamma=None, range_min=range_min, range_max=range_max,
                 store_dir=f'images/experiments/exp_fireflies/exp3/', relative_to=None)

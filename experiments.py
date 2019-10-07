import numpy as np
import matplotlib.pyplot as plt

def grid_search(algo, objective, objective_fct, T, repetitions, axis1_values, axis2_values, global_optimas=None, relative_to=False, except_for=[], store_prefix='', **kwargs):
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
    latency_grid = np.zeros((len(axis1_values), len(axis2_values)))
    # convergence_grid = np.zeros((len(axis2_values), len(axis1_values)))    
    if global_optimas:
        location_error_grid = np.zeros((len(axis1_values), len(axis2_values)))
        
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
            # r_convergence_list = []
            
            if global_optimas:
                r_location_error_list = []
            
            for r in range(repetitions):
                solution, latency = a.search(objective=objective, objective_fct=objective_fct, T=T)
                r_solution_y_list.append(solution[1])
                r_latency_list.append(latency)
                
#                y_history = [objective_fct(x) for x in a.history]
#                y_variation = y_history[-1] - y_history[0]
#                
#                epsilon = 0.05
#                convergence_border = y_variation * epsilon
#                
#                convergence_iteration = np.searchsorted(y_history, solution[1] - convergence_border)
#
#                r_convergence_list.append(convergence_iteration)
                
                if global_optimas:
                    d = orig_kwargs['d'] if orig_kwargs['d'] else kwargs['d']
                    global_optima = global_optimas[0] if orig_kwargs['d'] else global_optimas[kwargs['d']-1]
                    r_location_error_list.append(np.sum(np.abs(solution[0]-global_optima)) / d) # divide by dimensionality to get average error per dimension
                
            solution_y_grid[i, j] = np.mean(r_solution_y_list)
            latency_grid[i, j] = np.mean(r_latency_list)
            # convergence_grid[i,j] = np.mean(r_convergence_list)
            
            if global_optimas:
                location_error_grid[i, j] = np.mean(r_location_error_list)
    
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
        
        plt.savefig(store_prefix+algo.__name__+'_'+metric_name+'.png', dpi=300)
        np.savetxt(store_prefix+algo.__name__+'_'+metric_name+'.txt', matrix, delimiter=',')
    
    plot_grid_figure(solution_y_grid, 'solution_y')
    plot_grid_figure(latency_grid, 'latency')
    # plot_grid_figure(convergence_grid, 'convergence')    
    
    if global_optimas:
        plot_grid_figure(location_error_grid, 'location_error')
    

from bees import BeesAlgorithm, ImprovedBeesAlgorithm
from bat import BatAlgorithm
from firefly import FireflyAlgorithm

sphere_fct = lambda x: sum([x[i]**2 for i in range(len(x))])
rosenbrock_fct = lambda x: sum([100*(x[i+1]-x[i])**2+(1-x[i])**2 for i in range(len(x)-1)]) if len(x)>1 else np.nan
rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])

def get_ranges(fct_name):
    if fct_name == 'sphere':
        return -10.0, 10.0
    if fct_name == 'rosenbrock':
        return -10.0, 10.0
    if fct_name == 'rastrigin':
        return -5.12, 5.12

def get_global_optima(fct_name, d):
    if fct_name == 'sphere':
        return np.zeros(d)
    if fct_name == 'rosenbrock':
        return np.ones(d) if d>1 else np.nan
    if fct_name == 'rastrigin':
        return np.zeros(d)

# experiment hyperparameters
T = 100
repetitions = 100


for fct, fct_name in zip([sphere_fct, rosenbrock_fct, rastrigin_fct], ['sphere', 'rosenbrock', 'rastrigin']):
    
    range_min, range_max = get_ranges(fct_name)
    
    grid_search(BeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[1, 2, 3, 4, 5, 6, 7, 8],
                axis2_values=[10, 20, 30, 40, 50, 100, 250, 500],
                global_optimas=[get_global_optima(fct_name, d) for d in [1, 2, 3, 4, 5, 6, 7, 8]],
                d=None, n=None, nb=0.3, ne=0.1, nrb=0.1, nre=0.2, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bees/exp1/n_d_grid_{fct_name}_', relative_to='n', except_for=['d', 'range_min','range_max'])
    
    # experiment 2:
    grid_search(BeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                axis2_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                global_optimas=[get_global_optima(fct_name, 2)],
                d=2, n=100, nb=None, ne=None, nrb=0.1, nre=0.2, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bees/exp2/nb_ne_grid_{fct_name}_', relative_to='n', except_for=['d', 'range_min','range_max'])

    # experiment 3:
    grid_search(BeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                axis2_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                global_optimas=[get_global_optima(fct_name, 2)],
                d=2, n=100, nb=0.3, ne=0.1, nrb=None, nre=None, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bees/exp3/nrb_nre_grid_{fct_name}_', relative_to='n', except_for=['d', 'range_min','range_max'])
      
    # experiment 4:
    grid_search(ImprovedBeesAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                axis2_values=[1, 2, 3, 4, 5, 10, 15, 20],
                global_optimas=[get_global_optima(fct_name, 2)],
                d=2, n=100, nb=0.3, ne=0.1, nrb=0.1, nre=0.2, sf=None, sl=None, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bees/exp4/sf_sl_grid_{fct_name}_', relative_to='n', except_for=['d', 'sl', 'range_min','range_max'])
    
    
    # BATS ALGORITHM EXPERIMENTS
    # experiment 1:
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[1, 2, 3, 4, 5, 6, 7, 8],
                axis2_values=[10, 20, 30, 40, 50, 100, 250, 500],
                d=None, n=None, a=1.0, r0=0.1, alpha=0.9, gamma=0.9, q_min=0, q_max=10, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bats/exp1/n_d_grid_{fct_name}_', relative_to=None)
    
    # experiment 2:
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0, 1, 5, 10, 15, 20, 25, 50, 75, 100],
                axis2_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                d=2, n=100, a=None, r0=None, alpha=0.9, gamma=0.9, q_min=0, q_max=10, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bats/exp2/a_r0_grid_{fct_name}_', relative_to=None)
    
    # experiment 3:
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
                axis2_values=[0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
                d=2, n=100, a=1.0, r0=0.1, alpha=None, gamma=None, q_min=0, q_max=10, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bats/exp3/alpha_gamma_grid_{fct_name}_', relative_to=None)
   
    # experiment 4:
    grid_search(BatAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
                axis2_values=[0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
                d=2, n=100, a=1.0, r0=0.1, alpha=0.9, gamma=0.9, q_min=None, q_max=None, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_bats/exp4/q_min_q_max_grid_{fct_name}_', relative_to=None)
    
    
    # FIREFLY ALGORITHM EXPERIMENTS
    # experiment 1:
    grid_search(FireflyAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[1, 2, 3, 4, 5, 6, 7, 8],
                axis2_values=[10, 20, 30, 40, 50, 100, 250, 500],
                d=None, n=None, alpha=0.9, beta0=0.9, gamma=10, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_fireflies/exp1/n_d_grid_{fct_name}_', relative_to=None)
    
    # experiment 2:
    grid_search(FireflyAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                axis2_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                d=2, n=100, alpha=None, beta0=None, gamma=10, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_fireflies/exp2/alpha_beta0_grid_{fct_name}_', relative_to=None)
    
    # experiment 3:
    grid_search(FireflyAlgorithm, 'min', fct, T=T, repetitions=repetitions,
                axis1_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
                axis2_values=[0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
                d=2, n=100, alpha=None, beta0=0.9, gamma=None, range_min=range_min, range_max=range_max,
                store_prefix=f'images/experiments/exp_fireflies/exp3/alpha_beta0_grid_{fct_name}_', relative_to=None)
    

import numpy as np
import matplotlib.pyplot as plt

def grid_search(algo, objective, objective_fct, T, repititions, axis1_values, axis2_values, relative_to=False, except_for=[], store_prefix='', **kwargs):
    
    orig_kwargs = kwargs.copy()
    axis_kwargs = []
    
    for kwarg in kwargs:
        if kwargs[kwarg] is None:
            axis_kwargs.append(kwarg)
            
    if len(axis_kwargs) != 2:
        raise Exception
    
    solution_y_grid = np.zeros((len(axis2_values), len(axis1_values)))
    latency_grid = np.zeros((len(axis2_values), len(axis1_values)))
          
    if relative_to:
        if relative_to not in axis_kwargs:
            raise Exception
        if axis_kwargs[1] == relative_to:
            axis_kwargs[0], axis_kwargs[1] = axis_kwargs[1], axis_kwargs[0]
            axis1_values, axis2_values = axis2_values, axis1_values
        
    for i, value1 in enumerate(axis1_values):
            
        kwargs[axis_kwargs[0]] = value1
        
        if relative_to:
            for kwarg in kwargs:
                if kwarg not in except_for and kwarg not in axis_kwargs:
                    kwargs[kwarg] = int(value1 * orig_kwargs[kwarg])
        
        for j, value2 in enumerate(axis2_values):
            
            kwargs[axis_kwargs[1]] = value2
            
            a = algo(**kwargs)
            
            r_solution_y_list = []
            r_latency_list = []
            for r in range(repititions):
                solution, latency = a.search(objective=objective, objective_fct=objective_fct, T=T)
                r_solution_y_list.append(solution[1])
                r_latency_list.append(latency)
                
            solution_y_grid[i, j] = np.mean(r_solution_y_list)
            latency_grid[i, j] = np.mean(r_latency_list)
    
    fig = plt.figure()
    # fig.suptitle(algo.__name__ + ' - objective')
    ax = fig.add_subplot(111)
    cax = ax.matshow(solution_y_grid, interpolation='nearest')
    fig.colorbar(cax)
    
    x_ticklabels = ['']+axis2_values
    x_ticklabels[1] = axis_kwargs[1] + '=' + str(x_ticklabels[1])
    
    y_ticklabels = ['']+axis1_values
    y_ticklabels[1] = axis_kwargs[0] + '=' + str(y_ticklabels[1])
    
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)
    
    plt.savefig(store_prefix+algo.__name__+'_solution_y.png')
    np.savetxt(store_prefix+algo.__name__+'_solution_y.txt', solution_y_grid, delimiter=',')
    
    fig = plt.figure()
    # fig.suptitle(algo.__name__ + ' - latency')
    ax = fig.add_subplot(111)
    cax = ax.matshow(latency_grid, interpolation='nearest')
    fig.colorbar(cax)
    
    x_ticklabels = ['']+axis2_values
    x_ticklabels[1] = axis_kwargs[1] + '=' + str(x_ticklabels[1])
    
    y_ticklabels = ['']+axis1_values
    y_ticklabels[1] = axis_kwargs[0] + '=' + str(y_ticklabels[1])
    
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)
    
    plt.savefig(store_prefix+algo.__name__+'_latency.png')
    np.savetxt(store_prefix+algo.__name__+'_latency.txt', latency_grid, delimiter=',')
    
from bat import BatAlgorithm
from bees import BeesAlgorithm

sphere_fct = lambda x: sum([x[i]**2 for i in range(len(x))])
rosenbrock_fct = lambda x: sum([100*(x[i+1]-x[i])**2+(1-x[i])**2 for i in range(len(x)-1)])
rastrigin_fct = lambda x: 10*len(x)+sum([x[i]**2-10*np.cos(2*np.pi*x[i]) for i in range(len(x))])

grid_search(BeesAlgorithm, 'min', sphere_fct, T=50, repititions=100,
            axis1_values=[1, 2, 4, 8, 16, 32, 64, 128],
            axis2_values=[10, 15, 20, 25, 50, 100, 150, 200],
            d=None, n=None, nb=0.75, ne=0.25, nrb=0.1, nre=0.2, range_min=-5.0, range_max=5.0,
            store_prefix='images/experiments/n_d_grid_', relative_to='n', except_for=['range_min','range_max'])
    
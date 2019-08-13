# nature-inspired metaheuristics: bees, bats and fireflies

Implementation of three nature-inspired search algorithms:
- Bees Algorithm
- Bat Algorithm
- Firefly Algorithm

## TODO: What are metaheuristics?

## How to use the algorithms?
All three algorithms and their variants share a common interface. Basically, all one needs to do in order to use one of the algorithms for optimization, is invoking the ```search(objective, objective_fct, T)``` method. The parameters all algorithms have in common are the following:
- **objective**: 'min' or 'max', depending on whether it is a minimization or a maximization problem
- **objective_fct**: python-function or lambda, f: R^d -> R
- **d**: dimensionality of solution-space
- **T**: number of iterations

```python
objective = 'min' # minimization or maximation?
objective_fct = lambda x: x[0]**2 + x[1]**2 # function to optimize
d = 2 # dimensionality of solution-space
```

### The Bees Algorithm

```python
bees = BeesAlgorithm()

solution, latency = bees.search(objective, objective_fct, T=50)
bees.plot_history()
```
### The Bat Algorithm

```python
bat = BatAlgorithm()

solution, latency = bat.search(objective, objective_fct, T=50)
bat.plot_history()
```

### The Firefly Algorithm

```python
firefly = FireflyAlgorithm()

solution, latency = firefly.search(objective, objective_fct, T=50)
bat.plot_history()
```

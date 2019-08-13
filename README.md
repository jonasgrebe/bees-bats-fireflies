# nature-inspired metaheuristics: bees, bats and fireflies

Implementation of three nature-inspired search algorithms:
- Bees Algorithm
- Bat Algorithm
- Firefly Algorithm

## What are metaheuristics?


## How to use the algorithms?

```python
objective = 'min' # minimization or maximation?
objective_fct = lambda x: x[0]**2 + x[1]**2 # function to optimize
d = 2 # dimensionality of solution space
```

### The Bees Algorithm

```python
bees = BeeAlgorithm()

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

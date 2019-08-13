# Nature-inspired Metaheuristics: Bees, Bats and Fireflies

Implementation of three nature-inspired search algorithms:
- Bees Algorithm [[Pham et al., 2005](https://www.researchgate.net/publication/260985621_The_Bees_Algorithm_Technical_Note)]
- Bat Algorithm [[Yang, 2010](https://arxiv.org/abs/1004.4170v1)]
- Firefly Algorithm [[Yang, 2008](https://books.google.de/books?id=iVB_ETlh4ogC&lpg=PR5&ots=DwgyslGEp9&lr&hl=de&pg=PR5#v=onepage&q&f=false)]

## TODO: What are metaheuristics?

## How to use the algorithms?
All three algorithms and their variants share a common interface. Basically, all one needs to do in order to use one of the algorithms for optimization, is invoking the ```search(objective, objective_fct, T)``` method. The parameters all algorithms have in common are the following:
- **objective**: 'min' or 'max', depending on whether it is a minimization or a maximization problem
- **objective_fct**: python-function or lambda, f: R^d -> R
- **d**: dimensionality of solution-space
- **n**: number of solutions in the population
- **range_min**: lower bound of solution-space in all dimensions
- **range_max**: upper bound of solution-space in all dimensions
- **T**: number of iterations

We set these parameters for all following code snippets to exemplary values:
```python
objective = 'min' # minimization or maximation?
objective_fct = lambda x: x[0]**2 + x[1]**2 # function to optimize
d = 2 # dimensionality of solution-space
n = 100 # number of bees, bats or fireflies in the population
range_min, range_max = -5.0, 5.0 # hypercube centered in origin with edge length 10.0
T = 50 # number of iterations
```

### The Bees Algorithm

```python
from bees import BeesAlgorithm

bees = BeesAlgorithm()

solution, latency = bees.search(objective=objective, objective_fct=objective_fct, T=T)
bees.plot_history()
```
### The Bat Algorithm

```python
from bat import BatAlgorithm

bat = BatAlgorithm()

solution, latency = bat.search(objective=objective, objective_fct=objective_fct, T=T)
bat.plot_history()
```

### The Firefly Algorithm

```python
from firefly import FireflyAlgorithm

firefly = FireflyAlgorithm()

solution, latency = firefly.search(objective=objective, objective_fct=objective_fct, T=T)
firefly.plot_history()
```

# Nature-inspired Metaheuristics: Bees, Bats and Fireflies

Implementation of three nature-inspired search algorithms:
- Bees Algorithm [[Pham et al., 2005](https://www.researchgate.net/publication/260985621_The_Bees_Algorithm_Technical_Note)]
- Bat Algorithm [[Yang, 2010](https://arxiv.org/abs/1004.4170v1)]
- Firefly Algorithm [[Yang, 2008](https://books.google.de/books?id=iVB_ETlh4ogC&lpg=PR5&ots=DwgyslGEp9&lr&hl=de&pg=PR5#v=onepage&q&f=false)]

## TODO: What are metaheuristics?

## How to use the algorithms?
All three algorithms and their variants share a common interface. Basically, all one needs to do in order to use one of the algorithms for optimization, is invoking the ```search(objective, objective_fct, T)``` method. The parameters all algorithms have in common (algorithm-independent parameters) are the following:

|parameter    | description                                                                        |domain          |
|:-----------:|------------------------------------------------------------------------------------|----------------|
|objective    | minimization or a maximization problem                                             |'min' or 'max'  |
|objective_fct| python function or lambda to optimize                                              |f: R^d -> R^1   |
|d            | dimensionality of solution-space                                                   |positive integer|
|n            | size of the population, i.e. amount of bees, bats and fireflies                    |positive integer|
|range_min    | lower bound of solution-space in all dimensions                                    |real number     |
|range_max    | upper bound of solution-space in all dimensions                                    |real number     |
|T            | number of iterations                                                               |positive integer|

We set these algorithm-independent parameters for all following code snippets to these exemplary values:
```python
objective = 'min'
objective_fct = lambda x: sum([100*(x[i+1]-x[i])**2+(1-x[i])**2 for i in range(len(x)-1)]) # rosenbrock function
d = 2
n = 100
range_min, range_max = -5.0, 5.0 # hypercube centered in origin with edge length 10.0
T = 50
```

### The Bees Algorithm
Inspired by the foraging behavior of honeybees, Pham et a. designed an algorithm ...

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

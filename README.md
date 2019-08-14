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
|objective    | minimization or maximization problem                                               |'min' or 'max'  |
|objective_fct| python function or lambda to optimize                                              |f: R^d -> R^1   |
|d            | dimensionality of solution-space                                                   |positive integer|
|n            | size of the population, i.e. related to amount of bees, bats and fireflies         |positive integer|
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
Inspired by the foraging behavior of honeybees, Pham et al. designed an algorithm that tries to mimic the way a hive of bees manages to find and harvest fertile flower patches. The basic concept underlying this algorithm can be explained in terms of three different types of patches (basic, best and elite) and two different types of bees (scouts and foragers). Scout bees randomly fly around in the solution-space and inform the other bees after each iteration about the quality of the flower patch they have found (waggle dance). The best scouts recruit a certain number of forager bees to follow them. The scouts that discovered the best of the best flower patches, the elite patches, recruit even more foragers. All the other scouts that have not been that succesful start scanning the solution space randomly again. Foragers do nothing else than searching for even better flower patches nearby the ones their scout discovered, so they are actually only performing local search, while the scout bees are searching globally.

This are the algorithm-dependent parameters of the Bees Algorithm:

|parameter    | description                                                                        |domain          |
|:-----------:|------------------------------------------------------------------------------------|----------------|
|nb           | number of best patches (smaller than or equal to n)                                |positive integer|
|ne           | number of elite patches (smaller than or equal to nb                               |positive integer|
|nrb          | number of recruited foragers per best patch (greater than 0)                       |positive integer|
|nre          | number of recruited foragers per elite patch  (greater than nrb)                   |positive integer|
|shrink_factor| factor for shrinking the patch size                                                |(0, 1]     |
|stgn_lim     | stagnation limit                                                                   |positive integer|

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

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

### The Bees Algorithm

|parameter    | description                                                                        |domain          |
|:-----------:|------------------------------------------------------------------------------------|----------------|
|nb           | number of best patches (smaller than or equal to n)                                |positive integer|
|ne           | number of elite patches (smaller than or equal to nb                               |positive integer|
|nrb          | number of recruited foragers per best patch (greater than 0)                       |positive integer|
|nre          | number of recruited foragers per elite patch  (greater than nrb)                   |positive integer|
|sf           | factor for shrinking the patch size                                                |(0, 1]          |
|sl           | stagnation limit                                                                   |positive integer|

### The Bat Algorithm

|parameter    | description                                                                        |domain          |
|:-----------:|------------------------------------------------------------------------------------|----------------|
|a            | initial loudness of all bats                                                       |positive float  |
|r_max        | maximum pulse rate of bats                                                         |positive float  |
|alpha        | loudness decreasing factor                                                         |(0, 1]          |
|gamma        | pulse rate increasing factor                                                       |(0, 1]          |
|f_min        | minimum sampled frequency                                                          |positive float  |
|f_max        | maximum sampled frequency                                                          |positive float  |

### The Firefly Algorithm

|parameter    | description                                                                        |domain          |
|:-----------:|------------------------------------------------------------------------------------|----------------|
|alpha        | neighbor sphere radius                                                             |positive float  |
|beta_max     | maximum attractivneness                                                            |positive float  |
|gamma        | attractiveness descreasing factor                                                  |positive float  |

## Example Notebooks
You can find exemplary applications of these three implemented metaheuristics in the following few notebooks:
- mle_cauchy.ipynb: Maximum Likelihood Estimation for randomly generated cauchy-distributed samples
- spring_design.ipynb: Spring weight minimization given some constraints
- visualize.ipynb: Example on how to visualize the algorithms in the two-dimensional case

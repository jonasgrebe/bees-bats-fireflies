from search import SwarmSearchAlgorithm


class FireflyAlgorithm(SwarmSearchAlgorithm):
    
    def __init__(self, np, d, range_min, range_max):
        super().__init__("firefly", np, d, range_min, range_max)
        pass
    
    
    def initialize(self):
        raise NotImplementedError
    
    
    def single_search_step(self):
        raise NotImplementedError

    
    def highlight_population(self, ax):
        raise NotImplementedError    
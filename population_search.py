
'''

This module defines an abstract class 'Population' for particle filter search.


'''

import numpy as np



#-----------------------------------------------------------------------------

class Population(object):
    '''
    An abstract container class for a collection of individuals to be used in 
    a 'particle filter' type of search.
    
    At each generation, the probability that the ith individual W[i,:] will be 
    selected for the next generation is proportional to 
    exp(-C[i]/T) where C[i] is the cost of W[i,:] and T is a temperature parameter.
    See 'resample' method.
    
    Usage:
       - Initialize a population 'pop' with the constructor of a subclass 
           derived from Population.       
       - Call pop.particle_filter_search()
       - Retrieve pop.best_w
       
    '''
    def __init__(self, W):
        '''
        Initialize the population with a copy of W
        W[i,:] is a vector coding the ith individual 
        '''                
        self.n = W.shape[0] # population size
        self.W = W.copy()
        self.C = np.zeros((self.n,)) # self.C[i] is the cost of the ith individual
        self.temperature = 1.0
        self.best_cost = np.inf # cost of the best individual seen so far
            
    def mutate(self):
        '''
        This function should be overridden in subclasses.
        Mutate each individual. This operator is application specific.
        '''
        raise NotImplementedError("Method 'mutate' must be implemented in subclasses") 
        
        
    def evaluate(self):
        '''
        This function should be overridden in subclasses.
        Evaluate the cost of each individual.
        Store the result in self.C
        That is, self.C[i] is the cost of the ith individual.
        Keep track of the best individual seen so far in 
            self.best_w = self.W[i_min].copy()
            self.best_cost = cost_min            
        '''
        raise NotImplementedError("Method 'evaluate' must be implemented in subclasses") 

   
    def resample(self):
        '''
        Resample the population. The whole current population is replaced.
        The probability that the ith individual W[i,:] will be selected for the 
        next generation is proportional to 
            exp(-C[i]/T) where C[i] is the cost of W[i,:] 
            and T is a temperature parameter.        
        @pre
          the population has been evaluated by a call
          to self.evaluate()          
        '''
        Ec = np.exp(-self.C/self.temperature)
        sum_Ec = Ec.sum()
        P = Ec/sum_Ec  # P is a probability vector  
        # first argument of np.random.choice has to be 1D. Cannot use X!
        lottery_result = np.random.choice(np.arange(self.n), self.n, replace=True, p = P)  
        self.W = self.W[lottery_result,:]
                
    
    def particle_filter_search(self,
                               steps,
                               log=False):
        '''
        Perform a particle filter search with 'steps' generations.
        @param
            steps: number of generations to be computed
            log : if True, return a log (list) of the W arrays at each step.
        @post
            self.best_w   is the cost of the best solution found
            self.best_cost  is its cost           
        '''
        self.best_cost = np.inf # best cost so far
        if log:
            Lw = []
            Lc = []
        temperature_schedule = np.linspace(self.temperature,1,steps)
        for step in range(steps):
            self.temperature = temperature_schedule[step]
            cost = self.evaluate() # update best_cost 
            if log:
                Lw.append(self.W.copy())
                Lc.append(cost)
            self.resample()
            self.mutate()
        if log:
            return Lw, Lc
        
#-----------------------------------------------------------------------------       
        
        
    
    
#-----------------------------------------------------------------------------       
        

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
#def resample_population(X, C, temperature):
#    '''
#    Individual X[i] has non-negative cost C[i]
#    Its fitness is proportional to exp(-C[i]/temperature)
#    '''
#    n = X.shape[0] # size of the population
#    Ec = np.exp(-C/temperature)
#    sum_Ec = Ec.sum()
#    P = Ec/sum_Ec    
#    # first argument of np.random.choice has to be 1D. Cannot use X
#    lottery_result = np.random.choice(np.arange(n), n, replace=True, p = P)  
#    X_sampled = X[lottery_result,:]
#    return X_sampled
#    
#    #    
#    n=6
#    X = np.arange(10,10+2*6).reshape((-1,2))
#    #np.random.seed(42)
#    C = np.random.rand(n) *4
#    temperature = 1

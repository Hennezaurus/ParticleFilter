'''

2017 IFN680 Assignment

Instructions: 
    - You should implement the class PatternPosePopulation

'''

import numpy as np
import matplotlib.pyplot as plt
import math
import time

import pattern_utils
import population_search

#------------------------------------------------------------------------------

class PatternPosePopulation(population_search.Population):
    '''
    
    '''
    def __init__(self, W, pat):
        '''
        Constructor. Simply pass the initial population to the parent
        class constructor.
        @param
          W : initial population
        '''
        self.pat = pat
        self.true_distance = None
        self.true_cost = np.zeros((W.shape[0]))
        super().__init__(W)
    
    def evaluate(self):
        '''
        Evaluate the cost of each individual.
        Store the result in self.C
        That is, self.C[i] is the cost of the ith individual.
        Keep track of the best individual seen so far in 
            self.best_w 
            self.best_cost 
        @return 
           best cost of this generation            
        '''

        # Get height and width of image
        height, width = self.distance_image.shape[:2]
        
        # clip the x and y coords so they're inside the image
        np.clip(self.W[:,0],0,width-1,self.W[:,0])
        np.clip(self.W[:,1],0,height-1,self.W[:,1])
        
        # Ensure the scale can never drop below 1
        self.W[:,3] = np.maximum(self.W[:,3],1)
        
        # For each individual pose
        for idx, pose in enumerate(self.W):
            
            # Use the given evaluate function to give a score to this pose
            score, verticies = self.pat.evaluate(self.distance_image, pose)
            self.C[idx] = score
            
            # Only worth doing for image one
            if self.true_distance is not None:
                true_score, true_verticies = self.pat.evaluate(self.true_distance, pose)
                self.true_cost[idx] = true_score
            
        # Find index where true cost is the lowest
        if self.true_distance is not None:
            best_index = np.argmin(self.true_cost)
        else:
            best_index = np.argmin(self.C)

        # The cost and pose at this index are our best for this generation
        self.best_cost = self.C[best_index]
        self.best_w = self.W[best_index]
       
        # Return value so it graphs
        return(self.C[best_index])

    def mutate(self):
        '''
        Mutate each individual.
        The x and y coords should be mutated by adding with equal probability 
        -1, 0 or +1. That is, with probability 1/3 x is unchanged, with probability
        1/3 it is decremented by 1 and with the same probability it is 
        incremented by 1.
        The angle should be mutated by adding the equivalent of 1 degree in radians.
        The mutation for the scale coefficient is the same as for the x and y coords.
        @post:
          self.W has been mutated.
        '''
        
        assert self.W.shape==(self.n,4)

        # Cache value of single degree in radians
        degree = math.pi / 180

        # For each individual in population
        for individual in self.W:            
            individual[0] += np.random.choice([-1, 0, 1])             # Mutate X coord by -1, 0 or 1
            individual[1] += np.random.choice([-1, 0, 1])             # Mutate Y coord by -1, 0 or 1
            individual[2] += np.random.choice([-degree, 0, degree])   # Add 1 degree in radians
            individual[3] += np.random.choice([-1, 0, 1])             # Mutate scale by -1, 0 or 1
        
    # No idea why this isn't just in the constructor
    def set_distance_image(self, distance_image):
        self.distance_image = distance_image

#------------------------------------------------------------------------------        

def initial_population(region, scale = 10, pop_size=20):
    '''
    
    '''        
    # initial population: exploit info from region
    rmx, rMx, rmy, rMy = region
    
#    np.random.seed(72) #- Example of local minima with 60 pop / 40 generation
#    np.random.seed(1)    

    W = np.concatenate( (
                 np.random.uniform(low=rmx,high=rMx, size=(pop_size,1)) ,
                 np.random.uniform(low=rmy,high=rMy, size=(pop_size,1)) ,
                 np.random.uniform(low=-np.pi,high=np.pi, size=(pop_size,1)) ,
                 np.ones((pop_size,1))*scale
                 #np.random.uniform(low=scale*0.9, high= scale*1.1, size=(pop_size,1))
                        ), axis=1)    
    return W
    # Debug code to look at a specific pose
    #return np.array([[50, 50, 0, 50]])

#------------------------------------------------------------------------------   
     
def test_particle_filter_search(use_image_one=True, pop_size=60, num_generations=40, verbose=False):
    '''
    Run the particle filter search on test image 1 or image 2 of the pattern_utils module
    
    '''
    
    if use_image_one:
        # use image 1
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_1(verbose)
        ipat = 2 # index of the pattern to target
    else:
        # use image 2
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(verbose)
        ipat = 0 # index of the pattern to target
        
    # Narrow the initial search region
    pat = pat_list[ipat] #  (100,30, np.pi/3,40),
    #    print(pat)
    xs, ys = pose_list[ipat][:2]
    region = (xs-20, xs+20, ys-20, ys+20)
    scale = pose_list[ipat][3]
    
    # Get initial randomized population    
    W = initial_population(region, scale , pop_size)
    
    # Instantiate new PatterPosePopulation with our initial population and pattern
    pop = PatternPosePopulation(W, pat)
    
    # Give it a reference to our distance image
    pop.set_distance_image(imd)
    
    # Set up true distance if image one
    if use_image_one:
        pop.true_distance = make_true_distance_image()
    
    # Set temperature
    pop.temperature = 5
    
    if verbose:
        
        # Run particle filter and return log values for graphs
        Lw, Lc = pop.particle_filter_search(num_generations, log=verbose)
    
        # Generate neat cost vs generation graph
        plt.plot(Lc)
        plt.title('Cost vs generation index')
        plt.show()
    
        # Print out final best results
        print(pop.best_w)
        print(pop.best_cost)
    
        # Display our final solution
        pattern_utils.display_solution(pat_list, 
                                       pose_list, 
                                       pat,
                                       pop.best_w)
                      
        # Show each generation
        pattern_utils.replay_search(pat_list, 
                                    pose_list, 
                                    pat,
                                    Lw)
        
    else:
        # Time the run
        start = time.time()
        
        # Run without returning log values
        pop.particle_filter_search(num_generations, log=verbose)

        # Time end
        end = time.time()
        
        # Elapsed time
        time_elapsed = end - start

        # 99.99% sure I fixed this, but just keep an eye out in case
        if math.isnan(pop.best_cost):
            raise ValueError('A cost was NaN, revisit clipping code')
            return(np.inf)
        
        return(pop.best_cost, time_elapsed)
        
    
#-----------------------------------------------------------------------------    
    
def make_true_distance_image(show=False):
#    ps = pattern_utils.Square()
    pt = pattern_utils.Triangle(2)
    
    pat_list = [pt]#, pt]
    pose_list = [(100,30, np.pi/3,40)]#,
              #   (100,50, -np.pi/3,30)]    
#    region = (45,90,25,60)
    imf = pattern_utils.pat_image(pat_list, pose_list)

    imd = pattern_utils.dist_image(imf)
    
    if show:
        plt.figure()
        plt.imshow(imf)
        plt.title('imf')
        plt.figure()
        plt.imshow(imd)
        plt.title('imd')
        plt.colorbar()
        plt.show()
        
    return imd
    

#-----------------------------------------------------------------------------

def get_clean_combinations(individual_limit):
    
    # Generate all integers up to and including our limit
    values = np.arange(1, individual_limit+1, 1)
    
    # List to populate
    clean_combinations = []
    
    # Find and add all clean divisions
    for i in values:
        if individual_limit % i == 0:
            clean_combinations.append([i, (individual_limit/i).astype(int)])
            
    # Return neat list of divisible combinations
    return(clean_combinations)


#-----------------------------------------------------------------------------
        
def compare_pop_vs_gen(iterations, num_individuals, image_one=True, verbose=True):
        
    # All pairs of values which cleanly multiply to num_individuals
    combinations = get_clean_combinations(num_individuals)

    # List of combo values, cost and time
    combo_values = np.zeros((len(combinations), iterations, 2))

    # For each combo
    for idx, combo in enumerate(combinations):
    
        # Repeat for n iterations
        for i in range(iterations):
        
            # Run the search
            best_cost, delta_time = test_particle_filter_search(use_image_one=image_one,
                                                                pop_size=combo[0],
                                                                num_generations=combo[1],
                                                                verbose=False)
            # Store cost and time
            combo_values[idx,i,0] = best_cost
            combo_values[idx,i,1] = delta_time

    
    # To plot medians
    combo_medians = np.zeros((len(combinations), 2))
    
    # For each combination, get median cost and time across all iterations
    for i in range(len(combinations)):
        combo_medians[i,0] = np.median(combo_values[i,:,0])        
        combo_medians[i,1] = np.median(combo_values[i,:,1])        
        
        # Print exact data
        if(verbose):
            print("\nPopulation Count: " + str(combinations[i][0]) +
                  "\nGeneration Count: " + str(combinations[i][1]) +
                  "\nCosts:")
            for cost in combo_values[i,:,0]:
                print("\t" + str(cost))
            print("\tMedian Cost: " + str(combo_medians[i,0]))
            print("\nTimes:")
            for timetaken in combo_values[i,:,1]:
                print("\t" + str(timetaken))
            print("\tMedian Time: " + str(combo_medians[i,1]))

    if verbose:
        # Dummy x axis values for graph
        xAxis = np.arange(0, len(combinations), 1)

        # Graph Error       
        plt.bar(xAxis, combo_medians[:,0])
        plt.title("Error As Generations vs Population Change")
        plt.xlabel('Combination ID')
        plt.ylabel('Median Error of Final Pose')
        plt.xticks(np.arange(min(xAxis), max(xAxis)+1, 1.0))
        plt.show()
        
        # Graph Time
        plt.bar(xAxis, combo_medians[:,1])
        plt.title("Time Taken As Generations vs Population Change")
        plt.xlabel('Combination ID')
        plt.ylabel('Median Time Taken (seconds)')
        plt.xticks(np.arange(min(xAxis), max(xAxis)+1, 1.0))
        plt.show()
        
    # Otherwise return the best cost and best time found
    else:
        return(np.min(combo_medians[:,0]), np.min(combo_medians[:,1]))    


#------------------------------------------------------------------------------

    
def compare_computational_budgets(comp_budgets, iteration_count, use_image_one=True, verbose=True):
    
    # Prepare an array to populate with our best answers given various individual counts
    budget_results = np.zeros((len(comp_budgets), 2))
  
    # Loop through the list of given individual counts and track their best result
    for idx, budget in enumerate(comp_budgets):
        
        if verbose:
            print("Testing Budget: " + str(budget))
        
        # Run test for this computational budget, store best results
        cost, time_taken = compare_pop_vs_gen(iterations = iteration_count,
                                              num_individuals = budget,
                                              image_one = use_image_one,
                                              verbose = False)
        
        # Store results in final array
        budget_results[idx,0] = cost
        budget_results[idx,1] = time_taken
    
    
    if verbose:
        
        # Print out log data
        for i, budget in enumerate(comp_budgets):
            print("Testing with " + str(budget) + " individuals")
            print("\tBest cost: " + str(budget_results[i,0]))
            print("\tBest time: " + str(budget_results[i,1]))
    
        # Graph results
        x_axis = np.arange(0, len(comp_budgets), 1)
    
        # Plot Cost
        plt.bar(x_axis, budget_results[:,0])
        plt.title("Error of Result for Budgets")
        plt.xlabel('Budget ID')
        plt.ylabel('Lowest Final Error Achieved')
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.show()
        
        # Plot Time
        plt.bar(x_axis, budget_results[:,1])
        plt.title("Time Taken by Budgets")
        plt.xlabel('Budget ID')
        plt.ylabel('Lowest Time Taken')
        plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0))
        plt.show()
    
    
#------------------------------------------------------------------------------        

if __name__=='__main__':
    
    
    """
    test_particle_filter_search(use_image_one=True,
                                pop_size=100,
                                num_generations=10,
                                verbose=True)
    """

    """
    test = compare_pop_vs_gen(iterations = 9,
                              num_individuals = 1000,
                              image_one = True,
                              verbose=True)
    """
    
    
    budgets = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    compare_computational_budgets(budgets,
                                  iteration_count = 1,
                                  use_image_one=True,
                                  verbose=True)
    
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
#        
#    def test_2():
#        '''
#        Run the particle filter search on test image 2 of the pattern_utils module
#        
#        '''
#        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(False)
#        pat = pat_list[0]
#        
#        #region = (100,150,40,60)
#        xs, ys = pose_list[0][:2]
#        region = (xs-20, xs+20, ys-20, ys+20)
#        
#        W = initial_population_2(region, scale = 30, pop_size=40)
#        
#        pop = PatternPosePopulation(W, pat)
#        pop.set_distance_image(imd)
#        
#        pop.temperature = 5
#        
#        Lw, Lc = pop.particle_filter_search(40,log=True)
#        
#        plt.plot(Lc)
#        plt.title('Cost vs generation index')
#        plt.show()
#        
#        print(pop.best_w)
#        print(pop.best_cost)
#        
#        
#        
#        pattern_utils.display_solution(pat_list, 
#                          pose_list, 
#                          pat,
#                          pop.best_w)
#                          
#        pattern_utils.replay_search(pat_list, 
#                          pose_list, 
#                          pat,
#                          Lw)
#    
#    #------------------------------------------------------------------------------        
#        
    
'''

2017 IFN680 Assignment

Instructions: 
    - You should implement the class PatternPosePopulation

'''

import numpy as np
import matplotlib.pyplot as plt
import math, time, csv, string

import pattern_utils
import population_search

#------------------------------------------------------------------------------

class PatternPosePopulation(population_search.Population):
    '''
    Pose recognition algorithm.
    
    This class is initalized with data, and will run a particle filter
    test, looking for a specific pattern within an image
    '''
    def __init__(self, W, pat):
        '''
        Constructor. Simply pass the initial population to the parent
        class constructor.
        @param
          W : initial population
        '''
        self.pat = pat
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
        
        # Find score for each individual pose
        for idx, pose in enumerate(self.W):
            self.C[idx], verticies = self.pat.evaluate(self.distance_image, pose)

            
        # Find index where cost is the lowest
        best_index = np.argmin(self.C)

        # The cost and pose at this index are our best for this generation
        self.best_cost = self.C[best_index]
        self.best_w = self.W[best_index]

        # Return value so it graphs
        return(self.best_cost)
    

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

def initial_population(region, scale = 10, pop_size=20, seed=None):
    '''
    Generate initial population poses.
    Creates a series of random guesses to populate our self.W
    pose limit. These are bound within a region to keep the guesses
    reasonable (not outside the image)
    
    @return:
        Initial values to use for self.W 
    '''        
    # initial population: exploit info from region
    rmx, rMx, rmy, rMy = region
    
    # Randomize if not chosen
    if(seed is None):
        seed = np.random.randint(0, 1000)
    
    # For reproducability
    np.random.seed(seed)    

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
     
def test_particle_filter_search(use_image_one=True, pop_size=60, num_generations=40, seed=None, verbose=False):
    '''
    Run the particle filter search on test image 1 or image 2 of the pattern_utils module
    
    If verbose, prints out the neat graph replay of the search,
    including the final result
            
    @return:
        If not verbose, return the final best cost, and time taken
    '''
    
    # Create images needed
    true_image = None
    
    if use_image_one:
        # use image 1
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_1(verbose)
        ipat = 2 # index of the pattern to target
        true_image = make_true_distance_image()
    else:
        # use image 2
        imf, imd , pat_list, pose_list = pattern_utils.make_test_image_2(verbose)
        ipat = 0 # index of the pattern to target
        
    
    # Narrow the initial search region
    pat = pat_list[ipat]
    xs, ys = pose_list[ipat][:2]
    region = (xs-20, xs+20, ys-20, ys+20)
    scale = pose_list[ipat][3]
    
    # Get initial randomized population    
    W = initial_population(region, scale , pop_size, seed)
    
    # Instantiate new PatterPosePopulation with our initial population and pattern
    pop = PatternPosePopulation(W, pat)
    
    # Give it a reference to our distance image
    pop.set_distance_image(imd)
    
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
        print("Best Pose: \n" + str(pop.best_w))
        print("Best cost: " + str(pop.best_cost))
        
        # Find true cost of best pose
        if(true_image is not None):
            best_idx = np.argmin(pop.C)
            true_cost, verts = pop.pat.evaluate(true_image, pop.W[best_idx])
            print("Best true cost: " + str(true_cost))
    
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
        
        if(true_image is not None):
            # Find true cost of best pose
            best_idx = np.argmin(pop.C)
            true_cost, verts = pop.pat.evaluate(true_image, pop.W[best_idx])

        # 99.99% sure I fixed this, but just keep an eye out in case
        if math.isnan(pop.best_cost):
            raise ValueError('A cost was NaN, revisit clipping code')
            return(np.inf)
        
        return(pop.best_cost, true_cost, time_elapsed)
        
    
#-----------------------------------------------------------------------------    
    
def make_true_distance_image(show=False):
    '''
    Create a depth image with just the triangle we're searching for
    
    This allows us to compare to this for our final evaluation (not during
    the evaluate call) to see how close our final result was to the 'actual'
    image we're searching for, not just 'low cost' which could be a mis-
    classification.
    
    @return:
        Depth image to calculate true cost against
    '''
    
    pt = pattern_utils.Triangle(2)
    
    pat_list = [pt]#, pt]
    pose_list = [(100,30, np.pi/3,40)]

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
    '''
    Find all population/generation combinations for this budget
    
    @individual_limit:
        The evaluation budget we want to find combos for
        
    @return:
        All pairs of integers which can be cleanly multiplied together
        to get the individual_limit
    '''
    
    # Generate all integers up to and including our limit
    values = np.arange(1, individual_limit+1, 1)
    
    # Create list of clean combinations
    clean_combinations = [(i, individual_limit//i) for i in values if individual_limit % i == 0]
            
    # Return neat list of divisible combinations
    return(clean_combinations)


#-----------------------------------------------------------------------------
        
def compare_pop_vs_gen(iterations, num_individuals, image_one=True, verbose=True):
    '''
    Summary:
        Compare different population and generation counts for a given budget.
    
    Description:
        Run the particle filter [iterations] times for each clean combination,
        and keep the results in terms of cost, true cost, and time for
        every search run. Returns this as a numpy array
    
    Inputs:
    @iterations:
        Number of times to repeat filter for each combo to get more accuracy
        Takes median of all iterations
    @num_individuals:
        Computational budget to do the testing for
    @image_one:
        Whether to use test image one, or test image two
    @verbose:
        Print and graph the data if true
        Return useful minimums if false
    
    Outputs:
    @return:
        If verbose prints out and graphs values found
        If not verbose returns the full ndarray of values found
        
    '''
    
    # All pairs of values which cleanly multiply to num_individuals
    combinations = get_clean_combinations(num_individuals)

    # List of combo values, cost and time
    combo_values = np.zeros((len(combinations), iterations, 7))

    # For each combo
    for idx, combo in enumerate(combinations):
    
        # Repeat for n iterations
        for i in range(iterations):
        
            # Run the search
            best_cost, true_cost, delta_time = test_particle_filter_search(use_image_one=image_one,
                                                                           pop_size=combo[0],
                                                                           num_generations=combo[1],
                                                                           seed=i,
                                                                           verbose=False)
            # Store row of data for this filter
            combo_values[idx,i,4] = best_cost              # Best Cost
            combo_values[idx,i,5] = true_cost              # True Cost of best pose
            combo_values[idx,i,6] = delta_time             # Time taken to run filter
            combo_values[idx,i,0] = num_individuals   # Comp budget for this run
            combo_values[idx,i,2] = combo[0]          # Population Count
            combo_values[idx,i,1] = combo[1]          # Generation Count
            combo_values[idx,i,3] = i                 # Iteration number (also seed used)

    
    # To plot medians
    combo_medians = np.zeros((len(combinations), 3))
    
    # For each combination, get median cost and time across all iterations
    for i in range(len(combinations)):
        combo_medians[i,0] = np.median(combo_values[i,:,0])        
        combo_medians[i,1] = np.median(combo_values[i,:,1])
        combo_medians[i,2] = np.median(combo_values[i,:,2])        
        
        # Print exact data
        if(verbose):
            print("\nPopulation Count: " + str(combinations[i][0]) +
                  "\nGeneration Count: " + str(combinations[i][1]))
            
            print("\nCosts:")
            for cost in combo_values[i,:,0]:
                print("\t" + str(cost))
            print("\tMedian Cost: " + str(combo_medians[i,0]))
            
            print("\nTrue Costs:")
            for true_cost in combo_values[i,:,1]:
                print("\t" + str(true_cost))
            print("\tMedian True Cost: " + str(combo_medians[i,1]))
            
            print("\nTimes:")
            for timetaken in combo_values[i,:,2]:
                print("\t" + str(timetaken))
            print("\tMedian Time: " + str(combo_medians[i,2]))
            
            

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
        
        
        # Graph True Error
        plt.bar(xAxis, combo_medians[:,1])
        plt.title("True Cost As Generations vs Population Change")
        plt.xlabel('Combination ID')
        plt.ylabel('Median True Error of Final Pose')
        plt.xticks(np.arange(min(xAxis), max(xAxis)+1, 1.0))
        plt.show()
        
        # Graph Time
        plt.bar(xAxis, combo_medians[:,2])
        plt.title("Time Taken As Generations vs Population Change")
        plt.xlabel('Combination ID')
        plt.ylabel('Median Time Taken (seconds)')
        plt.xticks(np.arange(min(xAxis), max(xAxis)+1, 1.0))
        plt.show()
        
    # Otherwise return the entire data-set
    else:
        return(combo_values)    


#------------------------------------------------------------------------------

def compare_computational_budgets(comp_budgets, iteration_count, use_image_one=True, verbose=True):
    '''
    Compare different computational budgets
    
    Runs all combinations for each budget 'iteration_count' times (for accuracy)
    gets the median of the iterations, then the minimum (or best) cost and time
    accross the combo's tested, so we're comparing the 'best' combo for each 
    computational budget.
    
    @comp_budgets:
        A list of budgets we want to test
    @iteration_count:
        How many times to repeat each combo's test for accuracy
    @use_image_one:
        Whether to use test image one or test image two
    @verbose:
        Whether to print out and graph our results
        
    '''
    
    # Prepare an array to populate with our best answers given various individual counts
    budget_results = []
  
    # Loop through the list of given individual counts and track their best result
    for idx, budget in enumerate(comp_budgets):
        
        print("Testing Budget: " + str(budget))
        
        # Run test for this computational budget, store results
        budget_results.append(compare_pop_vs_gen(iterations = iteration_count,
                                                 num_individuals = budget,
                                                 image_one = use_image_one,
                                                 verbose = False))
        
    # Write the complete data output to a file
    save_data(budget_results)
    
    
def save_data(data):
    
    # Open connection to the file
    with open('data.csv', 'w', newline='') as csvfile:
        w = csv.writer(csvfile, delimiter=',')
        w.writerow(['Budget', 'Generations', 'Population', 'Seed', 'Best Cost', 'True Cost', 'Duration'])
        for budget in data:
            for combo in budget:
                for iteration in combo:
                    # Prepare output
                    output = []
                    output.append(str(int(iteration[0])))
                    output.append(str(int(iteration[1])))
                    output.append(str(int(iteration[2])))
                    output.append(str(int(iteration[3])))
                    output.append(str(iteration[4]))
                    output.append(str(iteration[5]))
                    output.append(str(iteration[6]))
                    
                    # Add to csv
                    w.writerow(output)
        
#------------------------------------------------------------------------------        

if __name__=='__main__':
    
    
    '''
    test_particle_filter_search(use_image_one=True,
                                pop_size=40,
                                num_generations=50,
                                verbose=True)
    '''
    

    '''
    comparison = compare_pop_vs_gen(iterations = 5,
                                    num_individuals = 200,
                                    image_one = True,
                                    verbose=True)
    '''
    

    '''     
    budgets = [100, 200], 400, 600] #, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    compare_computational_budgets(budgets,
                                  iteration_count = 3,
                                  use_image_one=True,
                                  verbose=False)
    '''
    
#------------------------------------------------------------------------------

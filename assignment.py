import random
import numpy as np

#first define the assign function to generate peer-review assignments
def assign(n=10,m=7):
    """
    n is the number of PIs and m is the number of proposals each PI review
    Create an n*m array to input the assigned proposals for each PI
    """
    pi_assigned=np.zeros((n,m))
    """
    create a propsal_pool list and a vector of numbers of each 
    proposal reviewed so far
    """
    prop_pool=list(range(n))
    count_assigned=np.zeros(n)
    for i in range(n):
        ##for each PI, randomly select m reviews from the proposal pool
        pi_assigned[i]=random.sample([x for x in prop_pool if x != i],m)
        """
        after each pi was assigned, re-count the number of each 
        proposal reviewed
        """
        for x in pi_assigned[i]:
            count_assigned[int(x)] += 1
            """
            remove the proposal from proposal_pool if a proposal has been
            reviewed m times
            """
            for j in prop_pool:
                if count_assigned[j]== m:
                    prop_pool.remove(j)
    return pi_assigned

##creat the parameter to count the times of success and failure 
count_failure=0
count_success=0

##define the number of iterations each assignment programming will run
iter_num=100

for iter in range(iter_num):    
    try:
        ##call assign function and try different values here
        assigned=assign(n=30,m=7)
        print(assigned)
        count_success += 1
    except:
        count_failure += 1

print("number of failure=", count_failure) 
print("number of success=", count_success)

import random
import numpy as np
from scipy.stats import truncnorm
from matplotlib import pyplot as plt
from scipy.stats import rankdata


##assign different values here
n=10
m=4
mean_true_scores,sd_true_scores=50,20
df_reliability,mean_bias,sd_bias=3,0,10
iter_num=20


########################## Step 1, assignment #################################

##first define the assign function to generate peer-review assignments
##the paramters are n (total No. of PIs) and m(No. of proposals each PI review)
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
        ##for each PI, randomly select m proposals from the proposal pool
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

#get the first successful random assignment matrix   
assignment = None
while assignment is None:  
    try:
        ##call assign function and try different values for parameters here
        assignment=assign(n,m)        
        assignment=assignment.astype(int)
        #print('assignment',assignment)       
    except:
        pass

######################### Step 2, Simulate scores##############################
#np.random.seed(12345)

##define function to simulate true scores using truncated normal distribution
##The parameters are n, mean,standard deviation, lower and upper bound
def true_scores (n=50,mean_true_scores=50,sd_true_scores=20,low=0,upp=100):
    true_scores= truncnorm.rvs((low - mean_true_scores) / sd_true_scores, 
                       (upp - mean_true_scores) / sd_true_scores, 
                       loc=mean_true_scores, scale=sd_true_scores,size=n)
    return true_scores

#call true_scores function and assign different values
true_scores=true_scores(n,mean_true_scores,sd_true_scores) 

##Write a function to generate the score of a proposal given by a PI
## the parameters are n, df_reliability, mean_bias and sd_bias
def prop_scores(n=50, df_reliability=6,mean_bias=0,sd_bias=5):
    prop_scores=np.zeros((n,n))
    ##Generate the reliability of PIs’ reviews using Chi-square distribution
    reliability = np.random.chisquare(df_reliability, size=n)
    ##Generate bias of each PI with different mean and s.d..
    bias=np.random.normal(mean_bias,sd_bias,size=n)    
    for i in range(0,n):
        for j in range(0,n):
            # the score of proposal j given by PI i
            prop_scores[i,j]=np.random.normal(true_scores[j]+bias[i],reliability[i])
            if prop_scores[i,j] <0:
                prop_scores[i,j]=0
            else:
                if prop_scores[i,j]>100:
                    prop_scores[i,j]=100                
    return prop_scores

#call prop_scores function and assign different values
prop_scores=prop_scores(n, df_reliability, mean_bias, sd_bias)  

#modify the scores to integers which are more reasonable in real life
true_scores=np.round(true_scores)
prop_scores=np.round(prop_scores)

##Output the generated numbers to files
prop_average_scores=np.mean(prop_scores,axis=0)
np.savetxt('true_scores.csv',true_scores)   #Output true_scores to files
np.savetxt('prop_score.csv',prop_scores)    #Output prop_scores to files

##plot the average scores of each proposal given by all PIs
plt.plot(true_scores, 'r--^',label='true_score')
plt.plot(prop_average_scores, 'b--^',label='PI_avg_score')
plt.legend(bbox_to_anchor=(0.7, 1))
plt.xlabel('proposal number')
plt.ylabel('scores')
plt.show()

######################## Step 3, Generate global ranking ######################
#generate score matrix and rank matrix for the assignment
simulated_assignment_scores=np.zeros((n,m))
simulated_assignment_ranks=np.zeros((n,m))
for i in range(n):
    for j in range(m):
        ##fill in the simulated scores for the successful assignment        
        simulated_assignment_scores[i,j]=prop_scores[i,assignment[i,j]]
    ##Assign ranks to assignment_scores, dealing with ties appropriately
    simulated_assignment_ranks[i]=rankdata(simulated_assignment_scores[i])-1   


prop_total_scores=np.zeros(n)
MBC_total_scores=np.zeros(n)
for i in range(n):
    #find index of assignment for each proposal
    find_index=np.where(assignment==i)
    prop_index=np.asarray(find_index).T.tolist()
    for (r,v) in prop_index:
        ##summarize simulated ranks from different PIs for each proposal
        prop_total_scores[i]+=simulated_assignment_ranks[r,v]
        MBC_total_scores[i]=prop_total_scores[i] / (m*(m-1))
        

prop_global_ranks=rankdata(MBC_total_scores)-1
#print('prop_global_ranks',prop_global_ranks)


################### Step 4, Concordance index computation #####################
##Use Concordance Index (CI) to measure the ranking accuracy
prop_true_ranks=rankdata(true_scores)-1
#print('prop_true_ranks',prop_true_ranks)

def compCI(r1, r2, n):
    C = 0
    T=0
    for i in range(n-1):
        for j in range(i+1,n):
            ##compute number of concordant pairs
            if (r1[i]-r1[j])*(r2[i]-r2[j])> 0:
                C += 1
            else:
                ##compute number of tied pairs
                if (r1[i]-r1[j]) * (r2[i]-r2[j]) == 0:
                    T += 1
    CI = (C+0.5*T)*2/(n*n-n)
    return CI

CI_global=compCI(prop_global_ranks, prop_true_ranks, n)


############## Step 5, Generate incentivized ranking and compute CI ###########
global_assignment_ranks=np.zeros((n,m))
global_assignment_ranks_ranks=np.zeros((n,m))
for i in range(n):
    for j in range(m):
        ##fill in the global ranks for the successful assignment        
        global_assignment_ranks[i,j]=prop_global_ranks[assignment[i,j]]
    ##Assign ranks to the global ranks
    global_assignment_ranks_ranks[i]=rankdata(global_assignment_ranks[i])-1

##compute quality index
quality_index_diff= abs(simulated_assignment_ranks - global_assignment_ranks_ranks)
QI=quality_index_diff.sum(axis=1)

#print('pi_quality_index',pi_quality_index)

##compute the average difference in score between adjacently ranked proposals
alpha=(max(prop_total_scores)-min(prop_total_scores)) / n
if m % 2==0:
    Qmax=m**2/2
else:
    Qmax=(m**2-1)/2
bonus=np.zeros(n)
for i in range(n):
    bonus[i]=2*alpha*(Qmax-QI[i])/Qmax
incentivized_scores=prop_total_scores+bonus
prop_incentivized_ranks=rankdata(incentivized_scores)-1
CI_incentivized=compCI(prop_incentivized_ranks, prop_true_ranks, n)        


##plot the average scores of each proposal given by all PIs
plt.plot(prop_true_ranks, 'b--^',label='prop_true_ranks')
plt.plot(prop_global_ranks, 'r--^',label='prop_global_ranks')
plt.plot(prop_incentivized_ranks, 'g--^',label='prop_incentivized_ranks')
plt.legend(bbox_to_anchor=(0.7, 1))
plt.xlabel('proposal number')
plt.ylabel('ranks')
plt.show()        

print('CI_global',CI_global)
print('CI_incentivized',CI_incentivized)
    
        
                    
                   
        
            
            
            
             
            
                        

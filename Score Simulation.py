# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 16:05:15 2017

@author: Hui
"""

from scipy.stats import truncnorm
import numpy as np
from matplotlib import pyplot as plt

#np.random.seed(60)

###############################################################
###Step 1 :                                                 ###
###Given number of PIs, simulate their proposal true scores ###
###using truncated normal distribution.n = 50; The lower    ###
###and upper bound is 0 and 100; The mean is 50.            ###
###The standard deviation varies (i.e. 5, 10, 20, etc.)     ###
###############################################################
def true_scores (sd_true_scores=20,low=0,upp=100,mean_true_scores=50,n=50):
    true_scores= truncnorm.rvs((low - mean_true_scores) / sd_true_scores, 
                       (upp - mean_true_scores) / sd_true_scores, 
                       loc=mean_true_scores, scale=sd_true_scores,size=n)
    return true_scores

true_scores=true_scores(sd_true_scores=10) #call function and assign different values

############################################################################
###Step 2 :                                                              ###
###Generate the reliability of PIs’ reviews using Chi-square distribution###
### with different degree of freedom.Generate bias of each PI using      ###
###normal distributions with different mean and s.d..                    ###
###Write a function to generate the score of a proposal given by a PI    ###
############################################################################

def prop_scores(df_reliability=6,mean_bias=0,sd_bias=5,n=50):
    prop_scores=np.zeros((n,n))
    reliability = np.random.chisquare(df_reliability, size=n)
    bias=np.random.normal(mean_bias,sd_bias,size=n)
    for i in range(0,n):
        for j in range(0,n):
            # the score of proposal i given by PI j
            prop_scores[i,j]=np.random.normal(true_scores[i]+bias[j],reliability[j])
    return prop_scores

prop_scores=prop_scores(6,0,5)  #call function and assign different values

###############################################################
###Step 3 :                                                 ###
###Output the generated numbers to files and plot figures to### 
###show that the parameters chosen are reasonable.          ###
###############################################################

prop_average_scores=np.mean(prop_scores,axis=1)
#np.savetxt('prop_score_HYuan.csv',prop_scores)   #Output to files
prop_average_scores=np.mean(prop_scores,axis=1)
plt.plot(true_scores, 'r--^',label='true_score')
plt.plot(prop_average_scores, 'b--^',label='PI_avg_score')
plt.legend(bbox_to_anchor=(0.7, 1))
plt.xlabel('proposal number')
plt.ylabel('scores')
#plt.savefig('pa1_HYuan.png') #save plots
plt.show()

#################################################################
###Conclusion:                                                ###
###When sd_true_scores>=10 and mean_bias=0, the simulation of ###
###prop_scores given by PIs mathches true_scores very well,   ###
###indicating the parameters chosen are reasonable.           ### 
###Change of other parameters will not make much difference on###
###the prop_average_scores given by PIs.                      ###
#################################################################

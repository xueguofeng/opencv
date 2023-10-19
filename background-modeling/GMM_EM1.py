# coding=utf-8
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


men = np.random.normal(1.80, 0.05, 100)
women = np.random.normal(1.60, 0.1, 100)

plt.figure(21)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax1.set_title('Height-Boys')
#ax1.hist(men, bins=60, histtype="stepfilled",normed=True,alpha=0.6, color='r')
ax1.hist(men, bins=60, histtype="stepfilled",density=True,alpha=0.6, color='r')
ax2.set_title('Height-Girls')
#ax2.hist(women, bins=60, histtype="stepfilled",normed=True,alpha=0.6)
ax2.hist(women, bins=60, histtype="stepfilled",density=True,alpha=0.6)
plt.savefig('Height-Boys-Girls')
#plt.show()


def em_single(observations, param_a, param_b):

    boys_probability_list = []   # Possibility - Boys for 1000 samples
    girls_probability_list = [] # Possibility - Girls for 1000 samples

    # Step E - calcualte Q(z) based on Models
    for i, v in enumerate(observations):
        p1 = norm.pdf(v, loc=param_a[0], scale=param_a[1]) # Possibility in Model - Boys
        p2 = norm.pdf(v, loc=param_b[0], scale=param_b[1]) # Possibility in Model - Girls
        # print(p1, p2)
        boys_probability_list.append( p1/(p1+p2) )     # Possibility - Boys
        girls_probability_list.append( p2/(p1+p2) )   # Possibility - Girls

        # GMM: p = 0.5 p1 + 0.5 p2
        # Possibility-Boys,  0.6 p1 / (0.6 p1 + 0.4 p2)
        # Possibility-Girls,  0.4 p2 / (0.6 p1 + 0.4 p2)


    # Step M - update θ to maximize L(θ) based the Q(z)

    sum1 = 0
    sum2 = 0

    for i, height in enumerate(observations):
        sum1 += boys_probability_list[i] * height     #
        sum2 += girls_probability_list[i] * height    #

    temp1 = sum(boys_probability_list)
    temp2 = sum(girls_probability_list)

    loc1 = sum1 / temp1
    loc2 = sum2 / temp2

    scale1 = 0.
    scale2 = 0.

    for i, v in enumerate(observations):
        scale1 += boys_probability_list[i] * (v-loc1) * (v-loc1)
        scale2 += girls_probability_list[i] * (v-loc2) * (v-loc2)


    scale1 = np.sqrt( scale1 / sum(boys_probability_list) )
    scale2 = np.sqrt( scale2 / sum(girls_probability_list) )

    return [[loc1, scale1], [loc2, scale2]] #  the updated θ to maximize L(θ) based on Q(Z)

def em(observations, param_a, param_b, tol = 1e-6, iterations=1000):

    for iter in range(iterations):
           # Boy        Girl
        [param_a_new, param_b_new] = em_single(observations, param_a, param_b)

        temp1 = abs( param_a_new[0]-param_a[0] )+abs( param_a_new[1]-param_a[1] ) # parameter changes of Model-A
        temp2 = abs( param_b_new[0]-param_b[0] )+abs( param_b_new[1]-param_b[1] ) # parameter changes of Model-B

        if temp1 < tol and temp2 < tol:  # if no changes on the model parameters, return
            return ([param_a_new, param_b_new], iter+1)

        #if abs(param_a_new[0]-param_a[0])+abs(param_a_new[1]-param_a[1]) < tol \
        #    and abs(param_b_new[0]-param_b[0])+abs(param_b_new[1]-param_b[1]) < tol:
        #    return ([param_a_new, param_b_new], iter+1)

        param_a = param_a_new.copy()  # keep the updated model parameters
        param_b = param_b_new.copy()  # keep the updated model parameters

    return ([param_a, param_b], iterations)

observations = []
for v in men:
    observations.append(v)
for v in women:
    observations.append(v)

param_a = [1.7, 1]  # initial model parameters for boys
param_b = [1.4, 1]  # initial model parameters for girls

       # 200 samples,  Model-Boys Model-Girls
temp = em(observations, param_a, param_b)
print( temp )
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:47:39 2021

@author: Marek
"""

import numpy as np
import scipy.stats as stat
import matplotlib.pyplot as plt

class MultiArmedBandit():
    """
    Implementation of the most basic algorithm to solve multi-armed bandit problem.
    Uses constant e-greedy action selection
    """
    
    def __init__(self, n_steps, e, n_dist):
        """
        Initializes array of action values (self.Q), array of number of times each action
        was selected (self.N), list of actions (self.A). Creates random gaussian 
        reward distribudions (self.dists).
        
        
        Args:
            n_steps: n interations for estimation of action values
            e: epsilon in epsilon greedy action selection
            n_dist: number of actions/distributions
        """
        self.n_steps = n_steps
        self.e = e
        self.dists = []
        self.Q = np.array([0]*n_dist, dtype=np.float32)
        self.N = np.array([0]*n_dist)
        self.A = []
        self.final_rewards = []
        for i in range(n_dist):
            self.A.append(i)
            m = np.random.uniform(-10, 10)
            std = np.random.uniform(0, 10)
            # loc == mean, scale == std
            self.dists.append(stat.norm(loc=m, scale=std))
            
    def call_sample_avg(self, stationary=True):
        """
        Iterates n_steps times for estimating action values through sample averages.
        If variable random less or equal to e -> random action else greedy action 
        with random tie breaking. 
        
        random: random nuber used to determine exploration/exploitation
        a : action at step i
        r : reward at step i
        """
        for i in range(n_steps):
            if not stationary:
                self.update_reward_dists()
                
            random = np.random.uniform(0, 1)
            if random <= self.e:
                a = np.random.choice(self.A)
            else:
                # random choice among greedy actions
                a = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
            r = self.dists[a].rvs()
            self.N[a] = self.N[a] + 1
            self.Q[a] = self.Q[a] + 1/self.N[a] * (r - self.Q[a])
            self.final_rewards.append(r)
    
    def update_reward_dists(self, m=0, std=1):
        for i, d in enumerate(self.dists):
            params = d.kwds
            random_change = stat.norm.rvs(loc=m, scale=std)
            self.dists[i] = stat.norm(
                loc=params['loc']+random_change,
                scale=params['scale']
            )

if __name__ == '__main__':
    np.random.seed(0)
    n_steps = 1000
    n_dist = 5
    e = 0.1
    
    bandit = MultiArmedBandit(n_steps, e, n_dist)
    bandit.call(False)
    
    print('Distributions args:', [d.kwds for d in bandit.dists])
    print('\nAction values:', bandit.Q)
    print('\nNumber of action choices:', bandit.N)
    
    plt.plot(range(n_steps), np.cumsum(bandit.final_rewards))
    plt.title('Cum sum of rewards')
    plt.xlabel('step')
    plt.ylabel('Sum')
    plt.show()
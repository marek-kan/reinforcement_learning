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
    
    def call_weighted_avg(self, stationary=True, step_size = 0.1, decay_step_size=False):
        """
        Iterates n_steps times for estimating action values through weighted averages.
        If variable random less or equal to e -> random action else greedy action 
        with random tie breaking. 
        
        random: random nuber used to determine exploration/exploitation
        a : action at step i
        r : reward at step i
        """
        o = 0
        for i in range(n_steps):
            if not stationary:
                self.update_reward_dists()
            if decay_step_size:
                step_size = step_size * (1 / (1 + 0.001 * i))
                
            random = np.random.uniform(0, 1)
            if random <= self.e:
                a = np.random.choice(self.A)
            else:
                # random choice among greedy actions
                a = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
            r = self.dists[a].rvs()
            self.N[a] = self.N[a] + 1
            self.Q[a] = self.Q[a] + step_size * (r - self.Q[a])
            self.final_rewards.append(r)
    
    def update_reward_dists(self, m=0, std=1):
        for i, d in enumerate(self.dists):
            params = d.kwds
            random_change_m = np.random.uniform(-3, 3)
            random_change_s = np.random.uniform(-1, 1)
            new_std = params['scale']+random_change_s
            if new_std<0:
                new_std = 0
            self.dists[i] = stat.norm(
                loc=params['loc']+random_change_m,
                scale=params['scale']
            )

if __name__ == '__main__':
    # np.random.seed(123)
    n_steps = 5000
    n_dist = 10
    e = 0.1
    
    res_simple = []
    res_weighted = []    
    for i in range(1):
        bandit = MultiArmedBandit(n_steps, e, n_dist)
        bandit.call_sample_avg(False)
        
        print('Distributions args:', [d.kwds for d in bandit.dists])
        print('\nAction values:', bandit.Q)
        print('\nNumber of action choices:', bandit.N)
        final_reward = np.sum(bandit.final_rewards)
        res_simple.append(final_reward)
        
        plt.plot(range(n_steps), np.cumsum(bandit.final_rewards))
        plt.title(f'Cum sum of rewards - sample avg, {final_reward}')
        plt.xlabel('step')
        plt.ylabel('Sum')
        plt.show()
        plt.close()
        
        bandit = MultiArmedBandit(n_steps, e, n_dist)
        bandit.call_weighted_avg(False, 0.1, decay_step_size=False)
        final_reward = np.sum(bandit.final_rewards)
        res_weighted.append(final_reward)
        
        print('\nAction values:', bandit.Q)
        print('\nNumber of action choices:', bandit.N)
        plt.plot(range(n_steps), np.cumsum(bandit.final_rewards))
        plt.title(f'Cum sum of rewards - weighted avg, {final_reward}')
        plt.xlabel('step')
        plt.ylabel('Sum')
        plt.show()
        plt.close()
        
"""
Takze moje zistenie je ze vo vseobecnosti (tychto podmienok) je sample mean 
stabilnejsi a trade off var a proftu nestoji za konstantny step size. Urcite nie
s postupnym znizovanim. Pri konstantnom cez vsetky casove kroky vie byt vykonnejsi. 
Takze na to aby tato estimacia bola lepsia treba dost specificke podmienky plus spravne 
nastavit epsilon a step size.

Lepsi vykon sa prejavi hlave na long rune pri nestacionarnych problemoch.
"""
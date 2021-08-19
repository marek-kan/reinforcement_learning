# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 05:13:36 2021

@author: Marek
"""

import numpy as np

class State():
    def __init__(self, _id):
        self.value = 0
        self.id = _id
        # boundaries of row where is given state
        self.left_bound = max(1, (self.id//4)*4)
        self.right_bound = min(14, (self.id//4)*4+3)
        # possible next states
        self.next_S = [self.move('L'), self.move('R'), self.move('U'), self.move('D')]
        
    def move(self, a):
        if a == 'L':
            if self.id -1 >= self.left_bound:
                return self.id -1
            elif self.id -1 == 0:
                return 0
            else:
                return self.id
        if a == 'R':
            if self.id +1 <= self.right_bound:
                return self.id +1
            elif self.id +1 == 15:
                return 0
            else:
                return self.id
        if a == 'U':
            if self.id -4 > 0:
                return self.id - 4
            elif self.id -4 == 0:
                return 0
            else:
                return self.id
        if a == 'D':
            if self.id +4 < 15:
                return self.id + 4
            elif self.id +4 == 15:
                return 0
            else:
                return self.id
            
    def update(self, S, gamma):
        """
        Constant reward -1. There are only probs 1 or 0 so update rule is in 
        simplified version. (and without policy)
        """
        v = 0
        for i in range(len(self.next_S)):
            v += -1 + gamma * S[self.next_S[i]].value
        self.value = v
        return None
        
        
def train(max_iter=10, theta=0.1, gamma=0.7):
    S = {0: State(0)} # Terminal
    for i in range(1, 15):
        S[i] = State(i)
    S[15] = State(0) # Terminal
    
    delta = 999
    # for _iter in range(max_iter):
    while delta > theta:
        v = []
        V = []
        for j in range(1, 15): # loop in S
            v.append(S[j].value)
            S[j].update(S, gamma)
            V.append(S[j].value)
        delta = max(abs(np.array(v) - np.array(V)))
    return S, delta
        
S, d = train(gamma=0.1, theta=1e-4)
arr = []
for s in range(len(S)):
    arr.append(S[s].value)
    #print(S[s].value)  
arr = np.array(arr).reshape(4, 4)
print(arr)
    
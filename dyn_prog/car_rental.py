import numpy as np
import pickle
"""
Jack manages two locations for a nationwide car
rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. Cars become available for
renting the day after they are returned. To help ensure that cars are available where
they are needed, Jack can move them between the two locations overnight, at a cost of
$2 per car moved. We assume that the number of cars requested and returned at each
location are Poisson random variables, meaning that the probability that the number is
n is n
n! e, where  is the expected number. Suppose  is 3 and 4 for rental requests at
the first and second locations and 3 and 2 for returns. To simplify the problem slightly,
we assume that there can be no more than 20 cars at each location (any additional cars
are returned to the nationwide company, and thus disappear from the problem) and a
maximum of five cars can be moved from one location to the other in one night. We take
the discount rate to be  = 0.9 and formulate this as a continuing finite MDP, where
the time steps are days, the state is the number of cars at each location at the end of
the day, and the actions are the net numbers of cars moved between the two locations
overnight. Figure 4.2 shows the sequence of policies found by policy iteration starting
from the policy that never moves any cars
"""

def poisson_calc(_lambda):
    """
    Calculates probability for cars/ returns. From no car rented/returned up to maximum
    20 cars.

    Parameters
    ----------
    _lambda : int
        Expected number.

    Returns
    -------
    res : Dict
        Dictionary of probabilities.

    """
    res = {}
    for i in range(21):
        res[i] = (_lambda**i/np.math.factorial(i)) * np.exp(-_lambda)
    return res

def init_prob():
    """
    Calculates joint probability for rentals and returns.
    For Example probability of renting 3 cars and returning 2 at station A 
    and renting 1 car, returning 0 at station B.

    Returns
    -------
    joint_prob : Dict
        Joint probability.

    """
    rental_A = poisson_calc(3)
    rental_B = poisson_calc(4)
    return_A = poisson_calc(3)
    return_B = poisson_calc(2)
    
    prob_a = {}
    prob_b = {}
    for i in range(21):
        for j in range(21):
            prob_a[(i, j)] = rental_A[i] * return_A[j]
            prob_b[(i, j)] = rental_B[i] * return_B[j]
    
    joint_prob = {}
    for i in range(21):
        for j in range(21):
            for k in range(21):
                for l in range(21):
                    joint_prob[(i, j, k, l)] = (
                        prob_a[i, j] * prob_b[k, l]
                    )
    return joint_prob

def P_calculate(joint_prob):
    """
    Calculates state - action - reward triplets.  

    Parameters
    ----------
    joint_prob : Dict
        Joint prob of rental/return at A/B occurs.

    Returns
    -------
    P : Dict; 
        {
            ((state_value_A, state_value_B), a): {
                ((value_A_Changed, value_B_Changed), reward): prob
            }
        }
        State - action - reward dict.

    """
    
    for state_value_A in range(21):  # car left at the start of the day at A
        print("State " + str(state_value_A))
        for state_value_B in range(21):  # car left at the start of the day at B
            P = {}
            for action in range(-5, 6):  # action range is from -5 to 5
                temp = {}
                #problem: action=-5 A=1 B=10
                if action <= state_value_A and -action <= state_value_B and \
                    action + state_value_B <= 20 and -action + state_value_A <= 20:
                    for customer_A in range(21):  # total customers come at the end of the day at A
                        for customer_B in range(21):  # total customers come at the end of the day at B
                            for returned_car_A in range(21):  # total cars returned at the end of the day at A
                                for returned_car_B in range(21):  # total cars returned at the end of the day at B
                                    rented_cars_a = min(customer_A, state_value_A-action) # lets say 20 customers and 1-(-5)
                                    rented_cars_b = min(customer_B, state_value_B+action)
                                    
                                    value_A_Changed = min(20, state_value_A - rented_cars_a + returned_car_A - action)
                                    value_B_Changed = min(20, state_value_B - rented_cars_b + returned_car_B + action)
                                    
                                    reward = 10 * rented_cars_a + \
                                             10 * rented_cars_b - \
                                             abs(action) * 2  # the reason for action here is the current action change the next stroes
                                    
                                    temp[((value_A_Changed, value_B_Changed),reward)] = temp.get(
                                        (value_A_Changed, value_B_Changed),0) # init with 0
                                    temp[((value_A_Changed, value_B_Changed),reward)]+= joint_prob[
                                        (customer_A, returned_car_A, customer_B, returned_car_B)]
                    P.update({(state_value_A, state_value_B, action):temp})
                with open(f'P_{state_value_A}_{state_value_B}.pkl', 'wb') as f:
                    pickle.dump(P, f, protocol=-1)
    return 

def policy_eval(V, policy, P, 
                theta=0.01, gamma=0.9):
    """
    Computes value functions for given policy. "Prediction problem"

    Parameters
    ----------
    V : Dict
        Value function.
    policy : Dict
        Policy.
    P : Dict
        State - action - reward dict.
    theta : float, optional
        Tolerance of approx. The default is 0.01.
    gamma : float, optional
        Discounting rate in value function estimate. The default is 0.9.

    Returns
    -------
    V : Dict
        Value function.

    """
    counter=0
    while True:
        delta = 0
        for i in range(21):
            for j in range(21):
                a = policy[(i, j)]
                P = pickle.load(open(f'P_{i}_{j}.pkl', 'rb'))
                p = P[(i, j, a)]
                old_value = V[(i, j)]
                
                V[i, j] = 0
                for k, v in p.items():
                    states, reward = k
                    prob = v
                    V[i, j] += prob * (reward + gamma*V[states])
                delta = max(delta, abs(V[i,j]-old_value))
        if delta < theta:
            return V
        counter += 1
        
def policy_improvement(V, P, policy={}, gamma=0.9):
    """
    Improves policy based on original value function

    Parameters
    ----------
    V : Dict
        Value function.
    policy : Dict
        Policy.
    P : Dict
        State - action - reward dict.
    gamma : float, optional
        Discounting rate in value function estimate. The default is 0.9.

    Returns
    -------
    policy : Dict
        Improved Policy.

    """
    counter = 0
    while True:
        policy_stable=True
        for k, old_action in policy.items():
            q = [0] * 11
            state_a, state_b = k
            for a in range(-5, 6):
                P = pickle.load(open(f'P_{state_a}_{state_b}.pkl', 'rb'))
                p = P[(state_a, state_b, a)]
                idx = a + 5
                if (a <= state_a and -a <= state_b and a + state_b <= 20 and -a + state_a <= 20):
                    for states, reward, prob in p.items():
                        q[idx] = prob * (reward + gamma*V[states])
                else:
                    q[idx] = -999 # not possible action
            policy[k] = np.argmax(q) - 5
            if policy[k] != old_action:
                policy_stable = False
        if policy_stable:
            return policy
    return

def train():
    # joint_prob = init_prob()
    # P = P_calculate(joint_prob)
    # P = pickle.load(open('P.pkl', 'rb'))
    # init value functions and policy
    V, policy = {}, {}
    for i in range(21):
        for j in range(21):
            V[(i, j)] = 10 * np.random.random()
            policy[(i, j)] = 0
            
    for i in range(5):
        print('Training step:', i+1)
        V = policy_eval(V, policy)
        policy = policy_improvement(V, policy)
        
    return V, policy
    
    
    
# V, policy = train()
joint_prob = init_prob()
P_calculate(joint_prob)
    
    
    
    
    
    
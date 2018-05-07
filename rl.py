##Cecilia Ferrando
##HW1

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import random
from queue import PriorityQueue


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    nS = env.nS
    nA = env.nA
    #initialize policy vector
    policy = np.zeros(nS,dtype='int')
    
    for state in range(nS):
        values = np.zeros(nA)
        
        for action in range(nA):
            possibleOutcomes = env.P[state][action]
            
            for outcome in possibleOutcomes:
                values[action] += outcome[0]*(outcome[2]+(1-int(outcome[3]))*gamma*(value_function[outcome[1]]))
        policy[state] = np.argmax(values)
        
    return policy

def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    nS = env.nS
    Vold = np.zeros(nS)
    Vnew = np.zeros(nS)
    i = 0
    delta = tol
    
   
    while i<max_iterations and delta>=tol:
        
        delta = 0
        
        for state in range(nS):
            
            poxOutcomes = env.P[state][policy[state]]
            Vnew[state]=0
            
            for outcome in poxOutcomes:   #value of state
                Vnew[state] += outcome[0] * (outcome[2]+(1-int(outcome[3]))*gamma*(Vold[outcome[1]]))
                
        delta = max(delta,abs(Vnew[state]-Vold[state]))
        
        i += 1
        Vold = Vnew
    
    return Vold,i
    
    

def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    V = np.zeros(env.nS)
    i = 0
    delta = tol
    nS = env.nS
    
    while i<max_iterations and delta>=tol:
        
        delta = 0
        
        for state in range(nS):
            
            v = V[state]    #allocating
            poxOutcomes = env.P[state][policy[state]]
            V[state]=0
            
            for outcome in poxOutcomes:   #value of state
                V[state] += outcome[0] * (outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
            delta = max(delta,abs(V[state]-v))
            
        i += 1
            
    return V,i
    


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    V = np.zeros(env.nS)
    i = 0
    delta = tol
    nS = env.nS
    
    while i<max_iterations and delta>=tol:
        
        delta = 0
        #indexes = list(range(nS))
        indexes = np.random.permutation(nS)
        
        for index in (indexes):
            state = index
            v = V[state]    #allocating
            poxOutcomes = env.P[state][policy[state]]
            V[state]=0
            
            for outcome in poxOutcomes:   #value of state
                V[state] += outcome[0] * (outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
            delta = max(delta,abs(V[state]-v))
            
            i += 1
            
    return V,i
    


def evaluate_policy_async_custom(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluate the value of a policy. Updates states by a student-defined
    heuristic. 

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    #attempting prioritized sweeping

    V = np.zeros(env.nS)
    Vnew = np.zeros(env.nS)
    diff = np.zeros(env.nS)
    i = 0
    delta = tol
    states = env.nS
    
    priority_queue = PriorityQueue()
  
    while i<max_iterations and delta>=tol:
        
        delta = 0
        
        for state in range(states):
            
            v = V[state]
            poxOutcomes = env.P[state][policy[state]]
            V[state]=0
            
            for outcome in poxOutcomes:   #value of state
                Vnew[state] += outcome[0] * (outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
            delta = max(delta,abs(Vnew[state]-v))
            diff[state] = delta           #update vector of differences
            i += 1
            
        priorityStates = [state for _,state in sorted(zip(diff,range(states)), key=lambda x: x[0])]
        weightMax = states
        
        for state in priorityStates:
            priority_queue.put((-weightMax, state))
            weightMax -= 1
        
        for state in range(states):
            stateObject = priority_queue.get()
            currentWeight, currentState = stateObject
            V[state] = Vnew[currentState]
            
    return V,i
    


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    changed = False
    nS = env.nS
    nA = env.nA
    
    for s in range(nS):
        values = np.zeros(nA)
        
        for a in range(nA):
            possibleOutcomes = env.P[s][a]
            
            for outcome in possibleOutcomes:
                values[a] += outcome[0] * (outcome[2] + (1-int(outcome[3]))*gamma*(value_func[outcome[1]]))
        foundPolicy = np.argmax(values) 
        
        if foundPolicy != policy[s]:
            policy[s] = foundPolicy
            changed = True
            
    return changed, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    changed = True
    
    policy = np.zeros(env.nS, dtype='int')
    valueFunction = np.zeros(env.nS)
    valueIters, policyIters = 0, 0
    
    while changed == True:
        
        valueFunction,iters = evaluate_policy_sync(env, gamma, policy, max_iterations, tol)
        changed, policy = improve_policy(env, gamma, valueFunction, policy)
        valueIters += iters
        policyIters += 1
        
    return policy, valueFunction, policyIters, valueIters


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    changed = True
    
    policy = np.zeros(env.nS, dtype='int')
    valueFunction = np.zeros(env.nS)
    valueIters, policyIters = 0, 0
    
    while changed == True:
        
        valueFunction,iters = evaluate_policy_async_ordered(env, gamma, policy, max_iterations, tol)
        changed, policy = improve_policy(env, gamma, valueFunction, policy)
        valueIters += iters
        policyIters += 1
        
    return policy, valueFunction, policyIters, valueIters


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    changed = True
    
    policy = np.zeros(env.nS, dtype='int')
    valueFunction = np.zeros(env.nS)
    valueIters, policyIters = 0, 0
    
    while changed == True:
        
        valueFunction,iters = evaluate_policy_async_randperm(env, gamma, policy, max_iterations, tol)
        changed, policy = improve_policy(env, gamma, valueFunction, policy)
        valueIters += iters
        policyIters += 1
        
    return policy, valueFunction, policyIters, valueIters


def policy_iteration_async_custom(env, gamma, max_iterations=int(1e3),
                                  tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_custom methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    changed = True
    
    policy = np.zeros(env.nS, dtype='int')
    valueFunction = np.zeros(env.nS)
    valueIters, policyIters = 0, 0
    
    while changed == True:
        
        valueFunction,iters = evaluate_policy_async_custom(env, gamma, policy, max_iterations, tol)
        changed, policy = improve_policy(env, gamma, valueFunction, policy)
        valueIters += iters
        policyIters += 1
        
    return policy, valueFunction, policyIters, valueIters

def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    delta = tol
    nS = env.nS
    V = np.zeros(nS)
    i = 0
    
    while delta>=tol and i<max_iterations:
        delta = 0
        
        for s in range(env.nS):
            oldValue = V[s]
            vals = np.zeros(env.nA)
            
            for action in range(env.nA):
                possibleOutcomes = env.P[s][action]
                vals[action]=0
                
                for outcome in possibleOutcomes:
                    vals[action] += outcome[0]*(outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
                    
            V[s] = max(vals)
            delta = max(abs(V[s]-oldValue),delta)
            i += 1

    return V,i


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    delta = tol
    nS = env.nS
    V = np.zeros(nS)
    i = 0
    
    while delta>=tol and i<max_iterations:
        
        delta = 0
        
        for s in range(env.nS):
            oldValue = V[s]
            vals = np.zeros(env.nA)
            
            for action in range(env.nA):
                possibleOutcomes = env.P[s][action]
                vals[action]=0
                
                for outcome in possibleOutcomes:
                    vals[action] += outcome[0]*(outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
                    
            V[s] = max(vals)
            delta = max(abs(V[s]-oldValue),delta)
            i += 1

    return V,i


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    delta = tol
    nS = env.nS
    V = np.zeros(nS)
    i = 0
    
    while delta>=tol and i<max_iterations:
        delta = 0
        
        indexes = np.random.permutation(nS)
        
        for index in (indexes):
            state = index
            oldValue = V[state]
            values = np.zeros(env.nA)
            
            for action in range(env.nA):
                possibleOutcomes = env.P[state][action]
                print(possibleOutcomes)
                values[action]=0
                
                for outcome in possibleOutcomes:
                    values[action] += outcome[0]*(outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
                    
            V[state] = max(values)
            delta = max(abs(V[state]-oldValue),delta)
            i += 1

    return V,i


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    # delta = tol
    # states = env.nS
    # V = np.zeros(nS)
    # i = 0
    # 
    # predecessors = {}
    # for state in states:
    #     predecessors[state] = set() 
    # 
    # while delta>=tol and i<max_iterations:
    #     delta = 0
    #     for state in states:
    #         oldValue = V[s]
    #         values = np.zeros(env.nA)
    #         for action in range(env.nA):
    #             possibleOutcomes = env.P[s][action]
    #             values[action]=0
    #             for outcome in possibleOutcomes:
    #                 values[action] += outcome[0]*(outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
    #         V[s] = max(values)
    #         delta = max(abs(V[s]-oldValue),delta)
    #         i += 1

   #   return V,i
    
    V = np.zeros(env.nS)
    Vnew = np.zeros(env.nS)
    diff = np.zeros(env.nS)
    i = 0
    delta = tol
    states = env.nS
    
    predecessors = {}
    for state in states:
        predecessors[state] = set() 
    
    priority_queue = PriorityQueue()

    delta = 0
    
    for state in range(states):
        
        v = V[state]
        values = np.zeros(env.nA)
        for action in range(env.nA):
            poxOutcomes = env.P[state][policy[state]]
            values[action]=0
            for outcome in poxOutcomes:   #value of state
                values[action] += outcome[0]*(outcome[2]+(1-int(outcome[3]))*gamma*(V[outcome[1]]))
        Vnew[state] = max(values)
        delta = max(delta,abs(Vnew[state]-v))
        diff[state] = delta           #update vector of differences

        
        priority_queue.put((-diff[state], state))
            
    while i<max_iterations and delta>=tol:
        if priority_queue.isEmpty():
            break
        #priorityStates = [state for _,state in sorted(zip(diff,range(states)), key=lambda x: x[0])]
        
        state = priority_queue.get()
        V[state] = Vnew[state]    
        
        for predecessor in predecessors[state]:
            diff = abs(V[predecessor] - Vnew[predecessor])
            if diff > tol:
              priority_queue.update(predecessor, -diff)
            
    return V,i
